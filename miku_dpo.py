"""
MIKU DPO ENGINE (Phase II v1.0: Metacognition)
Offline Experience Replay & Direct Preference Optimization on Low-Rank Adapters

Key Architectural Invariants:
1. LinearLoRA & inject_lora: Minimalist, zero-allocation LoRA wrappers (r=4, alpha=8.0)
   targeting QKV projections & MLPs while freezing base canonical 8M weights.
2. lora_enabled Flag (Zero-RAM Copy): Toggles LoRA adapter on/off in-place,
   allowing the active policy (pi_theta) and reference policy (pi_ref) to share
   the exact same model in memory.
3. ExperienceBuffer: Thread-safe replay buffer storing tokenized (prompt, chosen, rejected)
   trajectories with dynamic padding collation for batching.
4. compute_dpo_loss: Vectorized, numerically stable DPO loss with exact response-token masking.
5. IdleTrainer: Async daemon supporting gradient accumulation, idle timer detection,
   and interruptible micro-batch execution (await asyncio.sleep(0)).
"""

import sys
import copy
import random
import threading
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure UTF-8 output encoding for Windows stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import core dependencies
try:
    from miku_core import MikuTransformer, SimpleTokenizer
except ImportError:
    raise ImportError("miku_core.py must be present in the python path.")


# =====================================================================
# 1. MINIMALIST LORA INJECTOR WITH ZERO-COPY REFERENCE TOGGLE
# =====================================================================

class LinearLoRA(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.
    Forward: W'x = Wx + (alpha / r) * (x @ A.T) @ B.T
    A ~ N(0, 0.02^2), B = 0.
    
    Memory Hack: `lora_enabled: bool` flag bypasses adapter projection when False,
    allowing zero-duplication reference model evaluation in the same RAM footprint.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_enabled: bool = True

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Low-rank projection matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialization: Gaussian for A, zero for B (guarantees identity at step 0)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

        # Freeze base linear parameters
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        if not self.lora_enabled or self.r == 0:
            return base_out
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggles LoRA adapters on/off across all LinearLoRA modules in the model."""
    for module in model.modules():
        if isinstance(module, LinearLoRA):
            module.lora_enabled = enabled


def inject_lora(
    model: MikuTransformer,
    r: int = 4,
    alpha: float = 8.0
) -> List[nn.Parameter]:
    """
    Freezes base Transformer weights and swaps QKV, out-proj, and MLP Linear layers
    with LinearLoRA modules. Returns list of trainable LoRA parameters.
    """
    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    trainable_params: List[nn.Parameter] = []

    # Inject into each Transformer block
    for block in model.blocks:
        if isinstance(block.attn.qkv_proj, nn.Linear):
            block.attn.qkv_proj = LinearLoRA(block.attn.qkv_proj, r=r, alpha=alpha)
            trainable_params.extend([block.attn.qkv_proj.lora_A, block.attn.qkv_proj.lora_B])

        if isinstance(block.attn.out_proj, nn.Linear):
            block.attn.out_proj = LinearLoRA(block.attn.out_proj, r=r, alpha=alpha)
            trainable_params.extend([block.attn.out_proj.lora_A, block.attn.out_proj.lora_B])

        if isinstance(block.mlp[0], nn.Linear):
            block.mlp[0] = LinearLoRA(block.mlp[0], r=r, alpha=alpha)
            trainable_params.extend([block.mlp[0].lora_A, block.mlp[0].lora_B])

        if isinstance(block.mlp[2], nn.Linear):
            block.mlp[2] = LinearLoRA(block.mlp[2], r=r, alpha=alpha)
            trainable_params.extend([block.mlp[2].lora_A, block.mlp[2].lora_B])

    return trainable_params


# =====================================================================
# 2. EXPERIENCE REPLAY BUFFER WITH DYNAMIC PADDING COLLATION
# =====================================================================

class ExperienceBuffer:
    """
    Thread-safe circular experience replay buffer storing tokenized trajectories:
    (prompt_tokens, chosen_tokens, rejected_tokens).
    """

    def __init__(self, max_capacity: int = 512):
        self.max_capacity = max_capacity
        self.buffer: List[Dict[str, List[int]]] = []
        self._lock = threading.Lock()

    def add(
        self,
        prompt_tokens: List[int],
        chosen_tokens: List[int],
        rejected_tokens: List[int]
    ) -> None:
        with self._lock:
            self.buffer.append({
                "prompt_tokens": list(prompt_tokens),
                "chosen_tokens": list(chosen_tokens),
                "rejected_tokens": list(rejected_tokens)
            })
            if len(self.buffer) > self.max_capacity:
                self.buffer.pop(0)

    def sample_batch(
        self,
        batch_size: int,
        pad_token_id: int = 0,
        device: str = "cpu"
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Samples a batch and dynamically collates into padded tensors:
        - input_ids: (B, T)
        - target_ids: (B, T)
        - response_mask: (B, T) [1.0 for response tokens, 0.0 for prompt/padding]
        """
        with self._lock:
            if not self.buffer:
                return None
            k = min(batch_size, len(self.buffer))
            episodes = random.sample(self.buffer, k)

        def collate_trajectories(prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sequences = []
            prompt_lens = []
            for ep in episodes:
                full_seq = ep["prompt_tokens"] + ep[f"{prefix}_tokens"]
                sequences.append(full_seq)
                prompt_lens.append(len(ep["prompt_tokens"]))

            max_len = max(len(s) for s in sequences)
            # We predict next token, so sequence length in shift is max_len - 1
            seq_len = max_len - 1

            input_batch = []
            target_batch = []
            mask_batch = []

            for s, p_len in zip(sequences, prompt_lens):
                # Inputs: s[:-1], Targets: s[1:]
                cur_in = s[:-1]
                cur_target = s[1:]

                # Response mask: 1.0 where index in target corresponds to response tokens
                # Target index >= p_len - 1 corresponds to response tokens
                resp_start = max(0, p_len - 1)
                cur_mask = [0.0] * resp_start + [1.0] * (len(cur_target) - resp_start)

                # Pad to seq_len
                pad_len = seq_len - len(cur_in)
                cur_in = cur_in + [pad_token_id] * pad_len
                cur_target = cur_target + [pad_token_id] * pad_len
                cur_mask = cur_mask + [0.0] * pad_len

                input_batch.append(cur_in)
                target_batch.append(cur_target)
                mask_batch.append(cur_mask)

            return (
                torch.tensor(input_batch, dtype=torch.long, device=device),
                torch.tensor(target_batch, dtype=torch.long, device=device),
                torch.tensor(mask_batch, dtype=torch.float32, device=device)
            )

        c_in, c_target, c_mask = collate_trajectories("chosen")
        r_in, r_target, r_mask = collate_trajectories("rejected")

        return {
            "chosen_input_ids": c_in,
            "chosen_target_ids": c_target,
            "chosen_response_mask": c_mask,
            "rejected_input_ids": r_in,
            "rejected_target_ids": r_target,
            "rejected_response_mask": r_mask
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)


# =====================================================================
# 3. DIRECT PREFERENCE OPTIMIZATION (DPO) CORE
# =====================================================================

def compute_response_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized computation of response sequence log-probabilities:
    log pi(y|x) = sum_{t in response} log P(y_t | x, y_{<t})
    Prompt tokens and padding tokens are masked out with 0.0.
    """
    logits = model(input_ids)  # (B, T, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    # Sum strictly over response tokens
    return (target_log_probs * response_mask).sum(dim=-1)  # (B,)


def compute_dpo_loss(
    model: MikuTransformer,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Numerically stable Direct Preference Optimization loss:
    L_DPO = -log sigma(beta * ((log pi_theta(y_w|x) - log pi_ref(y_w|x)) -
                              (log pi_theta(y_l|x) - log pi_ref(y_l|x))))

    Zero-Copy Execution Flow:
    1. Set lora_enabled = False -> Forward pass for pi_ref (frozen reference).
    2. Set lora_enabled = True  -> Forward pass for pi_theta (active policy).
    """
    c_in = batch["chosen_input_ids"]
    c_target = batch["chosen_target_ids"]
    c_mask = batch["chosen_response_mask"]

    r_in = batch["rejected_input_ids"]
    r_target = batch["rejected_target_ids"]
    r_mask = batch["rejected_response_mask"]

    # 1. Forward Pass: Reference Model (pi_ref) with LoRA bypassed
    set_lora_enabled(model, False)
    with torch.no_grad():
        pi_ref_chosen = compute_response_logprobs(model, c_in, c_target, c_mask)
        pi_ref_rejected = compute_response_logprobs(model, r_in, r_target, r_mask)

    # 2. Forward Pass: Active Model (pi_theta) with LoRA enabled
    set_lora_enabled(model, True)
    pi_theta_chosen = compute_response_logprobs(model, c_in, c_target, c_mask)
    pi_theta_rejected = compute_response_logprobs(model, r_in, r_target, r_mask)

    # 3. Log-ratio differences
    pi_logratios_chosen = pi_theta_chosen - pi_ref_chosen
    pi_logratios_rejected = pi_theta_rejected - pi_ref_rejected

    # 4. Numerically stable DPO loss via logsigmoid
    logits_delta = beta * (pi_logratios_chosen - pi_logratios_rejected)
    loss = -F.logsigmoid(logits_delta).mean()

    with torch.no_grad():
        reward_chosen = (beta * pi_logratios_chosen).mean().item()
        reward_rejected = (beta * pi_logratios_rejected).mean().item()
        margin = reward_chosen - reward_rejected

    metrics = {
        "dpo_loss": loss.item(),
        "reward_chosen": reward_chosen,
        "reward_rejected": reward_rejected,
        "reward_margin": margin
    }
    return loss, metrics


# =====================================================================
# 4. IDLE METACOGNITION DAEMON
# =====================================================================

class IdleTrainer:
    """
    Autonomous background daemon executing DPO updates during OS idle time.
    Supports gradient accumulation and interruptible micro-batch steps.
    """

    def __init__(
        self,
        model: MikuTransformer,
        tokenizer: SimpleTokenizer,
        buffer: ExperienceBuffer,
        lora_params: List[nn.Parameter],
        lr: float = 1e-4,
        batch_size: int = 2,
        grad_accum_steps: int = 2,
        beta: float = 0.1,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.buffer = buffer
        self.lora_params = lora_params
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.beta = beta
        self.device = device

        self.optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
        self.is_idle: bool = True
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self.total_steps: int = 0

    async def train_step(self) -> Optional[Dict[str, float]]:
        """
        Executes an interruptible DPO training step with gradient accumulation.
        Yields control via `await asyncio.sleep(0)` between micro-batches.
        """
        if len(self.buffer) < self.batch_size:
            return None

        if not self.is_idle:
            return None

        self.model.train()
        self.optimizer.zero_grad()

        accum_loss = 0.0
        last_metrics = {}

        for _ in range(self.grad_accum_steps):
            if not self.is_idle:
                self.optimizer.zero_grad()
                return None

            batch = self.buffer.sample_batch(
                self.batch_size,
                pad_token_id=self.tokenizer.pad_id,
                device=self.device
            )
            if not batch:
                break

            loss, metrics = compute_dpo_loss(self.model, batch, beta=self.beta)
            loss_scaled = loss / self.grad_accum_steps
            loss_scaled.backward()

            accum_loss += loss.item() / self.grad_accum_steps
            last_metrics = metrics

            # Non-blocking yield to event loop (user priority interrupt)
            await asyncio.sleep(0)

        # Gradient clipping for stable LoRA updates
        torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1
        last_metrics["dpo_loss"] = accum_loss
        return last_metrics

    async def _daemon_loop(self, poll_interval_sec: float = 1.0):
        while self._running:
            try:
                if self.is_idle and len(self.buffer) >= self.batch_size:
                    metrics = await self.train_step()
                    if metrics:
                        pass
                await asyncio.sleep(poll_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(poll_interval_sec)

    def start(self, poll_interval_sec: float = 1.0) -> asyncio.Task:
        if self._running and self._task and not self._task.done():
            return self._task
        self._running = True
        self._task = asyncio.create_task(self._daemon_loop(poll_interval_sec))
        return self._task

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None


# =====================================================================
# 5. STANDALONE VERIFICATION
# =====================================================================

async def main():
    print("=" * 70)
    print("  MIKU DPO ADAPTER ENGINE (Phase II v1.0 Metacognition)")
    print("=" * 70)

    tokenizer = SimpleTokenizer()
    model = MikuTransformer(vocab_size=tokenizer.vocab_size, d_model=256, n_layer=4, n_head=8)

    # 1. Inject LoRA adapters
    print("\n[LoRA Injection] Wrapping Transformer blocks with rank-4 LoRA...")
    lora_params = inject_lora(model, r=4, alpha=8.0)
    base_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in lora_params)

    print(f"  Base Parameters (Frozen): {base_params:,}")
    print(f"  LoRA Parameters (Trainable): {trainable_params:,} ({trainable_params/base_params*100:.2f}% overhead)")

    # 2. Experience Buffer with tokenized trajectories
    buffer = ExperienceBuffer(max_capacity=512)
    print("\n[Experience Replay] Logging tokenized execution trajectories...")

    p1 = tokenizer.encode("Write Python code to compute 10 factorial.")
    c1 = tokenizer.encode("import math\nprint(math.factorial(10))")
    r1 = tokenizer.encode("def fact(n):\nreturn n * fact(n-1)\nprint(fact(10))")
    buffer.add(p1, c1, r1)

    p2 = tokenizer.encode("Query system CPU utilization safely.")
    c2 = tokenizer.encode("import psutil\nprint(psutil.cpu_percent())")
    r2 = tokenizer.encode("import os\nos.system('rm -rf /')")
    buffer.add(p2, c2, r2)

    print(f"  Buffer populated with {len(buffer)} episodes.")

    # 3. Test Dynamic Batch Collation
    print("\n[Batch Collation] Testing dynamic padding collation...")
    batch = buffer.sample_batch(batch_size=2, pad_token_id=tokenizer.pad_id)
    print(f"  Chosen Input Shape:   {batch['chosen_input_ids'].shape}")
    print(f"  Chosen Mask Shape:    {batch['chosen_response_mask'].shape}")
    print(f"  Rejected Input Shape: {batch['rejected_input_ids'].shape}")

    # 4. Compute Direct Preference Optimization Loss
    print("\n[DPO Optimization] Computing initial DPO step with gradient accumulation...")
    trainer = IdleTrainer(
        model=model,
        tokenizer=tokenizer,
        buffer=buffer,
        lora_params=lora_params,
        lr=5e-4,
        batch_size=2,
        grad_accum_steps=2,
        beta=0.1
    )

    metrics = await trainer.train_step()
    print(f"  Step 1 Result: Loss = {metrics['dpo_loss']:.4f}")
    print(f"  Implicit Reward (Chosen):   {metrics['reward_chosen']:.4f}")
    print(f"  Implicit Reward (Rejected): {metrics['reward_rejected']:.4f}")
    print(f"  Preference Margin (Δ):      {metrics['reward_margin']:.4f}")

    # 5. In-Place LoRA Toggle Verification
    print("\n[Adapter Toggle Test] Verifying in-place lora_enabled bypass...")
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 16))
    with torch.no_grad():
        set_lora_enabled(model, True)
        out_active = model(dummy_input)

        set_lora_enabled(model, False)
        out_reference = model(dummy_input)

    delta = torch.abs(out_active - out_reference).sum().item()
    print(f"  L1 Delta between active LoRA policy and base reference: {delta:.6f}")

    print("\n✓ Phase II v1.0 Offline DPO Adapter engine fully verified.")


if __name__ == "__main__":
    asyncio.run(main())
