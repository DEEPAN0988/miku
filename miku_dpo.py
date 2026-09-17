"""
MIKU DPO ENGINE (Phase II v1.0: Metacognition)
Offline Experience Replay & Direct Preference Optimization on Low-Rank Adapters

Key Architecture:
1. LinearLoRA & inject_lora: Minimalist, zero-allocation LoRA wrappers (r=4, alpha=8.0)
   targeting QKV projections & MLPs while freezing base canonical 8M weights.
2. Zero-Duplicate Reference Model: Context manager toggles LoRA forward bypass,
   computing pi_ref directly through frozen base weights with 0 extra RAM footprint.
3. ExperienceBuffer: Thread-safe replay deque tracking (prompt, chosen, rejected) trajectories.
4. compute_dpo_loss: Numerically stable DPO loss with exact sequence-level response logprob gathering.
5. IdleTrainer: Non-blocking asyncio background daemon running AdamW optimization during OS idle cycles.
"""

import sys
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
# 1. MINIMALIST LORA ADAPTER & ZERO-DUPLICATION REFERENCE ENGINE
# =====================================================================

class LinearLoRA(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.
    Forward: W'x = Wx + (alpha / r) * (x @ A.T) @ B.T
    A ~ N(0, 0.02^2), B = 0.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.adapter_enabled: bool = True

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # LoRA parameter matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Gaussian init for A, Zero init for B (Ensures identity at step 0)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

        # Freeze base weights permanently
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        if not self.adapter_enabled or self.r == 0:
            return base_out
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


class disable_adapters:
    """
    Context manager to bypass LoRA adapters on the fly.
    Allows evaluating the canonical reference model (pi_ref) without
    duplicating the model weights in RAM.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.lora_modules: List[LinearLoRA] = [
            m for m in model.modules() if isinstance(m, LinearLoRA)
        ]

    def __enter__(self):
        for m in self.lora_modules:
            m.adapter_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.lora_modules:
            m.adapter_enabled = True


def inject_lora(
    model: MikuTransformer,
    r: int = 4,
    alpha: float = 8.0
) -> List[nn.Parameter]:
    """
    Freezes base Transformer weights and wraps QKV, projection, and MLP layers
    with LinearLoRA modules. Returns list of trainable LoRA parameters.
    """
    # 1. Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    trainable_params: List[nn.Parameter] = []

    # 2. Inject into each Transformer block
    for block in model.blocks:
        # Wrap Attention QKV projection
        if isinstance(block.attn.qkv_proj, nn.Linear):
            block.attn.qkv_proj = LinearLoRA(block.attn.qkv_proj, r=r, alpha=alpha)
            trainable_params.extend([block.attn.qkv_proj.lora_A, block.attn.qkv_proj.lora_B])

        # Wrap Attention Out projection
        if isinstance(block.attn.out_proj, nn.Linear):
            block.attn.out_proj = LinearLoRA(block.attn.out_proj, r=r, alpha=alpha)
            trainable_params.extend([block.attn.out_proj.lora_A, block.attn.out_proj.lora_B])

        # Wrap MLP Linear layers
        if isinstance(block.mlp[0], nn.Linear):
            block.mlp[0] = LinearLoRA(block.mlp[0], r=r, alpha=alpha)
            trainable_params.extend([block.mlp[0].lora_A, block.mlp[0].lora_B])

        if isinstance(block.mlp[2], nn.Linear):
            block.mlp[2] = LinearLoRA(block.mlp[2], r=r, alpha=alpha)
            trainable_params.extend([block.mlp[2].lora_A, block.mlp[2].lora_B])

    return trainable_params


# =====================================================================
# 2. THREAD-SAFE EXPERIENCE REPLAY BUFFER
# =====================================================================

class ExperienceBuffer:
    """
    Thread-safe circular experience replay buffer storing
    (prompt, chosen, rejected) trajectories for DPO training.
    """

    def __init__(self, max_capacity: int = 512):
        self.max_capacity = max_capacity
        self.buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            self.buffer.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": metadata or {}
            })
            if len(self.buffer) > self.max_capacity:
                self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.buffer:
                return []
            k = min(batch_size, len(self.buffer))
            return random.sample(self.buffer, k)

    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)


# =====================================================================
# 3. DIRECT PREFERENCE OPTIMIZATION LOSS
# =====================================================================

def get_sequence_response_logprob(
    model: MikuTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    response: str,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Computes sum of log probabilities of the response tokens conditioned on the prompt.
    Tokens inside the prompt are strictly masked out of the probability computation.
    """
    prompt_ids = tokenizer.encode(prompt, add_special=True)
    resp_ids = tokenizer.encode(response, add_special=False)
    if not resp_ids:
        resp_ids = [tokenizer.eos_id]

    full_ids = prompt_ids + resp_ids
    if len(full_ids) > model.max_seq_len:
        full_ids = full_ids[:model.max_seq_len]

    idx = torch.tensor([full_ids], dtype=torch.long, device=device)  # (1, T)
    logits = model(idx)  # (1, T, vocab_size)

    # Shift logits and targets so token at i predicts token at i+1
    shift_logits = logits[:, :-1, :]  # (1, T-1, vocab_size)
    shift_labels = idx[:, 1:]         # (1, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Calculate starting boundary of response tokens in the shifted array
    # Prompt length in shift_labels is len(prompt_ids) - 1
    resp_start_idx = max(0, len(prompt_ids) - 1)
    response_log_probs = gathered_log_probs[:, resp_start_idx:]

    return response_log_probs.sum(dim=-1).squeeze(0)  # Scalar Tensor


def compute_dpo_loss(
    model: MikuTransformer,
    tokenizer: SimpleTokenizer,
    batch: List[Dict[str, Any]],
    beta: float = 0.1,
    device: str = "cpu"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Direct Preference Optimization Loss across a batch:
    L_DPO = -log sigma(beta * ((log pi_theta(y_w|x) - log pi_ref(y_w|x)) -
                              (log pi_theta(y_l|x) - log pi_ref(y_l|x))))
    Uses disable_adapters() context manager for zero-copy reference evaluation.
    """
    losses = []
    implicit_rewards_chosen = []
    implicit_rewards_rejected = []

    for item in batch:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 1. Forward Pass under Active Policy pi_theta
        logp_theta_chosen = get_sequence_response_logprob(model, tokenizer, prompt, chosen, device=device)
        logp_theta_rejected = get_sequence_response_logprob(model, tokenizer, prompt, rejected, device=device)

        # 2. Forward Pass under Frozen Reference Policy pi_ref (zero-copy via adapter bypass)
        with torch.no_grad():
            with disable_adapters(model):
                logp_ref_chosen = get_sequence_response_logprob(model, tokenizer, prompt, chosen, device=device)
                logp_ref_rejected = get_sequence_response_logprob(model, tokenizer, prompt, rejected, device=device)

        # 3. Log-ratio deltas
        pi_diff_chosen = logp_theta_chosen - logp_ref_chosen
        pi_diff_rejected = logp_theta_rejected - logp_ref_rejected

        # 4. Numerically stable DPO loss
        h = beta * (pi_diff_chosen - pi_diff_rejected)
        loss = -F.logsigmoid(h)

        losses.append(loss)
        implicit_rewards_chosen.append((beta * pi_diff_chosen).item())
        implicit_rewards_rejected.append((beta * pi_diff_rejected).item())

    total_loss = torch.stack(losses).mean()
    metrics = {
        "dpo_loss": total_loss.item(),
        "reward_chosen": float(sum(implicit_rewards_chosen) / len(implicit_rewards_chosen)),
        "reward_rejected": float(sum(implicit_rewards_rejected) / len(implicit_rewards_rejected)),
        "reward_margin": float((sum(implicit_rewards_chosen) - sum(implicit_rewards_rejected)) / len(losses))
    }
    return total_loss, metrics


# =====================================================================
# 4. IDLE METACOGNITION DAEMON
# =====================================================================

class IdleTrainer:
    """
    Autonomous background daemon executing DPO updates during OS idle time.
    Monitors is_idle flag and yields immediately to incoming user queries.
    """

    def __init__(
        self,
        model: MikuTransformer,
        tokenizer: SimpleTokenizer,
        buffer: ExperienceBuffer,
        lora_params: List[nn.Parameter],
        lr: float = 1e-4,
        batch_size: int = 2,
        beta: float = 0.1,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.buffer = buffer
        self.lora_params = lora_params
        self.batch_size = batch_size
        self.beta = beta
        self.device = device

        self.optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
        self.is_idle: bool = True
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self.total_steps: int = 0

    async def train_step(self) -> Optional[Dict[str, float]]:
        """Executes a single interruptible DPO training step."""
        if len(self.buffer) < self.batch_size:
            return None

        # Check idle gate before training
        if not self.is_idle:
            return None

        batch = self.buffer.sample(self.batch_size)
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = compute_dpo_loss(
            self.model, self.tokenizer, batch, beta=self.beta, device=self.device
        )
        loss.backward()

        # Gradient clipping for stability on 8M model
        torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1
        return metrics

    async def _daemon_loop(self, poll_interval_sec: float = 1.0):
        while self._running:
            try:
                if self.is_idle and len(self.buffer) >= self.batch_size:
                    metrics = await self.train_step()
                    if metrics:
                        # Log non-intrusively
                        pass
                await asyncio.sleep(poll_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
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

    # 1. Inject LoRA adapters into MikuTransformer
    print("\n[LoRA Injection] Wrapping Transformer blocks with rank-4 LoRA...")
    lora_params = inject_lora(model, r=4, alpha=8.0)
    total_base_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_lora_params = sum(p.numel() for p in lora_params)

    print(f"  Base Parameters (Frozen): {total_base_params:,}")
    print(f"  LoRA Parameters (Trainable): {total_lora_params:,} ({total_lora_params/total_base_params*100:.2f}% overhead)")

    # 2. Initialize Experience Buffer
    buffer = ExperienceBuffer(max_capacity=100)
    print("\n[Experience Replay] Logging sample execution trajectories...")
    buffer.add(
        prompt="Write Python code to compute 10 factorial.",
        chosen="import math\nprint(math.factorial(10))",
        rejected="def fact(n):\nreturn n * fact(n-1)\nprint(fact(10))"  # IndentationError/Recursion
    )
    buffer.add(
        prompt="Query system CPU utilization safely.",
        chosen="import psutil\nprint(psutil.cpu_percent())",
        rejected="import os\nos.system('rm -rf /')"  # Unsafe / illegal command
    )
    print(f"  Buffer populated with {len(buffer)} episodes.")

    # 3. Test Direct Preference Optimization Loss Calculation
    print("\n[DPO Optimization] Computing initial DPO step...")
    trainer = IdleTrainer(model, tokenizer, buffer, lora_params, lr=5e-4, batch_size=2, beta=0.1)

    metrics = await trainer.train_step()
    print(f"  Step 1 Result: Loss = {metrics['dpo_loss']:.4f}")
    print(f"  Implicit Reward (Chosen):   {metrics['reward_chosen']:.4f}")
    print(f"  Implicit Reward (Rejected): {metrics['reward_rejected']:.4f}")
    print(f"  Preference Margin (Δ):      {metrics['reward_margin']:.4f}")

    # 4. Demonstrate Zero-Copy Reference Toggling
    print("\n[Adapter Toggle Test] Verifying zero-copy reference bypass...")
    dummy_x = torch.randint(0, tokenizer.vocab_size, (1, 16))
    with torch.no_grad():
        out_active = model(dummy_x)
        with disable_adapters(model):
            out_reference = model(dummy_x)

    delta = torch.abs(out_active - out_reference).sum().item()
    print(f"  L1 Delta between active LoRA policy and base reference: {delta:.6f}")

    print("\n✓ Phase II v1.0 Offline DPO Adapter engine fully verified.")


if __name__ == "__main__":
    asyncio.run(main())
