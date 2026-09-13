"""
model/architecture.py — MIKU Transformer Decoder
STATUS: IMPLEMENTED — from scratch, no pretrained weights

Architecture: GPT-style decoder-only transformer
  - Rotary positional encoding (RoPE)
  - Pre-norm (LayerNorm before attention and FFN)
  - GELU activation in FFN
  - No bias in Q/K/V/O projections
  - Weight-tied token embedding ↔ LM head
  - PyTorch SDPA (flash-attention on PyTorch 2.x) when available

This file implements the algorithm. Every operation is explicit.
There are no pretrained weights, no external model calls.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (RoPE)
# ---------------------------------------------------------------------------
# Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
# (Su et al., 2021). Implementation via the rotate_half trick (no complex
# numbers) for clarity and broad hardware compatibility.


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of x by half its size.

    For a vector [x1, x2, x3, x4] → [-x3, -x4, x1, x2].
    This implements the 90-degree rotation in the complex plane
    without using torch.view_as_complex.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q:   [B, n_heads, T, head_dim]
        k:   [B, n_heads, T, head_dim]
        cos: [T, head_dim]  — broadcast over B and n_heads
        sin: [T, head_dim]

    Returns:
        q_rot, k_rot with same shape as inputs.
    """
    # Expand cos/sin for broadcasting: [1, 1, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RoPECache(nn.Module):
    """
    Precomputed sin/cos tables for RoPE.

    Stored as a non-parameter buffer so they move to the correct device
    with model.to(device) and are excluded from optimizer updates.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()

        # Frequency bands: one per pair of dimensions
        # freqs[i] = 1 / (theta ^ (2i / head_dim))
        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

        # Position indices
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Outer product: [max_seq_len, head_dim // 2]
        angles = torch.outer(positions, freqs)

        # Duplicate so rotate_half works: [max_seq_len, head_dim]
        angles = torch.cat([angles, angles], dim=-1)

        # Register as buffers (not parameters)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def get(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables sliced to the current sequence length."""
        return (
            self.cos_cached[:seq_len],  # [seq_len, head_dim]
            self.sin_cached[:seq_len],
        )


# ---------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    - No bias in Q/K/V/O projections (modern practice).
    - Causal mask ensures token i cannot attend to token j > i.
    - Uses torch.nn.functional.scaled_dot_product_attention (SDPA)
      which dispatches to FlashAttention on PyTorch 2.x when available.
    - Falls back to manual attention on older PyTorch.
    """

    _SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout_p = config.dropout

        # Q, K, V projections — no bias
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Output projection — no bias
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout on attention weights (only when not using SDPA, to avoid
        # double-applying dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

        # Causal mask (upper-triangular = -inf)
        # Registered as a buffer so it moves with .to(device)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
            ).view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   [B, T, d_model]
            cos: [T, head_dim]  — from RoPECache
            sin: [T, head_dim]

        Returns:
            out: [B, T, d_model]
        """
        B, T, C = x.shape

        # Project to Q, K, V and reshape to [B, n_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        q, k = apply_rope(q, k, cos, sin)

        if self._SDPA_AVAILABLE:
            # PyTorch 2.x path: may use FlashAttention internally
            # is_causal=True applies the causal mask efficiently
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention fallback (PyTorch < 2.0)
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale          # [B, H, T, T]
            att = att.masked_fill(
                ~self.causal_mask[:, :, :T, :T], float("-inf")
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v                                       # [B, H, T, head_dim]

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Layout: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout

    Note: biases are included here (unlike attention projections).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block (Pre-Norm)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Pre-norm layout (LayerNorm before the sublayer):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-norm is more stable than post-norm for deep networks.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Attention sublayer with residual
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # FFN sublayer with residual
        x = x + self.ff(self.ff_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------


class MikuLM(nn.Module):
    """
    MIKU decoder-only language model.
    STATUS: IMPLEMENTED — from scratch

    Forward pass:
      tokens → token embeddings
             → transformer blocks (with RoPE in each attention layer)
             → final LayerNorm
             → LM head (tied with token embedding if weight_tie_embeddings)
             → logits [B, T, vocab_size]

    No pretrained weights. No external model calls.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # RoPE cache (buffers, not parameters)
        self.rope = RoPECache(config.head_dim, config.max_seq_len, config.rope_theta)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final LayerNorm
        self.norm = nn.LayerNorm(config.d_model)

        # LM head (linear projection to vocab)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding and LM head share the same weight matrix
        if config.weight_tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Parameter initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights following GPT-2 conventions:
        - Embeddings: N(0, 0.02)
        - Linear layers: N(0, 0.02)
        - Output projections scaled by 1/sqrt(n_layers) for residual stability
        - LayerNorm: weight=1, bias=0
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual output projections (attention and FFN out) by
        # 1/sqrt(2 * n_layers) so the residual stream variance stays ~1
        # at initialization. The factor of 2 is because each block has
        # two residual additions.
        scale = (2 * self.config.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith(".net.2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 * scale)

    def count_parameters(self) -> int:
        """Count trainable parameters (excludes tied duplicates)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens:  [B, T] — integer token ids
            targets: [B, T] — integer token ids (next-token labels).
                     If None, loss is not computed.

        Returns:
            logits: [B, T, vocab_size]
            loss:   scalar cross-entropy loss, or None if targets not given
        """
        B, T = tokens.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len "
                f"{self.config.max_seq_len}. Truncate before calling forward()."
            )

        # Token embeddings
        x = self.token_embedding(tokens)        # [B, T, d_model]
        x = self.embedding_dropout(x)

        # Fetch RoPE sin/cos for current sequence length
        cos, sin = self.rope.get(T)

        # Run through all transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        # Final norm
        x = self.norm(x)                        # [B, T, d_model]

        # Project to vocabulary
        logits = self.lm_head(x)               # [B, T, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten to [B*T, vocab_size] and [B*T] for cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,    # -1 tokens are masked (padding)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation from a prompt.
        STATUS: IMPLEMENTED — greedy / top-k / top-p / temperature sampling
        with optional repetition penalty (Keskar et al. 2019) and no-repeat n-gram blocking.

        Args:
            prompt_tokens: [B, T_prompt] — integer token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = unmodified logits)
            top_k: if set, only sample from top-k logits
            top_p: if set, nucleus sampling (overrides top_k if both given)
            eos_token_id: stop generation when this token is produced
            repetition_penalty: penalty factor for repeated tokens (1.0 = none, >1.0 discourages repetition)
            no_repeat_ngram_size: if > 0, block repeating any n-gram of this size

        Returns:
            [B, T_prompt + max_new_tokens] token ids
        """
        self.eval()
        tokens = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed (sliding window)
            ctx = tokens if tokens.size(1) <= self.config.max_seq_len else \
                tokens[:, -self.config.max_seq_len:]

            logits, _ = self.forward(ctx)           # [B, T, vocab_size]
            logits = logits[:, -1, :]               # Take last position: [B, vocab_size]

            # Repetition penalty (Keskar et al. 2019 / CTRL)
            if repetition_penalty != 1.0:
                for b in range(tokens.size(0)):
                    prev_tokens = tokens[b].unique()
                    score = logits[b, prev_tokens]
                    score = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty,
                    )
                    logits[b, prev_tokens] = score

            # No-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and tokens.size(1) >= no_repeat_ngram_size - 1:
                prefix_len = no_repeat_ngram_size - 1
                for b in range(tokens.size(0)):
                    seq = tokens[b].tolist()
                    cur_prefix = tuple(seq[-prefix_len:])
                    banned_tokens = set()
                    for j in range(len(seq) - prefix_len):
                        if tuple(seq[j : j + prefix_len]) == cur_prefix:
                            banned_tokens.add(seq[j + prefix_len])
                    if banned_tokens:
                        banned = list(banned_tokens)
                        logits[b, banned] = float("-inf")

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            if top_p is not None:
                # Nucleus (top-p) sampling
                next_token = _sample_top_p(logits, top_p)
            elif top_k is not None and top_k > 0:
                # Top-k sampling
                next_token = _sample_top_k(logits, top_k)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=1)

            # Early stop on EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return tokens


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k sampling: zero out all but the k largest logits, then sample.

    Args:
        logits: [B, vocab_size]
        k: number of top candidates to keep

    Returns:
        [B, 1] sampled token ids
    """
    k = min(k, logits.size(-1))
    # Find the k-th largest logit value per batch element
    values, _ = torch.topk(logits, k, dim=-1)
    threshold = values[:, -1].unsqueeze(-1)  # [B, 1]
    # Mask everything below the threshold
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens whose
    cumulative probability ≥ p, then sample from that set.

    Args:
        logits: [B, vocab_size]
        p: nucleus probability threshold (e.g. 0.95)

    Returns:
        [B, 1] sampled token ids
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above p
    # Shift right by 1 so we always include the token that pushes past p
    remove = (cumulative - sorted_probs) > p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # Sample from filtered distribution
    sampled = torch.multinomial(sorted_probs, num_samples=1)

    # Map back to original vocabulary indices
    return sorted_idx.gather(dim=-1, index=sampled)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from model.config import ModelConfig

    cfg = ModelConfig()
    model = MikuLM(cfg)

    n_params = model.count_parameters()
    print(f"MikuLM initialized: {n_params:,} parameters ({n_params / 1e6:.1f}M)")
    print(cfg.summary())

    # Verify forward pass
    batch_size, seq_len = 2, 64
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits, loss = model(tokens, targets)
    print(f"Forward pass OK: logits={logits.shape}, loss={loss.item():.4f}")

    # Verify generation
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.8, top_k=50)
    print(f"Generate OK: {prompt.shape} → {generated.shape}")
