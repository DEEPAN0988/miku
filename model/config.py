"""
model/config.py — MIKU Model Configuration
STATUS: IMPLEMENTED

ModelConfig is the single source of truth for all architectural
hyperparameters. Loaded from YAML and passed to MikuLM and all
training/eval scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "tiny": {
        # ~12.5M params - optimized for CPU-only training on smaller corpora
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 256,
        "d_ff": 1024,
    },
    "nano": {
        # ~34M params - fastest training, minimum VRAM
        "n_layers": 8,
        "n_heads": 8,
        "d_model": 512,
        "d_ff": 2048,
    },
    "small": {
        # ~54M params — default Stage A target
        "n_layers": 12,
        "n_heads": 8,
        "d_model": 512,
        "d_ff": 2048,
    },
    "medium": {
        # ~85M params — for 8GB+ VRAM
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ff": 2048,
    },
}


@dataclass
class ModelConfig:
    """
    All architectural hyperparameters for MikuLM.

    Values here are the defaults for the "small" preset (~54M params).
    Override via stage_a.yaml or by passing kwargs directly.
    """

    # Vocabulary
    vocab_size: int = 32_000

    # Depth and width
    n_layers: int = 12
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048           # FFN hidden dimension (typically 4 × d_model)

    # Sequence
    max_seq_len: int = 512

    # Regularization
    dropout: float = 0.1

    # Architecture choices
    weight_tie_embeddings: bool = True   # Tie embedding ↔ LM head
    rope_theta: float = 10_000.0         # RoPE base frequency

    # Derived (computed, do not set manually)
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        object.__setattr__(self, "head_dim", self.d_model // self.n_heads)

    # ------------------------------------------------------------------
    # Parameter count estimate (no forward pass required)
    # ------------------------------------------------------------------

    def count_params(self) -> int:
        """
        Estimate total parameter count analytically.
        Excludes tied weights counted twice.
        """
        # Token embedding
        embed = self.vocab_size * self.d_model

        # Per transformer block (no biases in attn projections)
        attn = 4 * self.d_model * self.d_model            # Q K V O
        ffn = 2 * self.d_model * self.d_ff                # up + down
        norms = 2 * (2 * self.d_model)                    # 2 LayerNorms × (weight + bias)
        per_block = attn + ffn + norms
        all_blocks = self.n_layers * per_block

        # Final LayerNorm
        final_norm = 2 * self.d_model

        # LM head: if weight-tied, no extra params; else same as embedding
        lm_head = 0 if self.weight_tie_embeddings else self.vocab_size * self.d_model

        return embed + all_blocks + final_norm + lm_head

    def summary(self) -> str:
        n = self.count_params()
        return (
            f"MikuLM | "
            f"layers={self.n_layers} heads={self.n_heads} "
            f"d_model={self.d_model} d_ff={self.d_ff} "
            f"vocab={self.vocab_size} seq={self.max_seq_len} | "
            f"~{n / 1e6:.1f}M params"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "weight_tie_embeddings": self.weight_tie_embeddings,
            "rope_theta": self.rope_theta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = {k: v for k, v in d.items() if k != "head_dim"}
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """
        Load ModelConfig from a YAML config file (e.g. stage_a.yaml).

        Applies preset first, then overrides with explicit fields.
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg.get("model", {})
        preset_name = model_cfg.get("preset", "small")

        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Valid: {list(PRESETS.keys())}"
            )

        # Start from preset defaults, override with explicit yaml fields
        params = dict(PRESETS[preset_name])
        skip = {"preset"}
        for k, v in model_cfg.items():
            if k not in skip:
                params[k] = v

        return cls(**params)

    def save_yaml(self, path: str) -> None:
        """Serialize this config back to YAML for reproducibility."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"model": self.to_dict()}, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig()
    print(cfg.summary())
    for preset, overrides in PRESETS.items():
        c = ModelConfig(**overrides)
        print(f"  {preset:8s}: ~{c.count_params() / 1e6:.1f}M params")
