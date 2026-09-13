"""
train/checkpoint.py — MIKU Checkpoint Management
STATUS: IMPLEMENTED

Saves and loads model + optimizer state, training metadata, and
verbatim generation samples alongside every checkpoint.

Checkpoint format (single .pt file):
  {
    "step":        int,
    "model_state": OrderedDict,
    "optimizer_state": dict,
    "config":      dict (ModelConfig.to_dict()),
    "train_loss":  float,
    "val_loss":    float | None,
    "timestamp":   str (ISO 8601),
    "samples":     List[{"prompt": str, "generated": str}],
    "n_params":    int,
    "torch_version": str,
  }
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
    config_dict: Dict[str, Any],
    checkpoint_dir: str,
    val_loss: Optional[float] = None,
    samples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Save a checkpoint to disk.

    Args:
        model:           the MikuLM model
        optimizer:       the optimizer
        step:            current training step
        train_loss:      training loss at this step
        config_dict:     ModelConfig.to_dict() — saved for reproducibility
        checkpoint_dir:  directory to save checkpoints
        val_loss:        validation loss (None if not computed)
        samples:         list of {"prompt": str, "generated": str} dicts
                         saved verbatim alongside the loss

    Returns:
        path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")

    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config_dict,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": samples or [],
        "n_params": sum(p.numel() for p in set(model.parameters())),
        "torch_version": torch.__version__,
    }

    # Atomic save: write to .tmp then rename (avoids corrupt files on crash)
    tmp_path = ckpt_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    # Also save a human-readable samples file for quick inspection
    if samples:
        sample_path = ckpt_path.replace(".pt", "_samples.txt")
        _write_samples_txt(sample_path, step, train_loss, val_loss, samples)

    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optionally optimizer).

    Args:
        checkpoint_path: path to .pt file
        model:           model to load weights into (modified in-place)
        optimizer:       if given, optimizer state is also restored
        device:          device to map tensors to (default: model's current device)

    Returns:
        Full checkpoint dict (includes step, loss, samples, config, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device or next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    List all checkpoints in a directory, sorted by step.

    Returns:
        List of dicts with keys: path, step, train_loss, val_loss, timestamp
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    entries = []
    for pt_file in sorted(ckpt_dir.glob("step_*.pt")):
        try:
            # Load only metadata (not full state dicts) to keep this fast
            ckpt = torch.load(
                str(pt_file),
                map_location="cpu",
                weights_only=False,
            )
            entries.append({
                "path": str(pt_file),
                "step": ckpt.get("step"),
                "train_loss": ckpt.get("train_loss"),
                "val_loss": ckpt.get("val_loss"),
                "timestamp": ckpt.get("timestamp"),
                "n_samples": len(ckpt.get("samples", [])),
            })
        except Exception as e:
            print(f"[WARNING] Could not read {pt_file}: {e}")

    entries.sort(key=lambda e: e["step"] or 0)
    return entries


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Return the path to the most recent checkpoint (by step number),
    or None if no checkpoints exist.
    """
    entries = list_checkpoints(checkpoint_dir)
    if not entries:
        return None
    return entries[-1]["path"]


def print_checkpoint_table(checkpoint_dir: str) -> None:
    """Print a formatted table of all checkpoints to stdout."""
    entries = list_checkpoints(checkpoint_dir)
    if not entries:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"\n{'Step':>10} {'Train Loss':>12} {'Val Loss':>10} {'Samples':>8}  Timestamp")
    print("-" * 70)
    for e in entries:
        val = f"{e['val_loss']:.4f}" if e['val_loss'] is not None else "   N/A"
        print(
            f"{e['step']:>10,} "
            f"{e['train_loss']:>12.4f} "
            f"{val:>10} "
            f"{e['n_samples']:>8}  "
            f"{e['timestamp'] or '?'}"
        )
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_samples_txt(
    path: str,
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    samples: List[Dict[str, str]],
) -> None:
    """Write verbatim generation samples to a human-readable text file."""
    val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
    lines = [
        f"MIKU Generation Samples — Step {step}",
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_str}",
        f"Saved: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
        "",
    ]
    for i, sample in enumerate(samples, 1):
        lines.append(f"[Sample {i}]")
        lines.append(f"PROMPT:    {sample['prompt']!r}")
        lines.append(f"GENERATED: {sample['generated']!r}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stage_a"
    print_checkpoint_table(ckpt_dir)
