"""
eval/sample_generate.py — Verbatim Generation Comparison Across Checkpoints
STATUS: IMPLEMENTED

Loads one or more checkpoints and generates verbatim text samples from
the same set of prompts, printing side-by-side for qualitative comparison.

This is the primary tool for diagnosing quality/loss divergence:
if loss continues to decrease but generated text degrades, you'll see
it here by comparing samples from step N vs step N+500.

Per constitution Rule 2: outputs raw generated text only.
Does not assert quality claims without evidence.

Usage:
    # Compare specific steps:
    python eval/sample_generate.py \
        --checkpoints checkpoints/stage_a/ \
        --steps 1000 2000 3000 \
        --config configs/stage_a.yaml

    # Compare all checkpoints:
    python eval/sample_generate.py \
        --checkpoints checkpoints/stage_a/ \
        --config configs/stage_a.yaml

    # Single checkpoint:
    python eval/sample_generate.py \
        --checkpoints checkpoints/stage_a/step_0001000.pt \
        --config configs/stage_a.yaml

    # Custom prompts:
    python eval/sample_generate.py \
        --checkpoints checkpoints/stage_a/ \
        --steps 500 1000 \
        --config configs/stage_a.yaml \
        --prompts "What is 2+2?" "Explain why the sky is blue."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import list_checkpoints, load_checkpoint
from train.train import detect_device, select_dtype


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_from_checkpoint(
    checkpoint_path: str,
    prompts: List[str],
    config_path: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
) -> Dict:
    """
    Load a checkpoint and generate text for each prompt.

    Returns dict:
      {
        "step": int,
        "train_loss": float | None,
        "val_loss": float | None,
        "samples": [{"prompt": str, "generated": str}, ...]
      }
    """
    device = detect_device()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig.from_yaml(config_path)
    dtype = select_dtype(device, cfg.get("training", {}).get("dtype", "auto"))

    model = MikuLM(model_cfg).to(device)
    ckpt = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    step = ckpt.get("step", 0)
    train_loss = ckpt.get("train_loss", None)
    val_loss = ckpt.get("val_loss", None)

    tok_prefix = cfg["tokenizer"]["model_prefix"]
    tokenizer = MikuTokenizer.load(tok_prefix)

    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        try:
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_id,
            )
            new_ids = output[0, len(ids):].tolist()
            generated = tokenizer.decode(new_ids)
        except Exception as e:
            generated = f"[ERROR: {e}]"

        samples.append({"prompt": prompt, "generated": generated})

    return {
        "checkpoint_path": checkpoint_path,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(results: List[Dict], prompts: List[str]) -> None:
    """
    Print side-by-side comparison of generated text across checkpoints.
    """
    print("\n" + "=" * 70)
    print("MIKU GENERATION COMPARISON ACROSS CHECKPOINTS")
    print("=" * 70)
    print(f"  Checkpoints compared: {len(results)}")
    print(f"  Prompts:              {len(prompts)}")
    print()

    for prompt_idx, prompt in enumerate(prompts):
        print("─" * 70)
        print(f"PROMPT {prompt_idx + 1}: {prompt!r}")
        print("─" * 70)

        for r in results:
            step = r["step"]
            train_loss = r.get("train_loss")
            val_loss = r.get("val_loss")
            loss_str = ""
            if train_loss is not None:
                loss_str += f" train_loss={train_loss:.4f}"
            if val_loss is not None:
                loss_str += f" val_loss={val_loss:.4f}"

            # Find the sample for this prompt
            sample = next(
                (s for s in r["samples"] if s["prompt"] == prompt), None
            )
            generated = sample["generated"] if sample else "[NOT FOUND]"

            print(f"\n  ── Step {step:,}{loss_str} ──")
            print(f"  {generated!r}")

        print()

    print("=" * 70)
    print()
    print("HOW TO INTERPRET:")
    print("  - Compare text quality across steps, independent of loss.")
    print("  - If loss decreases but text degrades: quality/loss inversion.")
    print("  - Look for: coherence, relevance to prompt, non-repetition.")
    print("  - 'PARTIAL': model is in early training — incoherence expected.")
    print("  - Check checkpoint _samples.txt files for saved snapshots.")
    print()


def save_comparison_txt(results: List[Dict], prompts: List[str], output_path: str) -> None:
    """Save the comparison as a text file."""
    lines = ["MIKU GENERATION COMPARISON", ""]
    for prompt in prompts:
        lines.append(f"PROMPT: {prompt!r}")
        lines.append("-" * 60)
        for r in results:
            step = r["step"]
            sample = next((s for s in r["samples"] if s["prompt"] == prompt), None)
            generated = sample["generated"] if sample else "[NOT FOUND]"
            lines.append(f"  Step {step}: {generated!r}")
        lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Comparison saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare MIKU generation across checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoints", required=True,
        help="Path to checkpoint directory OR a single .pt file"
    )
    p.add_argument(
        "--config", default="configs/stage_a.yaml",
        help="YAML config path"
    )
    p.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Step numbers to compare (default: all checkpoints)"
    )
    p.add_argument(
        "--prompts", nargs="+", default=None,
        help="Prompts to generate from (overrides config held_out_prompts)"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=150
    )
    p.add_argument(
        "--temperature", type=float, default=0.8
    )
    p.add_argument(
        "--top-k", type=int, default=50
    )
    p.add_argument(
        "--output", default=None,
        help="Save comparison to this text file"
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Collect checkpoint paths
    ckpt_path = args.checkpoints
    if os.path.isfile(ckpt_path):
        ckpt_entries = [{"path": ckpt_path, "step": None}]
    elif os.path.isdir(ckpt_path):
        ckpt_entries = list_checkpoints(ckpt_path)
        if not ckpt_entries:
            print(f"No checkpoints found in {ckpt_path}")
            sys.exit(1)
    else:
        print(f"Path not found: {ckpt_path}")
        sys.exit(1)

    # Filter by step if requested
    if args.steps:
        step_set = set(args.steps)
        ckpt_entries = [e for e in ckpt_entries if e.get("step") in step_set]
        if not ckpt_entries:
            print(f"No checkpoints found for steps {args.steps}")
            sys.exit(1)

    print(f"Comparing {len(ckpt_entries)} checkpoint(s)")

    # Determine prompts
    prompts = args.prompts
    if prompts is None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        prompts = cfg.get("eval", {}).get("held_out_prompts", [
            "What is 2 + 2?",
            "Explain why the sky is blue.",
            "Once upon a time,",
        ])

    # Generate for each checkpoint
    results = []
    for entry in ckpt_entries:
        path = entry["path"]
        step = entry.get("step", "?")
        print(f"\nGenerating from step {step} ({path}) ...")
        result = generate_from_checkpoint(
            checkpoint_path=path,
            prompts=prompts,
            config_path=args.config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        results.append(result)

    # Print comparison
    print_comparison(results, prompts)

    # Save if requested
    if args.output:
        save_comparison_txt(results, prompts, args.output)


if __name__ == "__main__":
    main()
