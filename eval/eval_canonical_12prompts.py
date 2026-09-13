"""
eval/eval_canonical_12prompts.py — Broad Held-Out Evaluation on Canonical Checkpoint
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint


def run_eval(
    checkpoint_path: str = "checkpoints/stage_a/canonical_step_0042000.pt",
    seed: int = 42,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = ModelConfig.from_yaml("configs/stage_a.yaml")
    model = MikuLM(cfg).to(device)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    ckpt = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    prompts = [
        # Non-reasoning: Narrative (4)
        ("narrative", "Once upon a time, in a small village"),
        ("narrative", "The detective entered the quiet library and"),
        ("narrative", "Write a short story about an old sailor:"),
        ("narrative", "Write a short paragraph about autumn leaves."),
        # Non-reasoning: Encyclopedic (4)
        ("encyclopedic", "Describe the water cycle in simple terms."),
        ("encyclopedic", "The capital city of France is"),
        ("encyclopedic", "The solar system consists of the Sun and"),
        ("encyclopedic", "William Shakespeare was an English playwright who"),
        # Non-reasoning: Logic (3)
        ("logic", "All birds have feathers. A penguin is a bird. Therefore,"),
        ("logic", "If it rains, the grass gets wet. The grass is wet. Therefore,"),
        ("logic", "What is the logical conclusion if all A are B and all B are C?"),
        # Non-reasoning: General dialogue (1)
        ("dialogue", "Good morning! How are you doing today?"),
        # Reasoning / Math controls (4)
        ("reasoning_math", "A baker makes 12 cakes per day. In 5 days, he makes"),
        ("reasoning_math", "If a train travels at 60 mph for 3 hours, how far does it go?"),
        ("reasoning_math", "What is 17 multiplied by 13?"),
        ("reasoning_math", "What is 25 multiplied by 4?"),
    ]

    print("=" * 80)
    print(f"EVALUATION ON CANONICAL CHECKPOINT: {checkpoint_path}")
    print(f"Step: {ckpt.get('step')}, Train Loss: {ckpt.get('train_loss'):.4f}, Val Loss: {ckpt.get('val_loss')}")
    print(f"Device: {device}, Random Seed: {seed}")
    print(f"Temperature: {temperature}, Top-K: {top_k}, Top-P: {top_p}")
    print(f"Repetition Penalty: {repetition_penalty}, No-Repeat N-gram: {no_repeat_ngram_size}")
    print("=" * 80)

    for i, (category, p) in enumerate(prompts, 1):
        ids = tok.encode(p, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=90,
                temperature=temperature,
                top_k=top_k if top_p is None else None,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        gen_ids = out[0, len(ids):].tolist()
        gen = tok.decode(gen_ids).strip()

        print(f"\n[{i:02d}] Category: {category.upper()}")
        print(f"Prompt:    {p!r}")
        print(f"Generated: {gen!r}")
        print("-" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 12-prompt evaluation on canonical checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/stage_a/canonical_step_0042000.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    args = parser.parse_args()

    run_eval(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
