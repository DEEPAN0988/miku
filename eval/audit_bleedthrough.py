"""
eval/audit_bleedthrough.py — Systematic Bleed-Through Audit Across Broad Held-Out Prompts
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.stdout.reconfigure(encoding="utf-8")
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint


def run_audit(checkpoint_path: str):
    device = torch.device("cpu")
    cfg = ModelConfig.from_yaml("configs/stage_a.yaml")
    model = MikuLM(cfg).to(device)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    prompts = [
        # Narrative / Prose
        ("narrative", "Once upon a time, in a small village"),
        ("narrative", "The detective entered the quiet library and"),
        ("narrative", "Write a short story about an old sailor:"),
        ("narrative", "Write a short paragraph about autumn leaves."),
        # Encyclopedic / Factual
        ("encyclopedic", "Describe the water cycle in simple terms."),
        ("encyclopedic", "The capital city of France is"),
        ("encyclopedic", "The solar system consists of the Sun and"),
        ("encyclopedic", "William Shakespeare was an English playwright who"),
        # Logic
        ("logic", "All birds have feathers. A penguin is a bird. Therefore,"),
        ("logic", "If it rains, the grass gets wet. The grass is wet. Therefore,"),
        ("logic", "What is the logical conclusion if all A are B and all B are C?"),
        # General dialogue
        ("dialogue", "Good morning! How are you doing today?"),
        # Reasoning / Math controls
        ("reasoning_math", "A baker makes 12 cakes per day. In 5 days, he makes"),
        ("reasoning_math", "If a train travels at 60 mph for 3 hours, how far does it go?"),
        ("reasoning_math", "What is 17 multiplied by 13?"),
        ("reasoning_math", "What is 25 multiplied by 4?"),
    ]

    # Patterns indicating GSM8K template contamination
    # "Step-by-step Solution", "Therefore, the final answer is", arithmetic equations, dollar amounts
    gsm_regex = re.compile(
        r"(step-by-step solution|therefore, the final answer|####|\b\d+\s*[\+\*\-\/]\s*\d+\b|\$\d+)",
        re.IGNORECASE,
    )

    print("=" * 75)
    print(f"BROAD HELD-OUT EVALUATION & BLEED-THROUGH AUDIT")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 75)

    non_math_total = 0
    bleed_count = 0
    results = []

    for category, p in prompts:
        ids = tok.encode(p, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=64, temperature=0.8, top_k=50)
        gen_ids = out[0, len(ids):].tolist()
        gen = tok.decode(gen_ids).strip()

        is_math = category == "reasoning_math"
        matches = gsm_regex.findall(gen)

        if not is_math:
            non_math_total += 1
            if matches:
                bleed_count += 1
                classification = f"BLEED-THROUGH -> matched: {matches}"
            else:
                classification = "CLEAN (Topically appropriate, no arithmetic)"
        else:
            classification = "EXPECTED REASONING"

        results.append({
            "category": category,
            "prompt": p,
            "generated": gen,
            "classification": classification,
            "is_bleed": bool(matches and not is_math),
        })

        print(f"\n[{category.upper()}] Prompt: {p!r}")
        print(f"  Generated: {gen!r}")
        print(f"  Verdict:   {classification}")

    print("\n" + "=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)
    print(f"  Total Non-Math Prompts:     {non_math_total}")
    print(f"  Bleed-Through Count:        {bleed_count}")
    print(f"  Clean / Non-Bleed Count:    {non_math_total - bleed_count}")
    print(f"  Bleed-Through Rate:         {bleed_count / non_math_total:.1%}")
    print("=" * 75)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stage_a/step_0001500.pt"
    run_audit(ckpt)
