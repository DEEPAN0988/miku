"""
eval/held_out_eval.py — Held-Out Evaluation
STATUS: IMPLEMENTED

Evaluates a MIKU checkpoint on the held-out validation set.

Outputs two types of evidence (per constitution Rule 2):
  1. Perplexity — computed on val.bin, reported as a number with evidence
  2. Qualitative generation — verbatim text on held-out prompts that
     were NOT used during training, printed to stdout and saved to disk

The Stage A gate requires:
  - Coherent, on-topic, non-repetitive text on held-out prompts
  - Verified zero train/val leakage (run contamination_check.py first)
  - Reasoning-data-weighted corpus

Usage:
    python eval/held_out_eval.py \
        --checkpoint checkpoints/stage_a/step_0001000.pt \
        --config     configs/stage_a.yaml

    # With custom prompts from a file:
    python eval/held_out_eval.py \
        --checkpoint checkpoints/stage_a/step_0001000.pt \
        --config     configs/stage_a.yaml \
        --prompts-file my_eval_prompts.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml


sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint
from train.dataset import create_dataloader
from train.train import compute_val_loss, detect_device, select_dtype


# ---------------------------------------------------------------------------
# Qualitative analysis helpers
# ---------------------------------------------------------------------------

def measure_repetition(text: str, window: int = 10) -> float:
    """
    Measure repetition rate as the fraction of word n-grams (window=10)
    that have appeared earlier in the text.

    Returns:
        float in [0, 1] — 0 = no repetition, 1 = fully repetitive
    """
    words = text.lower().split()
    if len(words) < window:
        return 0.0

    seen: set = set()
    repeated = 0
    total = 0
    for i in range(len(words) - window + 1):
        ngram = tuple(words[i : i + window])
        if ngram in seen:
            repeated += 1
        seen.add(ngram)
        total += 1

    return repeated / max(total, 1)


def measure_avg_sentence_length(text: str) -> float:
    """Average number of words per sentence (rough punctuation split)."""
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def analyze_generation(prompt: str, generated: str) -> Dict:
    """
    Compute qualitative metrics for a single generated sample.

    Returns a dict of diagnostic signals — not a pass/fail verdict.
    """
    repetition_rate = measure_repetition(generated, window=8)
    avg_sent_len = measure_avg_sentence_length(generated)
    n_words = len(generated.split())
    n_unique_words = len(set(generated.lower().split()))
    lexical_diversity = n_unique_words / max(n_words, 1)

    # Simple coherence signal: does the generated text stay on topic?
    # (Heuristic: does it contain words from the prompt?)
    prompt_words = set(prompt.lower().split())
    gen_words = set(generated.lower().split())
    topic_overlap = len(prompt_words & gen_words) / max(len(prompt_words), 1)

    return {
        "n_words": n_words,
        "n_unique_words": n_unique_words,
        "lexical_diversity": lexical_diversity,
        "repetition_rate_8gram": repetition_rate,
        "avg_sentence_length": avg_sent_len,
        "prompt_topic_overlap": topic_overlap,
        "flags": {
            "high_repetition": repetition_rate > 0.3,
            "very_short_output": n_words < 10,
            "very_low_diversity": lexical_diversity < 0.3,
        },
    }


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    checkpoint_path: str,
    config_path: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    eval_max_batches: int = 50,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run held-out evaluation on a checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        config_path:     path to stage_a.yaml config
        prompts:         held-out prompts (if None, loaded from config)
        max_new_tokens:  tokens to generate per prompt
        temperature:     sampling temperature
        top_k:           top-k sampling
        eval_max_batches: cap on val loss computation batches
        output_dir:      if set, save report here

    Returns:
        dict with perplexity, per-prompt results, and qualitative metrics
    """
    # ── Setup ─────────────────────────────────────────────────────────
    device = detect_device()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig.from_yaml(config_path)
    dtype = select_dtype(device, cfg.get("training", {}).get("dtype", "auto"))

    # ── Load model from checkpoint ─────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    model = MikuLM(model_cfg).to(device)
    ckpt = load_checkpoint(checkpoint_path, model, device=device)
    step = ckpt.get("step", "?")
    ckpt_train_loss = ckpt.get("train_loss", None)
    ckpt_val_loss = ckpt.get("val_loss", None)
    print(f"  Step: {step} | Recorded train_loss={ckpt_train_loss} | val_loss={ckpt_val_loss}")
    print(f"  {model_cfg.summary()}")
    model.eval()

    # ── Load tokenizer ─────────────────────────────────────────────────
    tok_prefix = cfg["tokenizer"]["model_prefix"]
    tokenizer = MikuTokenizer.load(tok_prefix)

    # ── Val perplexity ─────────────────────────────────────────────────
    val_bin = cfg["data"]["val_bin"]
    print(f"\nComputing perplexity on {val_bin} ...")
    val_loader = create_dataloader(
        bin_path=val_bin,
        seq_len=model_cfg.max_seq_len,
        batch_size=8,
        shuffle=False,
    )
    val_loss = compute_val_loss(model, val_loader, device, dtype, max_batches=eval_max_batches)
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    print(f"  Val NLL loss:   {val_loss:.4f}")
    print(f"  Val perplexity: {perplexity:.2f}")
    print(f"  (Batches used: {eval_max_batches} - this is a sample, not the full val set)")

    # ── Held-out generation ────────────────────────────────────────────
    if prompts is None:
        prompts = cfg.get("eval", {}).get("held_out_prompts", [
            "The answer to this problem is",
            "Step by step,",
            "The capital of France is",
        ])

    print(f"\nGenerating on {len(prompts)} held-out prompts ...")
    print(f"  max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")
    print()

    prompt_results = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_id,
        )
        new_ids = output[0, len(ids):].tolist()
        generated = tokenizer.decode(new_ids)

        metrics = analyze_generation(prompt, generated)

        prompt_results.append({
            "prompt": prompt,
            "generated": generated,
            "metrics": metrics,
        })

        # Print to stdout verbatim
        print(f"--- Prompt {i+1} ---------------------------------------------")
        print(f"PROMPT:    {prompt!r}")
        print(f"GENERATED: {generated!r}")
        print()
        print(f"  Metrics:")
        print(f"    Words:              {metrics['n_words']}")
        print(f"    Lexical diversity:  {metrics['lexical_diversity']:.3f}")
        print(f"    Repetition (8g):    {metrics['repetition_rate_8gram']:.3f}")
        print(f"    Avg sentence len:   {metrics['avg_sentence_length']:.1f}")
        flags = [k for k, v in metrics["flags"].items() if v]
        if flags:
            print(f"    [WARN] FLAGS: {', '.join(flags)}")
        else:
            print(f"    [OK] No quality flags triggered")
        print()

    # -- Summary --------------------------------------------------------
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Checkpoint:     {checkpoint_path}")
    print(f"  Step:           {step}")
    print(f"  Val NLL loss:   {val_loss:.4f}")
    print(f"  Val perplexity: {perplexity:.2f}")
    print()
    avg_rep = sum(r["metrics"]["repetition_rate_8gram"] for r in prompt_results) / max(len(prompt_results), 1)
    avg_div = sum(r["metrics"]["lexical_diversity"] for r in prompt_results) / max(len(prompt_results), 1)
    n_flagged = sum(1 for r in prompt_results if any(r["metrics"]["flags"].values()))
    print(f"  Prompts evaluated:       {len(prompt_results)}")
    print(f"  Mean repetition rate:    {avg_rep:.3f}  (lower is better, <0.1 is good)")
    print(f"  Mean lexical diversity:  {avg_div:.3f}  (higher is better, >0.5 is good)")
    print(f"  Prompts with quality flags: {n_flagged}/{len(prompt_results)}")
    print()
    print("  Stage A gate requires:")
    print("    - Coherent, on-topic, non-repetitive text")
    print("    - Verified zero train/val leakage (run contamination_check.py)")
    print("    - Reasoning-weighted corpus (check data/processed/stats.json)")
    print()
    print("  These requirements must be verified with EVIDENCE - not asserted.")
    print("=" * 60)

    result = {
        "checkpoint_path": checkpoint_path,
        "step": step,
        "val_nll_loss": val_loss,
        "val_perplexity": perplexity,
        "eval_batches_used": eval_max_batches,
        "prompt_results": prompt_results,
        "summary": {
            "n_prompts": len(prompt_results),
            "mean_repetition_rate": avg_rep,
            "mean_lexical_diversity": avg_div,
            "n_prompts_with_flags": n_flagged,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import json
        step_str = str(step).zfill(7)
        report_path = os.path.join(output_dir, f"eval_step_{step_str}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport saved to {report_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MIKU held-out evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", default="configs/stage_a.yaml", help="Path to YAML config")
    p.add_argument("--prompts-file", default=None,
                   help="Text file with one prompt per line (overrides config)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=50,
                   help="Max val batches for perplexity (more = more accurate but slower)")
    p.add_argument("--output-dir", default="logs/stage_a",
                   help="Directory to save eval report JSON")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    prompts = None
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().strip().split("\n")
        prompts = [p.strip() for p in prompts if p.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    run_eval(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eval_max_batches=args.eval_batches,
        output_dir=args.output_dir,
    )
