"""
data/prepare_sft_dolly.py — Fetch and Prepare Databricks Dolly-15k for MIKU SFT

Downloads databricks/databricks-dolly-15k, formats with natural prompt/response tags,
splits 95/5 into train/val, computes token counts, and runs contamination verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer

DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def download_dolly(dest_path: Path) -> Path:
    """Download Dolly-15k raw jsonl if not already present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        print(f"  Dolly-15k already present at {dest_path} ({dest_path.stat().st_size / 1e6:.2f} MB)")
        return dest_path

    print(f"  Downloading databricks/databricks-dolly-15k from Hugging Face ...")
    req = urllib.request.Request(DOLLY_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)

    print(f"  Downloaded {dest_path.stat().st_size / 1e6:.2f} MB -> {dest_path}")
    return dest_path


def format_example(instruction: str, context: str, response: str) -> Tuple[str, str]:
    """
    Format a Dolly example into prompt and response parts.
    Uses natural keywords (Instruction:/Response:/Context:) to avoid tokenizer subword fragmentation.
    """
    ctx = context.strip()
    inst = instruction.strip()
    resp = response.strip()

    if ctx:
        prompt = f"Context:\n{ctx}\n\nInstruction:\n{inst}\n\nResponse:\n"
    else:
        prompt = f"Instruction:\n{inst}\n\nResponse:\n"

    return prompt, resp


def prepare_sft(
    raw_dir: str = "data/raw/sft",
    out_dir: str = "data/processed/sft",
    val_fraction: float = 0.05,
    seed: int = 42,
    max_seq_len: int = 256,
):
    print("=" * 70)
    print("MIKU PHASE 4: DOLLY-15K SFT DATASET PREPARATION")
    print("=" * 70)

    raw_path = Path(raw_dir) / "databricks-dolly-15k.jsonl"
    download_dolly(raw_path)

    print("\n[STAGE 1] Loading and validating raw dataset ...")
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_records = len(records)
    print(f"  Total raw records loaded: {total_records:,}")
    print(f"  License: CC-BY-SA 3.0 (Commercial & open use)")

    # Category counts
    category_counts: Dict[str, int] = {}
    for r in records:
        cat = r.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("  Category breakdown:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<25s}: {cnt:>5,} ({cnt/total_records:.1%})")

    # Deduplicate examples by (instruction, context) to prevent leakage
    print("\n[STAGE 2] Deduplicating and formatting examples into Prompt/Response templates ...")
    seen_prompts = set()
    formatted_data = []
    n_duplicates = 0

    for r in records:
        prompt_key = (r["instruction"].strip().lower(), r.get("context", "").strip().lower())
        if prompt_key in seen_prompts:
            n_duplicates += 1
            continue
        seen_prompts.add(prompt_key)

        prompt, resp = format_example(r["instruction"], r.get("context", ""), r["response"])
        formatted_data.append({
            "instruction": r["instruction"],
            "context": r.get("context", ""),
            "prompt": prompt,
            "response": resp,
            "category": r.get("category", ""),
        })

    print(f"  Removed {n_duplicates} duplicate prompt entries ({n_duplicates/total_records:.1%})")
    print(f"  Unique records remaining: {len(formatted_data):,}")

    # Shuffle and split
    print(f"\n[STAGE 3] Train/Val split (val_fraction={val_fraction:.0%}, seed={seed}) ...")
    rng = random.Random(seed)
    rng.shuffle(formatted_data)

    n_val = max(1, int(len(formatted_data) * val_fraction))
    val_data = formatted_data[:n_val]
    train_data = formatted_data[n_val:]

    print(f"  Train records: {len(train_data):,}")
    print(f"  Val records:   {len(val_data):,}")

    # Tokenizer stats
    print("\n[STAGE 4] Measuring token counts with MikuTokenizer ...")
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    train_tokens = 0
    val_tokens = 0
    train_truncated = 0
    val_truncated = 0

    for item in train_data:
        full_text = item["prompt"] + item["response"]
        tokens = tok.encode(full_text, add_bos=True, add_eos=True)
        train_tokens += len(tokens)
        if len(tokens) > max_seq_len:
            train_truncated += 1

    for item in val_data:
        full_text = item["prompt"] + item["response"]
        tokens = tok.encode(full_text, add_bos=True, add_eos=True)
        val_tokens += len(tokens)
        if len(tokens) > max_seq_len:
            val_truncated += 1

    print(f"  Train: {train_tokens:,} tokens (mean {train_tokens/len(train_data):.1f} tok/ex, "
          f"{train_truncated/len(train_data):.1%} > {max_seq_len} tok)")
    print(f"  Val:   {val_tokens:,} tokens (mean {val_tokens/len(val_data):.1f} tok/ex, "
          f"{val_truncated/len(val_data):.1%} > {max_seq_len} tok)")
    print(f"  Total SFT tokens: {train_tokens + val_tokens:,}")

    # Save processed jsonl files
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_file = out_path / "train_sft.jsonl"
    val_file = out_path / "val_sft.jsonl"
    meta_file = out_path / "sft_meta.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "databricks/databricks-dolly-15k",
        "license": "CC-BY-SA 3.0",
        "total_records": total_records,
        "train_records": len(train_data),
        "val_records": len(val_data),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "max_seq_len": max_seq_len,
        "seed": seed,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved train split: {train_file}")
    print(f"  Saved val split:   {val_file}")
    print(f"  Saved metadata:    {meta_file}")

    # Contamination check (exact match on instruction + context)
    print("\n[STAGE 5] Train/Val contamination check ...")
    train_hashes = {
        hashlib.sha256(f"{x['instruction'].strip()} ||| {x['context'].strip()}".encode("utf-8")).hexdigest()
        for x in train_data
    }
    leaks = sum(
        1 for x in val_data
        if hashlib.sha256(f"{x['instruction'].strip()} ||| {x['context'].strip()}".encode("utf-8")).hexdigest() in train_hashes
    )
    print(f"  Exact-match leaking examples: {leaks} ({leaks / len(val_data):.2%})")
    if leaks == 0:
        print("  [OK] ZERO LEAKAGE between SFT train and val sets.")
    else:
        print("  [WARN] Leakage detected in SFT split!")

    print("\n[OK] SFT dataset preparation complete.\n")


if __name__ == "__main__":
    prepare_sft()
