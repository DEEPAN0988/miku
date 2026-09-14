"""
train/prepare_unified_v03_data.py — Grand Unified SFT Dataset Compiler for Miku v0.3 (2-Hour Training)

Combines and balances:
  - Real-Time OS Scenarios (Settings, VS Code, Browser, Calc, Paint, Explorer)
  - Tool Calling & Media Tool Dispatch (Music, Camera, OCR, Vision Grounder, Memory, Web)
  - Hierarchical Multi-Intent Execution
  - Multi-Turn Dialogue & Context Continuation
  - Diversified Conversational QA
  - Dolly-15k General Reasoning & Knowledge Replay

Outputs to:
  data/processed/sft_v03_unified/train_unified.jsonl
  data/processed/sft_v03_unified/val_unified.jsonl
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer


DATASET_SOURCES = [
    ("scenario", "data/processed/sft_scenarios/train_scenarios.jsonl", "data/processed/sft_scenarios/val_scenarios.jsonl"),
    ("tool", "data/processed/sft_tool/train_tool.jsonl", "data/processed/sft_tool/val_tool.jsonl"),
    ("hierarchical", "data/processed/sft_hierarchical/train.jsonl", "data/processed/sft_hierarchical/val.jsonl"),
    ("multiturn", "data/processed/sft_multiturn/train_multiturn.jsonl", "data/processed/sft_multiturn/val_multiturn.jsonl"),
    ("diversified", "data/processed/sft_diversified/train_diversified.jsonl", "data/processed/sft_diversified/val_diversified.jsonl"),
    ("dolly", "data/processed/sft/train_sft.jsonl", "data/processed/sft/val_sft.jsonl"),
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    p = Path(path)
    if not p.exists():
        print(f"[!] Warning: {path} not found; skipping.")
        return []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    return samples


def main():
    random.seed(42)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    out_dir = Path("data/processed/sft_v03_unified")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPILING GRAND UNIFIED MIKU v0.3 SFT DATASET FOR 2-HOUR TRAINING")
    print("=" * 80)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    domain_stats: Dict[str, Dict[str, int]] = {}

    for domain, train_path, val_path in DATASET_SOURCES:
        t_data = load_jsonl(train_path)
        v_data = load_jsonl(val_path)

        # Tag domain if not present
        for s in t_data:
            s["domain"] = domain
        for s in v_data:
            s["domain"] = domain

        train_samples.extend(t_data)
        val_samples.extend(v_data)
        domain_stats[domain] = {"train": len(t_data), "val": len(v_data)}
        print(f" [*] Loaded domain '{domain:<12}': {len(t_data):>6,} train | {len(v_data):>5,} val")

    print("-" * 80)
    print(f"Total Raw Samples: {len(train_samples):,} train | {len(val_samples):,} val")

    # Filter to ensure max_seq_len <= 256
    valid_train = []
    for s in train_samples:
        prompt = s.get("prompt", "")
        resp = s.get("response", "")
        tok_len = len(tok.encode(prompt + resp, add_bos=True, add_eos=True))
        if tok_len <= 256:
            valid_train.append(s)

    valid_val = []
    for s in val_samples:
        prompt = s.get("prompt", "")
        resp = s.get("response", "")
        tok_len = len(tok.encode(prompt + resp, add_bos=True, add_eos=True))
        if tok_len <= 256:
            valid_val.append(s)

    random.shuffle(valid_train)
    random.shuffle(valid_val)

    train_file = out_dir / "train_unified.jsonl"
    val_file = out_dir / "val_unified.jsonl"
    stats_file = out_dir / "stats.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in valid_train:
            f.write(json.dumps(s) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in valid_val:
            f.write(json.dumps(s) + "\n")

    stats = {
        "total_train": len(valid_train),
        "total_val": len(valid_val),
        "domains": domain_stats,
    }
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[+] Successfully saved Grand Unified v0.3 Dataset:")
    print(f"    Train: {len(valid_train):,} samples -> {train_file}")
    print(f"    Val  : {len(valid_val):,} samples -> {val_file}")
    print(f"    Stats: {stats_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
