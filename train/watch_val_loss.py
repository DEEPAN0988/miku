"""
train/watch_val_loss.py — Monitor val loss in logs/stage_a/loss.csv
and report checkpoints, global minimum, and consecutive rises.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/stage_a_small/loss.csv")


def read_losses():
    if not CSV_PATH.exists():
        return []
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "step": int(r["step"]),
                    "train_loss": float(r["train_loss"]),
                    "val_loss": float(r["val_loss"]),
                    "lr": float(r["lr"]),
                    "tok_s": float(r["tokens_per_sec"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def summarize():
    rows = read_losses()
    if not rows:
        print("No loss records found yet.")
        return

    print(f"\n--- LOSS MONITOR (total checkpoints logged: {len(rows)}) ---")
    min_row = min(rows, key=lambda x: x["val_loss"])
    print(f"Global min val loss: {min_row['val_loss']:.4f} at step {min_row['step']}")
    print("\nRecent checkpoints:")
    for r in rows[-5:]:
        diff_str = ""
        print(f"  Step {r['step']:>6,}: train={r['train_loss']:.4f} | val={r['val_loss']:.4f} | lr={r['lr']:.2e} | {r['tok_s']:>6.0f} tok/s")

    # Check for consecutive rises past min_row
    min_idx = rows.index(min_row)
    post_min = rows[min_idx:]
    if len(post_min) >= 3:
        # Check if the last 2 steps rose consecutively
        v0 = post_min[-3]["val_loss"]
        v1 = post_min[-2]["val_loss"]
        v2 = post_min[-1]["val_loss"]
        if v2 > v1 > v0:
            print(f"\n[ALERT] 2 consecutive val loss rises detected past minimum! ({v0:.4f} -> {v1:.4f} -> {v2:.4f})")
            print(f"Candidate canonical checkpoint: step_{min_row['step']:07d}.pt")


if __name__ == "__main__":
    summarize()
