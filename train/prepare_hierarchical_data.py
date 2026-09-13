"""
train/prepare_hierarchical_data.py — SFT Dataset Generator for Hierarchical Tool-Calling.
Outputs:
  Category: <category>
  Action: <action>
  Argument: <argument>
Preserves exact 2:1 conversational replay ratio.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer
import train.prepare_tool_data as ptd

TOOL_TO_CATEGORY = {
    "Get Time": "System Info",
    "Get Date": "System Info",
    "Get Battery": "System Info",
    "Get Volume": "System Info",
    "List Running Apps": "System Info",
    "Play Media": "Media Control",
    "Pause Media": "Media Control",
    "Next Track": "Media Control",
    "Previous Track": "Media Control",
    "Play Query": "Media Control",
    "Open App": "App Management",
    "Close App": "App Management",
    "Set Volume": "Device Settings",
    "Search Web": "Web Search"
}


def generate_hierarchical_samples(count_per_tool: int = 180) -> List[Dict[str, str]]:
    samples = []
    for tool_name, tmpls in ptd.TEMPLATES.items():
        cat = TOOL_TO_CATEGORY[tool_name]
        for _ in range(count_per_tool):
            tmpl = random.choice(tmpls)
            if tool_name in (
                "Get Time", "Get Date", "Get Battery", "Get Volume", "List Running Apps",
                "Play Media", "Pause Media", "Next Track", "Previous Track"
            ):
                user_text = tmpl
                response = f"Category: {cat}\nAction: {tool_name}\nArgument: None"
            elif tool_name == "Play Query":
                q = random.choice(ptd.TRAIN_PLAY_QUERIES)
                user_text = tmpl.format(query=q)
                response = f"Category: {cat}\nAction: Play Query\nArgument: {q}"
            elif tool_name == "Open App":
                app = random.choice(ptd.TRAIN_APPS)
                user_text = tmpl.format(app=app)
                clean_arg = app.replace(" app", "").replace(" the ", "").replace(" window", "").strip()
                response = f"Category: {cat}\nAction: Open App\nArgument: {clean_arg}"
            elif tool_name == "Close App":
                app = random.choice(ptd.TRAIN_APPS)
                user_text = tmpl.format(app=app)
                clean_arg = app.replace(" app", "").replace(" the ", "").replace(" window", "").strip()
                response = f"Category: {cat}\nAction: Close App\nArgument: {clean_arg}"
            elif tool_name == "Search Web":
                query = random.choice(ptd.TRAIN_QUERIES)
                user_text = tmpl.format(query=query)
                response = f"Category: {cat}\nAction: Search Web\nArgument: {query}"
            elif tool_name == "Set Volume":
                lvl = random.choice(ptd.TRAIN_VOLUME_LEVELS)
                user_text = tmpl.format(level=lvl)
                response = f"Category: {cat}\nAction: Set Volume\nArgument: {lvl}"
            else:
                continue

            prompt = f"Instruction:\n{user_text}\n\nResponse:\n"
            samples.append({"prompt": prompt, "response": response, "category": f"tool_{tool_name.lower().replace(' ', '_')}"})

    return samples


def main():
    random.seed(42)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    print("=" * 60)
    print("GENERATING HIERARCHICAL SFT DATASET (14 TOOLS)")
    print("=" * 60)

    tool_samples = generate_hierarchical_samples(count_per_tool=180)
    replay_samples = ptd.load_conversational_replay(dolly_count=3300, multiturn_count=1800)

    all_samples = tool_samples + replay_samples
    random.shuffle(all_samples)

    valid_samples = []
    category_counts = {}
    for s in all_samples:
        full_text = s["prompt"] + s["response"]
        tok_ids = tok.encode(full_text, add_bos=True, add_eos=True)
        if len(tok_ids) <= 220:
            valid_samples.append(s)
            c = s["category"]
            category_counts[c] = category_counts.get(c, 0) + 1

    total_valid = len(valid_samples)
    tool_count = sum(c for k, c in category_counts.items() if k.startswith("tool_"))
    replay_count = sum(c for k, c in category_counts.items() if k.startswith("replay_"))

    print(f"Total Valid Samples: {total_valid:,}")
    print(f"  Tool Samples  : {tool_count:,} ({tool_count / total_valid * 100:.1f}%)")
    print(f"  Replay Samples: {replay_count:,} ({replay_count / total_valid * 100:.1f}%)")
    print(f"  Ratio (Replay:Tool): {replay_count / max(1, tool_count):.2f}:1")

    n_val = int(total_valid * 0.05)
    val_set = valid_samples[:n_val]
    train_set = valid_samples[n_val:]

    out_dir = Path("data/processed/sft_hierarchical")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    print(f"Saved hierarchical datasets to {out_dir}:")
    print(f"  Train: {len(train_set):,} samples | Val: {len(val_set):,} samples")


if __name__ == "__main__":
    main()
