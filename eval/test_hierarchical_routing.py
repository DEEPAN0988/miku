"""
eval/test_hierarchical_routing.py — Benchmark Hierarchical Two-Stage Routing on 14 OOD Suite.

Evaluates:
  1. Broad Category Match Rate (System Info, Media Control, App Management, Device Settings, Web Search)
  2. Specific Tool Action Match Rate (compared directly to the flat 42.9% baseline)
  3. Verbatim per-case generation logs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.prepare_hierarchical_data import TOOL_TO_CATEGORY

OOD_TEST_CASES = [
    {"instruction": "Do you have the time on you?", "expected_action": "Get Time"},
    {"instruction": "What's on the calendar for today's date?", "expected_action": "Get Date"},
    {"instruction": "Can you see how much power is remaining in the battery?", "expected_action": "Get Battery"},
    {"instruction": "Could you inspect the audio level on my machine?", "expected_action": "Get Volume"},
    {"instruction": "Give me an inventory of open software.", "expected_action": "List Running Apps"},
    {"instruction": "Get the audio moving again.", "expected_action": "Play Media"},
    {"instruction": "Put the tunes on hold.", "expected_action": "Pause Media"},
    {"instruction": "Forward to the subsequent tune.", "expected_action": "Next Track"},
    {"instruction": "Revisit the preceding title.", "expected_action": "Previous Track"},
    {"instruction": "Cue up bohemian rhapsody.", "expected_action": "Play Query"},
    {"instruction": "Fire up the calculator.", "expected_action": "Open App"},
    {"instruction": "End the terminal process.", "expected_action": "Close App"},
    {"instruction": "Tune the audio output to 50%.", "expected_action": "Set Volume"},
    {"instruction": "Search online to find out about weather forecast.", "expected_action": "Search Web"},
]


def parse_hierarchical_output(text: str) -> Tuple[str, str, str]:
    cleaned = text.strip()
    cat_m = re.search(r"Category\s*:\s*(.*?)(?:\n|\r|\s+Action\s*:|$)", cleaned, re.I)
    act_m = re.search(r"Action\s*:\s*(.*?)(?:\n|\r|\s+Argument\s*:|$)", cleaned, re.I)
    arg_m = re.search(r"Argument\s*:\s*(.*)", cleaned, re.I)

    cat = cat_m.group(1).strip() if cat_m else ""
    act = act_m.group(1).strip() if act_m else ""
    arg = arg_m.group(1).strip() if arg_m else ""
    return cat, act, arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/hierarchical_tool_sft/canonical_hierarchical_sft.pt")
    parser.add_argument("--config", default="configs/hierarchical_tool_sft.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Evaluating Hierarchical Routing on device: {device}")

    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mc = ModelConfig(
        vocab_size=cfg["model"]["vocab_size"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"].get("dropout", 0.0),
        weight_tie_embeddings=cfg["model"].get("weight_tie_embeddings", True),
        rope_theta=cfg["model"].get("rope_theta", 10000.0),
    )
    model = MikuLM(mc).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    print("=" * 75)
    print("HIERARCHICAL TWO-STAGE ROUTING EVALUATION (14 OOD CASES)")
    print("=" * 75)

    cat_matches = 0
    act_matches = 0

    for i, item in enumerate(OOD_TEST_CASES, 1):
        q = item["instruction"]
        exp_act = item["expected_action"]
        exp_cat = TOOL_TO_CATEGORY[exp_act]

        prompt = f"Instruction:\n{q}\n\nResponse:\n"
        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=40,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tok.eos_id,
            )
        raw_gen = tok.decode(out[0, len(ids):].tolist()).strip()
        cat, act, arg = parse_hierarchical_output(raw_gen)

        cat_ok = (cat.lower() == exp_cat.lower())
        act_ok = (act.lower() == exp_act.lower())

        if cat_ok:
            cat_matches += 1
        if act_ok:
            act_matches += 1

        print(f"\nCase {i:2d}: \"{q}\"")
        print(f"  Expected: Category='{exp_cat}' | Action='{exp_act}'")
        print(f"  Raw Generated:\n    {raw_gen}")
        print(f"  Parsed:   Category='{cat}' | Action='{act}'")
        print(f"  Score:    Category Match: {cat_ok} | Action Match: {act_ok}")
        print("-" * 65)

    n = len(OOD_TEST_CASES)
    print("\n" + "=" * 75)
    print(f"HIERARCHICAL ROUTING RESULTS SUMMARY (N={n}):")
    print(f"  Broad Category Match Rate: {cat_matches}/{n} ({cat_matches/n*100:.1f}%)")
    print(f"  Specific Action Match Rate: {act_matches}/{n} ({act_matches/n*100:.1f}%)")
    print(f"  Baseline Flat Action Match: 6/{n} (42.9%)")
    diff = (act_matches/n*100) - 42.9
    sign = "+" if diff >= 0 else ""
    print(f"  Net Delta vs Flat Baseline: {sign}{diff:.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
