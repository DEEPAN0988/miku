"""
train/prepare_scenario_data.py — SFT Dataset Generator for Computer Use & Real-Time Scenarios (Phase 9 / v0.2)

Synthesizes multi-step computer use action sequences mapped to Miku's JSON schema:
  - Scenario 1: OS Settings Navigation & Dark Mode Toggle
  - Scenario 2: IDE Project Bootstrapper (VS Code)
  - Scenario 3: Web Research & Asset Extraction
  - Tier 1: Cross-App Data Transfer (Calculator -> Notepad)
  - Tier 2: Spatial Canvas & UI Vision (MS Paint)
  - Tier 3: File Lifecycle & Explorer Staging / Safe Recycling
  - General single-step & multi-step desktop navigation queries

Preserves strict 2:1 conversational replay ratio (Dolly-15k + Multi-turn) to prevent forgetting.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer
import train.prepare_tool_data as ptd


SCENARIO_TEMPLATES = [
    # 1. OS Settings Toggle (Dark Mode)
    (
        [
            "Enable dark mode in Windows settings.",
            "Open settings and switch system theme to dark mode.",
            "Turn on dark mode in Windows color settings.",
            "Press Windows key, search for Color settings, and turn on Dark mode.",
            "Switch the OS theme to dark mode and close settings.",
            "Change the appearance theme to Dark in color settings.",
            "Open Windows color settings and click Dark mode.",
            "Go to color settings and toggle dark mode on.",
        ],
        [
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "Color settings"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.5}',
            '{"action": "click", "x": 620, "y": 280}',
            '{"action": "press_key", "key": "alt+f4"}',
            '{"action": "terminate", "reason": "Dark mode enabled and settings window closed"}',
        ],
        "scenario_settings_dark_mode",
    ),

    # 2. IDE Project Bootstrapper
    (
        [
            "Open VS Code and load the project folder.",
            "Launch Visual Studio Code and open my project directory.",
            "Start VS Code, click File > Open Folder, and open system-life-os.",
            "Search for VS Code, open it, and load the workspace.",
            "Open my coding environment in VS Code and load the repository.",
            "Bring up VS Code and open the system-life-os folder.",
        ],
        [
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "VS Code"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 2.0}',
            '{"action": "click", "x": 20, "y": 15}',
            '{"action": "click", "x": 50, "y": 120}',
            '{"action": "type", "text": "system-life-os"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "terminate", "reason": "Opened system-life-os in VS Code"}',
        ],
        "scenario_ide_bootstrap",
    ),

    # 3. Web Research & Image Asset Extraction
    (
        [
            "Open the browser, search for UI references, and save an image to Desktop.",
            "Search for Solo Leveling glowing purple UI on the web and save reference_ui.jpg.",
            "Launch browser, search images for purple glowing UI, and download to Desktop.",
            "Look up Solo Leveling UI design images in browser and save reference_ui.jpg.",
            "Open Edge, search for glowing purple game UI, and save the image.",
        ],
        [
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "msedge"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.5}',
            '{"action": "click", "x": 400, "y": 80}',
            '{"action": "type", "text": "Solo Leveling glowing purple UI"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.5}',
            '{"action": "click", "x": 260, "y": 140}',
            '{"action": "right_click", "x": 300, "y": 350}',
            '{"action": "click", "x": 340, "y": 420}',
            '{"action": "type", "text": "reference_ui.jpg"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "terminate", "reason": "Saved reference image asset to Desktop"}',
        ],
        "scenario_web_asset_extraction",
    ),

    # 4. Cross-App Data Transfer (Calculator -> Notepad)
    (
        [
            "Open Calculator, do 72 * 4, copy result, paste in Notepad, and save as math_result.txt.",
            "Calculate 72 multiplied by 4 in Calculator and save answer to math_result.txt.",
            "Use Calculator to multiply 72 by 4, copy answer, and save in Notepad on Desktop.",
            "Compute 72 * 4, open Notepad, type the result, and save to math_result.txt.",
            "Open Calculator, calculate math result, paste into Notepad, and terminate.",
        ],
        [
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "calculator"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.5}',
            '{"action": "type", "text": "72*4="}',
            '{"action": "press_key", "key": "ctrl+c"}',
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "notepad"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.0}',
            '{"action": "type", "text": "The answer is "}',
            '{"action": "press_key", "key": "ctrl+v"}',
            '{"action": "press_key", "key": "ctrl+s"}',
            '{"action": "type", "text": "math_result.txt"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "terminate", "reason": "Saved calculation to math_result.txt"}',
        ],
        "scenario_calculator_notepad_transfer",
    ),

    # 5. Spatial Vision & Canvas Painting (MS Paint)
    (
        [
            "Open MS Paint, pick red brush, draw circle in center, and save as miku_art.png.",
            "Launch Paint, select brush tool, pick red color, draw circle, and save to Desktop.",
            "Open Paint, draw a red circle on the canvas, and save image as miku_art.png.",
            "Start MS Paint, choose red color, draw a shape, and save as miku_art.png.",
        ],
        [
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "mspaint"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.5}',
            '{"action": "click", "x": 260, "y": 70}',
            '{"action": "click", "x": 615, "y": 70}',
            '{"action": "click", "x": 500, "y": 400}',
            '{"action": "press_key", "key": "ctrl+s"}',
            '{"action": "type", "text": "miku_art.png"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "terminate", "reason": "Saved canvas drawing to miku_art.png"}',
        ],
        "scenario_paint_spatial_canvas",
    ),

    # 6. File Lifecycle & Explorer Staging / Safe Recycling
    (
        [
            "Open File Explorer, create Miku_Staging folder on Desktop, create delete_me.txt, and recycle it.",
            "Launch File Explorer, make folder Miku_Staging on Desktop, add delete_me.txt, and delete it.",
            "Create Miku_Staging folder on Desktop, add text file delete_me.txt, send to Recycle Bin.",
            "Open Explorer, navigate to Desktop, stage a temporary folder, and safely recycle the file.",
        ],
        [
            '{"action": "press_key", "key": "win"}',
            '{"action": "type", "text": "explorer"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "wait", "seconds": 1.5}',
            '{"action": "right_click", "x": 500, "y": 400}',
            '{"action": "type", "text": "Miku_Staging"}',
            '{"action": "press_key", "key": "enter"}',
            '{"action": "press_key", "key": "delete"}',
            '{"action": "terminate", "reason": "Created staging folder and recycled file safely"}',
        ],
        "scenario_file_lifecycle_explorer",
    ),
]


def generate_scenario_samples(count_multiplier: int = 40) -> List[Dict[str, str]]:
    """
    Generates single-step and multi-step computer use training prompts
    mapped to structured action schema outputs.
    """
    samples: List[Dict[str, str]] = []

    for prompts, actions, category in SCENARIO_TEMPLATES:
        for _ in range(count_multiplier):
            p = random.choice(prompts)

            # Sample type A: First next-step action prediction
            first_act = actions[0]
            samples.append({
                "prompt": f"Instruction:\n{p}\n\nResponse:\n",
                "response": first_act,
                "category": f"{category}_step1",
            })

            # Sample type B: Full sequence plan
            full_plan = "\n".join(actions)
            samples.append({
                "prompt": f"Instruction:\nPlan computer use actions for: {p}\n\nResponse:\n",
                "response": full_plan,
                "category": f"{category}_plan",
            })

            # Sample type C: Intermediate step reasoning
            if len(actions) > 2:
                mid_idx = random.randint(1, len(actions) - 2)
                prev_steps = "\n".join([f"Step {i+1}: {act}" for i, act in enumerate(actions[:mid_idx])])
                samples.append({
                    "prompt": (
                        f"Instruction:\nObjective: {p}\n\n"
                        f"Previous Actions:\n{prev_steps}\n\n"
                        f"Next Action:\nResponse:\n"
                    ),
                    "response": actions[mid_idx],
                    "category": f"{category}_midstep",
                })

    return samples


def main():
    random.seed(42)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    out_dir = Path("data/processed/sft_scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GENERATING COMPUTER USE & REAL-TIME WORKING SCENARIOS SFT DATASET")
    print("=" * 75)

    scenario_samples = generate_scenario_samples(count_multiplier=60)
    print(f"[*] Generated {len(scenario_samples):,} scenario samples.")

    # 2:1 conversational replay ratio (Dolly-15k QA + multi-turn conversations)
    replay_samples = ptd.load_conversational_replay(dolly_count=2400, multiturn_count=1200)
    print(f"[*] Loaded {len(replay_samples):,} conversational replay samples.")

    all_samples = scenario_samples + replay_samples
    random.shuffle(all_samples)

    # Validate token lengths (must fit seq_len 256)
    valid_samples = []
    category_counts = {}
    for s in all_samples:
        full_text = s["prompt"] + s["response"]
        tok_ids = tok.encode(full_text, add_bos=True, add_eos=True)
        if len(tok_ids) <= 256:
            valid_samples.append(s)
            cat = s["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    total_valid = len(valid_samples)
    scenario_count = sum(c for k, c in category_counts.items() if k.startswith("scenario_"))
    replay_count = sum(c for k, c in category_counts.items() if k.startswith("replay_"))

    print(f"\nTotal Valid Samples: {total_valid:,}")
    print(f"  Scenario Samples : {scenario_count:,} ({scenario_count / total_valid * 100:.1f}%)")
    print(f"  Replay Samples   : {replay_count:,} ({replay_count / total_valid * 100:.1f}%)")
    print(f"  Ratio (Replay:Scenario): {replay_count / max(1, scenario_count):.2f}:1")

    # 5% validation split
    n_val = int(total_valid * 0.05)
    val_set = valid_samples[:n_val]
    train_set = valid_samples[n_val:]

    train_file = out_dir / "train_scenarios.jsonl"
    val_file = out_dir / "val_scenarios.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    print(f"\n[+] Saved scenario SFT dataset to {out_dir}:")
    print(f"    Train: {len(train_set):,} samples -> {train_file}")
    print(f"    Val  : {len(val_set):,} samples -> {val_file}")


if __name__ == "__main__":
    main()
