"""
eval/test_app_routing_ambiguity.py — Phase 10 Ambiguity Stress-Testing & Real App Execution Audit

Tests whether Phase 9's BM25 lexical pre-filtering holds up or degrades when candidate tools
share high semantic/lexical overlap across 17 tools:
  - Open App vs Find App vs Restart App vs Focus App vs List Running Apps vs Close App
  - Search Web vs Find App
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from tools.dispatcher import (
    TOOL_DESCRIPTORS,
    TOOL_REGISTRY,
    RiskLevel,
    ToolCall,
    _bm25_rank,
    dispatch_tool,
    extract_deterministic_slot,
    parse_tool_call,
    resolve_intent_anchor,
)


def load_model(ckpt_path: str, config_path: str, device: torch.device) -> Tuple[MikuLM, MikuTokenizer]:
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    with open(config_path, "r", encoding="utf-8") as f:
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, tok


def generate_response(
    model: MikuLM,
    tokenizer: MikuTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 35,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_id,
        )
    gen_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(gen_ids).strip()


AMBIGUITY_TEST_CASES = [
    # 1. Distinct Clear Queries for New Tools
    {
        "id": "clear_find_1",
        "instruction": "Where is blender installed on this computer?",
        "expected_action": "Find App",
        "expected_arg": "blender",
        "ambiguity_type": "clear_new_tool",
    },
    {
        "id": "clear_find_2",
        "instruction": "Locate the discord application on my machine.",
        "expected_action": "Find App",
        "expected_arg": "discord",
        "ambiguity_type": "clear_new_tool",
    },
    {
        "id": "clear_restart_1",
        "instruction": "Relaunch slack because it stopped responding.",
        "expected_action": "Restart App",
        "expected_arg": "slack",
        "ambiguity_type": "clear_new_tool",
    },
    {
        "id": "clear_restart_2",
        "instruction": "Reboot the visual studio code process.",
        "expected_action": "Restart App",
        "expected_arg": "visual studio code",
        "ambiguity_type": "clear_new_tool",
    },
    {
        "id": "clear_focus_1",
        "instruction": "Switch to the active chrome window.",
        "expected_action": "Focus App",
        "expected_arg": "chrome",
        "ambiguity_type": "clear_new_tool",
    },
    {
        "id": "clear_focus_2",
        "instruction": "Bring calculator to the foreground.",
        "expected_action": "Focus App",
        "expected_arg": "calculator",
        "ambiguity_type": "clear_new_tool",
    },

    # 2. High-Ambiguity Stress-Test Boundary Cases
    {
        "id": "ambig_open_vs_find",
        "instruction": "Get chrome running on my PC.",
        "expected_action": "Open App",
        "expected_arg": "chrome",
        "ambiguity_type": "high_overlap_open_find_running",
    },
    {
        "id": "ambig_restart_vs_close",
        "instruction": "Restart my browser.",
        "expected_action": "Restart App",
        "expected_arg": "browser",
        "ambiguity_type": "high_overlap_restart_close",
    },
    {
        "id": "ambig_find_vs_web",
        "instruction": "Find where spotify is installed.",
        "expected_action": "Find App",
        "expected_arg": "spotify",
        "ambiguity_type": "high_overlap_find_web",
    },
    {
        "id": "ambig_bring_up_focus_open",
        "instruction": "Bring up notepad right now.",
        "expected_action": "Open App",
        "expected_arg": "notepad",
        "ambiguity_type": "high_overlap_focus_open",
    },
    {
        "id": "ambig_search_pc_vs_web",
        "instruction": "Search for chrome on my computer.",
        "expected_action": "Find App",
        "expected_arg": "chrome",
        "ambiguity_type": "high_overlap_search_web_find",
    },
    {
        "id": "ambig_kill_relaunch",
        "instruction": "Kill and restart slack.",
        "expected_action": "Restart App",
        "expected_arg": "slack",
        "ambiguity_type": "high_overlap_kill_restart",
    },
    {
        "id": "ambig_check_installed",
        "instruction": "Discover which software is installed on this machine.",
        "expected_action": "Find App",
        "expected_arg": "software",
        "ambiguity_type": "high_overlap_find_running",
    },
    {
        "id": "ambig_switch_window",
        "instruction": "Switch over to the terminal window.",
        "expected_action": "Focus App",
        "expected_arg": "terminal",
        "ambiguity_type": "high_overlap_switch_open",
    },
    {
        "id": "ambig_cycle_app",
        "instruction": "Cycle and bounce the terminal app.",
        "expected_action": "Restart App",
        "expected_arg": "terminal",
        "ambiguity_type": "high_overlap_cycle_restart",
    },
    {
        "id": "ambig_fire_up_app",
        "instruction": "Fire up the calculator application.",
        "expected_action": "Open App",
        "expected_arg": "calculator",
        "ambiguity_type": "high_overlap_fireup_app",
    },
]


def run_ambiguity_audit(checkpoint_path: str, config_path: str):
    print("=" * 80)
    print("PHASE 10 AUDIT: APP-ROUTING & AMBIGUITY STRESS-TEST ACROSS 17 TOOLS")
    print("=" * 80)
    print(f"Total registered tools: {len(TOOL_REGISTRY)}")
    print(f"Total tool descriptors: {len(TOOL_DESCRIPTORS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(checkpoint_path, config_path, device)

    bm25_top1_passes = 0
    bm25_top3_passes = 0
    neural_passes = 0
    anchored_passes = 0
    slot_passes = 0

    results = []

    print(f"\nEvaluating {len(AMBIGUITY_TEST_CASES)} High-Ambiguity App Test Cases...\n")

    for case in AMBIGUITY_TEST_CASES:
        inst = case["instruction"]
        exp_action = case["expected_action"]
        exp_arg = case["expected_arg"]

        # 1. BM25 Lexical Ranking
        ranked = _bm25_rank(inst)
        top1_tool, top1_score = ranked[0]
        top3_tools = [t for t, s in ranked[:3] if s > 0]
        bm25_top1_ok = (top1_tool == exp_action)
        bm25_top3_ok = (exp_action in top3_tools)

        if bm25_top1_ok:
            bm25_top1_passes += 1
        if bm25_top3_ok:
            bm25_top3_passes += 1

        # 2. Neural Generation
        prompt = f"Instruction:\n{inst}\n\nResponse:\n"
        raw_gen = generate_response(model, tok, prompt, device)
        tc = parse_tool_call(raw_gen)
        neural_ok = (tc.action == exp_action)
        if neural_ok:
            neural_passes += 1

        # 3. Hybrid Anchored Resolution
        anchored_action = resolve_intent_anchor(inst, tc.action)
        anchored_ok = (anchored_action == exp_action)
        if anchored_ok:
            anchored_passes += 1

        # 4. Slot Extraction
        det_arg = extract_deterministic_slot(inst, exp_action)
        slot_ok = (det_arg is not None and det_arg.lower() == exp_arg.lower())
        if slot_ok:
            slot_passes += 1

        print(f"[{case['id']}] '{inst}'")
        print(f"  Expected: '{exp_action}' (Arg: '{exp_arg}')")
        print(f"  BM25 Rank Top 3: {[(t, round(s, 2)) for t, s in ranked[:3]]}")
        print(f"  BM25 Top-1 Match: {bm25_top1_ok} | BM25 In Top-3: {bm25_top3_ok}")
        print(f"  Neural Parsed: '{tc.action}' (OK: {neural_ok}) | Anchored: '{anchored_action}' (OK: {anchored_ok})")
        print(f"  Slot Fill: '{det_arg}' (OK: {slot_ok})")
        print("-" * 75)

        results.append({
            "id": case["id"],
            "instruction": inst,
            "expected_action": exp_action,
            "expected_arg": exp_arg,
            "bm25_top3": [(t, float(s)) for t, s in ranked[:3]],
            "bm25_top1_ok": bm25_top1_ok,
            "bm25_top3_ok": bm25_top3_ok,
            "neural_action": tc.action,
            "neural_ok": neural_ok,
            "anchored_action": anchored_action,
            "anchored_ok": anchored_ok,
            "slot_extracted": det_arg,
            "slot_ok": slot_ok,
        })

    total = len(AMBIGUITY_TEST_CASES)
    print("\n" + "=" * 80)
    print("PHASE 10 AMBIGUITY STRESS-TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"BM25 Top-1 Lexical Accuracy : {bm25_top1_passes}/{total} ({(bm25_top1_passes/total)*100:.1f}%)")
    print(f"BM25 Top-3 Candidate Recall : {bm25_top3_passes}/{total} ({(bm25_top3_passes/total)*100:.1f}%)")
    print(f"Pure Neural Action Match    : {neural_passes}/{total} ({(neural_passes/total)*100:.1f}%)")
    print(f"Hybrid Anchored Action Match: {anchored_passes}/{total} ({(anchored_passes/total)*100:.1f}%)")
    print(f"Deterministic Slot Match    : {slot_passes}/{total} ({(slot_passes/total)*100:.1f}%)")
    print("=" * 80)

    # 4. Real Execution Verification & HITL Safety Audit
    print("\n" + "=" * 80)
    print("PHASE 10 REAL EXECUTION & HITL SAFETY VERIFICATION")
    print("=" * 80)

    # Test 4a: Unconfirmed Open App (HIGH risk -> MUST block)
    unconf_call = ToolCall(action="Open App", argument="notepad", risk=RiskLevel.HIGH, is_valid=True)
    res_blocked = dispatch_tool(unconf_call, hitl_confirmed=False)
    print("1. Unconfirmed High-Risk Call ('Open App(notepad)'):")
    print(f"   Status: {res_blocked.status} (Executed: {res_blocked.executed})")
    print(f"   Message: {res_blocked.output}")
    assert res_blocked.status == "DRY_RUN_PENDING_HITL", "Safety breach: unconfirmed high-risk call executed!"

    # Test 4b: Real Live App Search ('Find App(calc)' -> LOW risk -> Executes live)
    find_call = ToolCall(action="Find App", argument="calc", risk=RiskLevel.LOW, is_valid=True)
    res_find = dispatch_tool(find_call, hitl_confirmed=False)
    print("\n2. Live Low-Risk App Search ('Find App(calc)'):")
    print(f"   Status: {res_find.status} (Executed: {res_find.executed})")
    print(f"   Message: {res_find.output}")
    assert res_find.status == "SUCCESS" and res_find.executed, "Live app search failed!"

    # Test 4c: Confirmed Open App ('Open App(notepad)' -> HIGH risk + confirmed=True -> Launches real process)
    print("\n3. Confirmed High-Risk App Launch ('Open App(notepad)' with confirmed=True):")
    conf_call = ToolCall(action="Open App", argument="notepad", risk=RiskLevel.HIGH, is_valid=True)
    res_launch = dispatch_tool(conf_call, hitl_confirmed=True)
    print(f"   Status: {res_launch.status} (Executed: {res_launch.executed})")
    print(f"   Message: {res_launch.output}")
    assert res_launch.status == "SUCCESS" and res_launch.executed, "Real app launch failed under confirmation!"

    # Give OS a moment to register window
    time.sleep(1.0)

    # Test 4d: Live Low-Risk Window Focus ('Focus App(notepad)' -> LOW risk -> Focuses window)
    focus_call = ToolCall(action="Focus App", argument="notepad", risk=RiskLevel.LOW, is_valid=True)
    res_focus = dispatch_tool(focus_call, hitl_confirmed=False)
    print("\n4. Live Low-Risk Window Focus ('Focus App(notepad)'):")
    print(f"   Status: {res_focus.status} (Executed: {res_focus.executed})")
    print(f"   Message: {res_focus.output}")

    # Clean up test notepad process
    print("\nCleaning up test notepad process...")
    subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Cleanup complete.")

    # Save results to log
    os.makedirs("logs/phase10_app_routing", exist_ok=True)
    log_path = "logs/phase10_app_routing/ambiguity_audit_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total": total,
                "bm25_top1_passes": bm25_top1_passes,
                "bm25_top3_passes": bm25_top3_passes,
                "neural_passes": neural_passes,
                "anchored_passes": anchored_passes,
                "slot_passes": slot_passes,
            },
            "cases": results,
            "real_execution": {
                "blocked": res_blocked.output,
                "find": res_find.output,
                "launch": res_launch.output,
                "focus": res_focus.output,
            }
        }, f, indent=2)
    print(f"\nSaved detailed Phase 10 audit log to {log_path}")


if __name__ == "__main__":
    ckpt = "checkpoints/phase8_tool_sft/canonical_media_tool_sft.pt"
    cfg = "configs/phase8_media_tool_sft.yaml"
    run_ambiguity_audit(ckpt, cfg)
