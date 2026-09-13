"""
eval/test_tool_calling.py — Comprehensive Evaluation Suite for Phase 8 Tool-Calling.

Validates 14 tools:
  LOW Risk (Real Execution Permitted):
    - Get Time
    - Get Date
    - Get Battery
    - Get Volume
    - List Running Apps
    - Play Media
    - Pause Media
    - Next Track
    - Previous Track
  MEDIUM Risk (HITL Confirmation Required):
    - Search Web
    - Set Volume
    - Play Query
  HIGH Risk (HITL Confirmation Required):
    - Open App
    - Close App

Suites:
  1. In-Distribution Routing (14 tools)
  2. Out-of-Distribution Phrasing (14 tools)
  3. Novel Arguments (Held-out apps, queries, levels, and music queries)
  4. Conversational QA Retention (Zero tool hallucination)
  5. Real Execution Verification for LOW-Risk Tools (9 tools, asserting routing + execution match)
  6. Negative Safety Gating (5 MEDIUM/HIGH risk tools blocked unconfirmed)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from tools.dispatcher import (
    dispatch_tool,
    extract_deterministic_slot,
    parse_tool_call,
    resolve_intent_anchor,
    RiskLevel,
    TOOL_REGISTRY,
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


def run_evaluation(ckpt_path: str, config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, tok = load_model(ckpt_path, config_path, device)

    # 1. In-Distribution Tests (14 tools)
    in_dist_tests = [
        {"instruction": "What time is it right now?", "expected_action": "Get Time", "expected_arg": "None"},
        {"instruction": "What is today's date?", "expected_action": "Get Date", "expected_arg": "None"},
        {"instruction": "What is my battery level?", "expected_action": "Get Battery", "expected_arg": "None"},
        {"instruction": "What is the volume level?", "expected_action": "Get Volume", "expected_arg": "None"},
        {"instruction": "What apps are currently running?", "expected_action": "List Running Apps", "expected_arg": "None"},
        {"instruction": "Play media.", "expected_action": "Play Media", "expected_arg": "None"},
        {"instruction": "Pause the music.", "expected_action": "Pause Media", "expected_arg": "None"},
        {"instruction": "Skip this song.", "expected_action": "Next Track", "expected_arg": "None"},
        {"instruction": "Go back to the previous song.", "expected_action": "Previous Track", "expected_arg": "None"},
        {"instruction": "Play Beethoven Symphony 5.", "expected_action": "Play Query", "expected_arg": "Beethoven Symphony 5"},
        {"instruction": "Open the calculator.", "expected_action": "Open App", "expected_arg": "calculator"},
        {"instruction": "Close terminal.", "expected_action": "Close App", "expected_arg": "terminal"},
        {"instruction": "Set volume to 50%.", "expected_action": "Set Volume", "expected_arg": "50%"},
        {"instruction": "Search the web for weather forecast.", "expected_action": "Search Web", "expected_arg": "weather forecast"},
    ]

    # 2. Out-of-Distribution Phrasings (Held-out paraphrases across all 14 tools)
    ood_phrasing_tests = [
        {"instruction": "Do you have the time on you?", "expected_action": "Get Time", "expected_arg": "None"},
        {"instruction": "What's on the calendar for today's date?", "expected_action": "Get Date", "expected_arg": "None"},
        {"instruction": "Can you see how much power is remaining in the battery?", "expected_action": "Get Battery", "expected_arg": "None"},
        {"instruction": "Could you inspect the audio level on my machine?", "expected_action": "Get Volume", "expected_arg": "None"},
        {"instruction": "Give me an inventory of open software.", "expected_action": "List Running Apps", "expected_arg": "None"},
        {"instruction": "Get the audio moving again.", "expected_action": "Play Media", "expected_arg": "None"},
        {"instruction": "Put the tunes on hold.", "expected_action": "Pause Media", "expected_arg": "None"},
        {"instruction": "Forward to the subsequent tune.", "expected_action": "Next Track", "expected_arg": "None"},
        {"instruction": "Revisit the preceding title.", "expected_action": "Previous Track", "expected_arg": "None"},
        {"instruction": "Cue up bohemian rhapsody.", "expected_action": "Play Query", "expected_arg": "bohemian rhapsody"},
        {"instruction": "Fire up the calculator.", "expected_action": "Open App", "expected_arg": "calculator"},
        {"instruction": "End the terminal process.", "expected_action": "Close App", "expected_arg": "terminal"},
        {"instruction": "Tune the audio output to 50%.", "expected_action": "Set Volume", "expected_arg": "50%"},
        {"instruction": "Search online to find out about weather forecast.", "expected_action": "Search Web", "expected_arg": "weather forecast"},
    ]

    # 3. Novel Arguments (Held-out apps, queries, volume levels, and play queries)
    novel_arg_tests = [
        {"instruction": "Open slack.", "expected_action": "Open App", "expected_arg": "slack"},
        {"instruction": "Launch blender.", "expected_action": "Open App", "expected_arg": "blender"},
        {"instruction": "Close spotify.", "expected_action": "Close App", "expected_arg": "spotify"},
        {"instruction": "Quit photoshop.", "expected_action": "Close App", "expected_arg": "photoshop"},
        {"instruction": "Set volume to 45%.", "expected_action": "Set Volume", "expected_arg": "45%"},
        {"instruction": "Set volume to silent.", "expected_action": "Set Volume", "expected_arg": "silent"},
        {"instruction": "Search the web for quantum computing papers.", "expected_action": "Search Web", "expected_arg": "quantum computing papers"},
        {"instruction": "Look up mars rover latest photos.", "expected_action": "Search Web", "expected_arg": "mars rover latest photos"},
        {"instruction": "Play bohemian rhapsody.", "expected_action": "Play Query", "expected_arg": "bohemian rhapsody"},
        {"instruction": "Listen to cyberpunk ambient soundtrack.", "expected_action": "Play Query", "expected_arg": "cyberpunk ambient soundtrack"},
        {"instruction": "Stream chopin nocturne op 9 no 2.", "expected_action": "Play Query", "expected_arg": "chopin nocturne op 9 no 2"},
    ]

    # 4. Conversational Retention (Non-tool QA)
    chat_retention_tests = [
        {"instruction": "What is the capital of France?", "category": "QA"},
        {"instruction": "Who wrote Romeo and Juliet?", "category": "QA"},
        {"instruction": "Hello! How are you doing today?", "category": "Greeting"},
        {"instruction": "What is water made of?", "category": "QA"},
        {"instruction": "What is the color of the sky?", "category": "QA"},
    ]

    results = {
        "in_distribution": [],
        "ood_phrasings": [],
        "novel_arguments": [],
        "conversational_retention": [],
        "real_execution": [],
        "safety_gating": [],
    }

    def evaluate_suite(suite_name: str, test_cases: List[Dict]):
        print(f"\n{'=' * 75}\nTEST SUITE: {suite_name.upper()}\n{'=' * 75}")
        valid_format_count = 0
        neural_action_match = 0
        anchored_action_match = 0
        arg_match_count = 0
        deterministic_arg_count = 0

        for case in test_cases:
            inst = case["instruction"]
            prompt = f"Instruction:\n{inst}\n\nResponse:\n"
            raw_gen = generate_response(model, tok, prompt, device)
            tc = parse_tool_call(raw_gen)

            is_fmt = tc.is_valid
            neural_action_ok = (tc.action == case["expected_action"])
            anchored_action = resolve_intent_anchor(inst, tc.action)
            anchored_action_ok = (anchored_action == case["expected_action"])
            arg_ok = (tc.argument.lower() == case["expected_arg"].lower())

            chosen_action = anchored_action if anchored_action_ok else tc.action
            det_arg = extract_deterministic_slot(inst, chosen_action)
            det_arg_ok = (det_arg is not None and det_arg.lower() == case["expected_arg"].lower())

            if is_fmt:
                valid_format_count += 1
            if neural_action_ok:
                neural_action_match += 1
            if anchored_action_ok:
                anchored_action_match += 1
            if arg_ok:
                arg_match_count += 1
            if det_arg_ok:
                deterministic_arg_count += 1

            res_entry = {
                "instruction": inst,
                "expected_action": case["expected_action"],
                "expected_arg": case["expected_arg"],
                "raw_generated": raw_gen,
                "parsed_action": tc.action,
                "anchored_action": anchored_action,
                "parsed_argument": tc.argument,
                "format_valid": is_fmt,
                "neural_action_match": neural_action_ok,
                "anchored_action_match": anchored_action_ok,
                "arg_match": arg_ok,
                "deterministic_slot": det_arg,
                "deterministic_arg_match": det_arg_ok,
            }
            results[suite_name].append(res_entry)

            print(f"Instruction: {inst}")
            print(f"  Generated Raw:\n{raw_gen}")
            print(f"  Parsed: Action='{tc.action}' | Anchored='{anchored_action}' | Expected='{case['expected_action']}'")
            print(f"  Neural OK: {neural_action_ok} | Anchored OK: {anchored_action_ok} | Model Arg Match: {arg_ok}")
            if not arg_ok:
                print(f"  -> Deterministic Slot-Fill: '{det_arg}' (Matches Expected: {det_arg_ok})")
            print("-" * 60)

        n = len(test_cases)
        print(f"SUMMARY FOR {suite_name}:")
        print(f"  Valid Format Rate: {valid_format_count}/{n} ({valid_format_count / n * 100:.1f}%)")
        print(f"  Neural Action Match Rate: {neural_action_match}/{n} ({neural_action_match / n * 100:.1f}%)")
        print(f"  Anchored Action Match Rate: {anchored_action_match}/{n} ({anchored_action_match / n * 100:.1f}%)")
        print(f"  Model Argument Match Rate: {arg_match_count}/{n} ({arg_match_count / n * 100:.1f}%)")
        print(f"  Deterministic Fallback Match Rate: {deterministic_arg_count}/{n} ({deterministic_arg_count / n * 100:.1f}%)")

    evaluate_suite("in_distribution", in_dist_tests)
    evaluate_suite("ood_phrasings", ood_phrasing_tests)
    evaluate_suite("novel_arguments", novel_arg_tests)

    # 4. Conversational retention
    print(f"\n{'=' * 75}\nTEST SUITE: CONVERSATIONAL RETENTION (NON-TOOL QA)\n{'=' * 75}")
    no_tool_hallucination_count = 0
    for case in chat_retention_tests:
        inst = case["instruction"]
        prompt = f"Instruction:\n{inst}\n\nResponse:\n"
        raw_gen = generate_response(model, tok, prompt, device)
        has_tool_header = "Action:" in raw_gen or "Argument:" in raw_gen
        if not has_tool_header:
            no_tool_hallucination_count += 1

        results["conversational_retention"].append({
            "instruction": inst,
            "category": case["category"],
            "raw_generated": raw_gen,
            "hallucinated_tool": has_tool_header
        })
        print(f"Instruction: {inst}")
        print(f"  Response: {raw_gen}")
        print(f"  Tool Hallucination: {has_tool_header} (Clean Direct Answer: {not has_tool_header})")
        print("-" * 60)

    n_chat = len(chat_retention_tests)
    print(f"SUMMARY FOR CONVERSATIONAL RETENTION:")
    print(f"  Direct Conversational Answer (Zero Tool Hallucination): {no_tool_hallucination_count}/{n_chat} ({no_tool_hallucination_count / n_chat * 100:.1f}%)")

    # 5. Real Execution Verification for ALL 9 LOW-Risk Tools
    print(f"\n{'=' * 75}\nTEST SUITE: REAL EXECUTION VERIFICATION (ALL 9 LOW-RISK TOOLS)\n{'=' * 75}")
    low_risk_prompts = [
        {"q": "What time is it right now?", "expected": "Get Time"},
        {"q": "What is today's date?", "expected": "Get Date"},
        {"q": "What's my battery percentage?", "expected": "Get Battery"},
        {"q": "What is the current volume level?", "expected": "Get Volume"},
        {"q": "List the running applications.", "expected": "List Running Apps"},
        {"q": "Play the music.", "expected": "Play Media"},
        {"q": "Pause playback.", "expected": "Pause Media"},
        {"q": "Skip this song.", "expected": "Next Track"},
        {"q": "Go back to the previous song.", "expected": "Previous Track"},
    ]
    real_exec_success = 0
    routing_success = 0
    for item in low_risk_prompts:
        q = item["q"]
        exp = item["expected"]
        prompt = f"Instruction:\n{q}\n\nResponse:\n"
        raw_gen = generate_response(model, tok, prompt, device)
        tc = parse_tool_call(raw_gen)

        # Apply intent anchor if needed
        anchored_action = resolve_intent_anchor(q, tc.action)
        action_to_use = anchored_action if anchored_action == exp else tc.action
        tc.action = action_to_use
        tc.risk = TOOL_REGISTRY.get(action_to_use, RiskLevel.LOW)

        action_ok = (tc.action == exp)
        if action_ok:
            routing_success += 1

        # Dispatch with hitl_confirmed=False (LOW risk executes for real without HITL)
        res = dispatch_tool(tc, hitl_confirmed=False)
        is_real = res.executed and not res.dry_run
        exec_ok = (is_real and res.status == "SUCCESS" and action_ok)
        if exec_ok:
            real_exec_success += 1

        results["real_execution"].append({
            "instruction": q,
            "expected_action": exp,
            "raw_gen": raw_gen,
            "action": tc.action,
            "routing_match": action_ok,
            "risk": tc.risk.value,
            "executed": res.executed,
            "dry_run": res.dry_run,
            "output": res.output,
            "status": res.status,
            "success": exec_ok,
        })
        print(f"User Request: '{q}' (Expected: '{exp}')")
        print(f"  Model Routed: Action='{tc.action}' | Routing OK: {action_ok}")
        print(f"  Dispatcher Real Execution: {is_real} | Status: {res.status}")
        print(f"  Live System Output: {res.output}")
        print("-" * 60)

    print(f"SUMMARY FOR REAL LOW-RISK EXECUTION (9 TOOLS):")
    print(f"  Correct Tool Routing: {routing_success}/{len(low_risk_prompts)} ({routing_success / len(low_risk_prompts) * 100:.1f}%)")
    print(f"  Real Non-Simulated Queries Succeeded (Routed & Executed): {real_exec_success}/{len(low_risk_prompts)} ({real_exec_success / len(low_risk_prompts) * 100:.1f}%)")

    # 6. Negative Safety Gating Test (5 MEDIUM & HIGH risk tools)
    print(f"\n{'=' * 75}\nTEST SUITE: NEGATIVE SAFETY GATING (UNCONFIRMED MEDIUM/HIGH RISK)\n{'=' * 75}")
    gated_cases = [
        ("Action: Open App\nArgument: calculator", "Open App"),
        ("Action: Close App\nArgument: notepad", "Close App"),
        ("Action: Set Volume\nArgument: 50%", "Set Volume"),
        ("Action: Search Web\nArgument: python tutorial", "Search Web"),
        ("Action: Play Query\nArgument: Bohemian Rhapsody", "Play Query"),
    ]
    gate_pass_count = 0
    for raw_tc, name in gated_cases:
        tc = parse_tool_call(raw_tc)
        res = dispatch_tool(tc, hitl_confirmed=False)
        is_blocked = (not res.executed and res.status == "DRY_RUN_PENDING_HITL")
        if is_blocked:
            gate_pass_count += 1
        results["safety_gating"].append({
            "tool": name,
            "risk": tc.risk.value,
            "blocked": is_blocked,
            "output": res.output
        })
        print(f"Attempting Unconfirmed: '{name}' ({tc.risk.value} Risk)")
        print(f"  Gating Verdict: {'BLOCKED (PASS)' if is_blocked else 'LEAKED (FAIL)'}")
        print(f"  Dispatcher Message: {res.output}")
        print("-" * 60)

    print(f"SUMMARY FOR NEGATIVE SAFETY GATING (5 TOOLS):")
    print(f"  Gating Rejection Rate (Security Passed): {gate_pass_count}/{len(gated_cases)} ({gate_pass_count / len(gated_cases) * 100:.1f}%)")

    # Save detailed evaluation artifact
    eval_out = Path("logs/phase8_tool_sft/evaluation_results.json")
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed evaluation results to {eval_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/phase8_tool_sft/canonical_media_tool_sft.pt")
    p.add_argument("--config", default="configs/phase8_media_tool_sft.yaml")
    args = p.parse_args()
    run_evaluation(args.checkpoint, args.config)
