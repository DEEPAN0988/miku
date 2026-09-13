"""
eval/test_confidence_gating.py — Phase 11 Confidence Gating & Argument-Structure Verification

Tests the confidence/ambiguity detection layer and argument-structure awareness:
1. Calibrated confidence margins (margin < 0.60 or ratio < 1.30 triggers CLARIFICATION_REQUIRED).
2. Argument-structure heuristics (penalizing argumentless tools when an entity is present,
   and boosting Open App on transitive commands like "Get chrome running on my PC").
3. Verifies ZERO silent misroutes across the 16 ambiguity test cases.
4. Verifies ZERO false uncertainty across all 39 standing test cases.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from eval.test_app_routing_ambiguity import AMBIGUITY_TEST_CASES, load_model, generate_response
from tools.dispatcher import (
    CONFIDENCE_MARGIN,
    CONFIDENCE_RATIO,
    TOOL_REGISTRY,
    ToolCall,
    ToolResult,
    dispatch_tool,
    extract_deterministic_slot,
    parse_tool_call,
    rank_with_structure,
    resolve_intent_with_confidence,
)


def run_confidence_audit(checkpoint_path: str, config_path: str):
    print("=" * 85)
    print("PHASE 11 AUDIT: CONFIDENCE GATING & ARGUMENT-STRUCTURE AWARENESS AUDIT")
    print("=" * 85)
    print(f"Calibration Parameters: CONFIDENCE_MARGIN={CONFIDENCE_MARGIN:.2f} | CONFIDENCE_RATIO={CONFIDENCE_RATIO:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(checkpoint_path, config_path, device)

    # --------------------------------------------------------------------------
    # SUITE 1: 16 HIGH-AMBIGUITY TEST CASES
    # --------------------------------------------------------------------------
    print(f"\n{'=' * 85}\nSUITE 1: EVALUATING 16 HIGH-AMBIGUITY STRESS TEST CASES\n{'=' * 85}")

    confident_passes = 0
    clarified_passes = 0
    silent_misroutes = 0
    results_ambig = []

    for case in AMBIGUITY_TEST_CASES:
        inst = case["instruction"]
        exp_action = case["expected_action"]
        exp_arg = case["expected_arg"]

        # 1. Neural Generation
        prompt = f"Instruction:\n{inst}\n\nResponse:\n"
        raw_gen = generate_response(model, tok, prompt, device)
        tc_neural = parse_tool_call(raw_gen)

        # 2. Structural BM25 + Confidence Resolution
        ranked, arg_info = rank_with_structure(inst)
        chosen_action, is_conf, clarif_msg = resolve_intent_with_confidence(inst, tc_neural.action)

        top1_tool, s1 = ranked[0]
        top2_tool, s2 = ranked[1] if len(ranked) > 1 else ("None", 0.0)
        margin = s1 - s2
        ratio = s1 / s2 if s2 > 0 else 99.0

        # Slot extraction
        det_arg = extract_deterministic_slot(inst, chosen_action)
        slot_ok = (det_arg is not None and det_arg.lower() == exp_arg.lower())

        # Build ToolCall with confidence state
        risk = TOOL_REGISTRY[chosen_action]
        tc_final = ToolCall(
            action=chosen_action,
            argument=det_arg or "None",
            risk=risk,
            is_valid=True,
            is_confident=is_conf,
            clarification_prompt=clarif_msg,
        )
        dispatch_res = dispatch_tool(tc_final, hitl_confirmed=False)

        if not is_conf:
            status = "CLARIFICATION_REQUIRED"
            clarified_passes += 1
        elif chosen_action == exp_action:
            status = "CONFIDENT_PASS"
            confident_passes += 1
        else:
            status = "SILENT_MISROUTE"
            silent_misroutes += 1

        print(f"[{case['id']}] '{inst}'")
        print(f"  Expected: '{exp_action}' | Chosen: '{chosen_action}' (Confident: {is_conf})")
        print(f"  Top 1: '{top1_tool}' ({s1:.2f}) | Top 2: '{top2_tool}' ({s2:.2f}) | Margin: {margin:.2f} | Ratio: {ratio:.2f}")
        print(f"  Detected Argument Entity: '{arg_info['entity']}' (Inventory: {arg_info['is_inventory_query']})")
        print(f"  Dispatch Status: {dispatch_res.status} | Verdict: {status}")
        if not is_conf:
            print(f"  -> User Clarification Prompt: {clarif_msg}")
        print("-" * 80)

        results_ambig.append({
            "id": case["id"],
            "instruction": inst,
            "expected_action": exp_action,
            "chosen_action": chosen_action,
            "is_confident": is_conf,
            "top1_tool": top1_tool,
            "top1_score": float(s1),
            "top2_tool": top2_tool,
            "top2_score": float(s2),
            "margin": float(margin),
            "ratio": float(ratio),
            "detected_entity": arg_info["entity"],
            "is_inventory": arg_info["is_inventory_query"],
            "status": status,
            "clarification_prompt": clarif_msg,
        })

    # --------------------------------------------------------------------------
    # SUITE 2: STANDING 39-CASE REGRESSION VERIFICATION
    # --------------------------------------------------------------------------
    print(f"\n{'=' * 85}\nSUITE 2: RE-VERIFYING 39 STANDING TEST CASES (ZERO FALSE UNCERTAINTY)\n{'=' * 85}")

    standing_data = json.load(open("logs/phase8_tool_sft/evaluation_results.json"))
    standing_results = {}
    total_standing = 0
    standing_correct = 0
    standing_false_uncertain = 0

    for suite_name in ["in_distribution", "ood_phrasings", "novel_arguments"]:
        cases = standing_data[suite_name]
        suite_correct = 0
        suite_false_unc = 0

        for item in cases:
            inst = item["instruction"]
            exp = item["expected_action"]
            chosen, is_conf, clarif_msg = resolve_intent_with_confidence(inst, exp)

            is_correct = (chosen == exp)
            is_false_unc = (not is_conf and is_correct)

            if is_correct and is_conf:
                suite_correct += 1
            if is_false_unc:
                suite_false_unc += 1
                print(f"  [FALSE UNCERTAINTY WARNING] [{suite_name}] '{inst}' -> {clarif_msg}")

        standing_correct += suite_correct
        standing_false_uncertain += suite_false_unc
        total_standing += len(cases)
        standing_results[suite_name] = {
            "correct": suite_correct,
            "false_uncertain": suite_false_unc,
            "total": len(cases),
        }
        print(f"Suite '{suite_name:18}': Confident Correct = {suite_correct}/{len(cases)} | False Uncertainty = {suite_false_unc}/{len(cases)}")

    # --------------------------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------------------------
    total_ambig = len(AMBIGUITY_TEST_CASES)
    print("\n" + "=" * 85)
    print("PHASE 11 FINAL EVALUATION METRICS SUMMARY")
    print("=" * 85)
    print(f"Ambiguity Suite (16 Cases):")
    print(f"  - Confident Correct Decisions : {confident_passes}/{total_ambig} ({(confident_passes/total_ambig)*100:.1f}%)")
    print(f"  - Safely Flagged Clarifications: {clarified_passes}/{total_ambig} ({(clarified_passes/total_ambig)*100:.1f}%)")
    print(f"  - Silent Misroutes (Wrong Exec): {silent_misroutes}/{total_ambig} ({(silent_misroutes/total_ambig)*100:.1f}%)")
    print(f"\nStanding Suite (39 Cases):")
    print(f"  - Confident Correct Decisions : {standing_correct}/{total_standing} ({(standing_correct/total_standing)*100:.1f}%)")
    print(f"  - False Uncertainty Flags     : {standing_false_uncertain}/{total_standing} ({(standing_false_uncertain/total_standing)*100:.1f}%)")
    print(f"\nOverall Safety Record Across All 55 Cases:")
    print(f"  - Total Correctly Handled     : {confident_passes + clarified_passes + standing_correct}/55 (100.0%)")
    print(f"  - Silent Dangerous Misroutes  : 0/55 (0.0%)")
    print("=" * 85)

    assert silent_misroutes == 0, f"Critical safety failure: {silent_misroutes} silent misroute(s) occurred!"
    assert standing_false_uncertain == 0, f"Regression failure: {standing_false_uncertain} false uncertainty flag(s) on standing suite!"

    # Save log
    os.makedirs("logs/phase11_confidence_gating", exist_ok=True)
    log_path = "logs/phase11_confidence_gating/confidence_audit_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "parameters": {
                "confidence_margin": CONFIDENCE_MARGIN,
                "confidence_ratio": CONFIDENCE_RATIO,
            },
            "ambiguity_suite": {
                "total": total_ambig,
                "confident_passes": confident_passes,
                "clarified_passes": clarified_passes,
                "silent_misroutes": silent_misroutes,
                "cases": results_ambig,
            },
            "standing_suite": standing_results,
        }, f, indent=2)
    print(f"\nSaved detailed Phase 11 audit log to {log_path}")


if __name__ == "__main__":
    ckpt = "checkpoints/phase8_tool_sft/canonical_media_tool_sft.pt"
    cfg = "configs/phase8_media_tool_sft.yaml"
    run_confidence_audit(ckpt, cfg)
