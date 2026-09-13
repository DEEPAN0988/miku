"""
eval/audit_bm25_v03_scaling.py — Consolidated BM25 Re-Audit across 20 Registered Tools

Fulfills Phase 10's explicit commitment:
"re-run the full OOD-scaling audit again after the next batch of tools rather than assuming BM25 solved this permanently."

Tests:
1. Inter-cluster baseline (14 standing OOD test cases across distinct capability domains)
2. Intra-cluster app-management suite (16 high-ambiguity app cases from Phase 10)
3. Cross-domain messaging overlap suite (16 challenging boundary cases testing lexical collisions
   on "send", "message", "open", "tell", "ping", "search", "draft", "stage", etc.)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dispatcher import (
    CONFIDENCE_MARGIN,
    CONFIDENCE_RATIO,
    TOOL_DESCRIPTORS,
    TOOL_REGISTRY,
    _bm25_rank,
    rank_with_structure,
    resolve_intent_with_confidence,
)

# ------------------------------------------------------------------------------
# 1. INTER-CLUSTER BASELINE SUITE (14 Tools, Phase 8/9 OOD queries)
# ------------------------------------------------------------------------------
INTER_CLUSTER_CASES = [
    {"id": "inter_time", "query": "Do you have the time on you?", "expected": "Get Time", "cluster": "Inter-Cluster Info"},
    {"id": "inter_date", "query": "What's on the calendar for today's date?", "expected": "Get Date", "cluster": "Inter-Cluster Info"},
    {"id": "inter_battery", "query": "Can you see how much power is remaining in the battery?", "expected": "Get Battery", "cluster": "Inter-Cluster Info"},
    {"id": "inter_volume", "query": "Could you inspect the audio level on my machine?", "expected": "Get Volume", "cluster": "Inter-Cluster Info"},
    {"id": "inter_running_apps", "query": "Give me an inventory of open software.", "expected": "List Running Apps", "cluster": "Inter-Cluster Info"},
    {"id": "inter_play_media", "query": "Get the audio moving again.", "expected": "Play Media", "cluster": "Inter-Cluster Media"},
    {"id": "inter_pause_media", "query": "Put the tunes on hold.", "expected": "Pause Media", "cluster": "Inter-Cluster Media"},
    {"id": "inter_next_track", "query": "Forward to the subsequent tune.", "expected": "Next Track", "cluster": "Inter-Cluster Media"},
    {"id": "inter_prev_track", "query": "Revisit the preceding title.", "expected": "Previous Track", "cluster": "Inter-Cluster Media"},
    {"id": "inter_play_query", "query": "Cue up bohemian rhapsody.", "expected": "Play Query", "cluster": "Inter-Cluster Media"},
    {"id": "inter_open_app", "query": "Fire up the calculator.", "expected": "Open App", "cluster": "Inter-Cluster App"},
    {"id": "inter_close_app", "query": "End the terminal process.", "expected": "Close App", "cluster": "Inter-Cluster App"},
    {"id": "inter_set_volume", "query": "Tune the audio output to 50%.", "expected": "Set Volume", "cluster": "Inter-Cluster Settings"},
    {"id": "inter_search_web", "query": "Search online to find out about weather forecast.", "expected": "Search Web", "cluster": "Inter-Cluster Web"},
]

# ------------------------------------------------------------------------------
# 2. INTRA-CLUSTER APP SUITE (16 High-Ambiguity App Cases from Phase 10)
# ------------------------------------------------------------------------------
INTRA_CLUSTER_APP_CASES = [
    {"id": "app_find_1", "query": "Where is blender installed on this computer?", "expected": "Find App", "cluster": "Intra-Cluster App"},
    {"id": "app_find_2", "query": "Locate the discord application on my machine.", "expected": "Find App", "cluster": "Intra-Cluster App"},
    {"id": "app_restart_1", "query": "Relaunch slack because it stopped responding.", "expected": "Restart App", "cluster": "Intra-Cluster App"},
    {"id": "app_restart_2", "query": "Reboot the visual studio code process.", "expected": "Restart App", "cluster": "Intra-Cluster App"},
    {"id": "app_focus_1", "query": "Switch to the active chrome window.", "expected": "Focus App", "cluster": "Intra-Cluster App"},
    {"id": "app_focus_2", "query": "Bring calculator to the foreground.", "expected": "Focus App", "cluster": "Intra-Cluster App"},
    {"id": "app_open_vs_find", "query": "Get chrome running on my PC.", "expected": "Open App", "cluster": "Intra-Cluster App"},
    {"id": "app_restart_vs_close", "query": "Restart my browser.", "expected": "Restart App", "cluster": "Intra-Cluster App"},
    {"id": "app_find_vs_web", "query": "Find where spotify is installed.", "expected": "Find App", "cluster": "Intra-Cluster App"},
    {"id": "app_focus_open", "query": "Bring up notepad right now.", "expected": "Open App", "cluster": "Intra-Cluster App"},
    {"id": "app_search_pc_vs_web", "query": "Search for chrome on my computer.", "expected": "Find App", "cluster": "Intra-Cluster App"},
    {"id": "app_kill_relaunch", "query": "Kill and restart slack.", "expected": "Restart App", "cluster": "Intra-Cluster App"},
    {"id": "app_check_installed", "query": "Discover which software is installed on this machine.", "expected": "Find App", "cluster": "Intra-Cluster App"},
    {"id": "app_switch_window", "query": "Switch over to the terminal window.", "expected": "Focus App", "cluster": "Intra-Cluster App"},
    {"id": "app_cycle_app", "query": "Cycle and bounce the terminal app.", "expected": "Restart App", "cluster": "Intra-Cluster App"},
    {"id": "app_fire_up", "query": "Fire up the calculator application.", "expected": "Open App", "cluster": "Intra-Cluster App"},
]

# ------------------------------------------------------------------------------
# 3. CROSS-DOMAIN MESSAGING & OVERLAP SUITE (16 Boundary Cases)
# ------------------------------------------------------------------------------
CROSS_DOMAIN_MESSAGING_CASES = [
    {"id": "msg_direct_send", "query": "Send message to David saying are we meeting today?", "expected": "Send Message", "cluster": "Messaging-Cross"},
    {"id": "msg_tell_contact", "query": "Tell Sarah I will be 15 minutes late for lunch.", "expected": "Send Message", "cluster": "Messaging-Cross"},
    {"id": "msg_text_mom", "query": "Text mom asking if she needs milk from the grocery store.", "expected": "Send Message", "cluster": "Messaging-Cross"},
    {"id": "msg_ping_contact", "query": "Ping Bob on whatsapp to check the git commit.", "expected": "Send Message", "cluster": "Messaging-Cross"},
    {"id": "msg_dispatch_telegram", "query": "Dispatch a message to Alice on telegram.", "expected": "Send Message", "cluster": "Messaging-Cross"},
    {"id": "msg_stage_draft", "query": "Stage a draft message to David saying hello.", "expected": "Stage Message", "cluster": "Messaging-Cross"},
    {"id": "msg_prepare_draft", "query": "Prepare a message for Sarah regarding the presentation.", "expected": "Stage Message", "cluster": "Messaging-Cross"},
    {"id": "overlap_web_search_query", "query": "Send a web search query for python tutorials.", "expected": "Search Web", "cluster": "Messaging-Cross"},
    {"id": "overlap_web_weather", "query": "Search google online for the current weather forecast.", "expected": "Search Web", "cluster": "Messaging-Cross"},
    {"id": "overlap_web_search_contact", "query": "Search online for the phone number of the restaurant.", "expected": "Search Web", "cluster": "Messaging-Cross"},
    {"id": "overlap_find_app_whatsapp", "query": "Find where whatsapp is installed on this PC.", "expected": "Find App", "cluster": "Messaging-Cross"},
    {"id": "overlap_open_app_whatsapp", "query": "Open WhatsApp application on my desktop.", "expected": "Open App", "cluster": "Messaging-Cross"},
    {"id": "overlap_open_app_discord", "query": "Launch the discord chat application.", "expected": "Open App", "cluster": "Messaging-Cross"},
    {"id": "overlap_tell_time", "query": "Tell me what time it is right now.", "expected": "Get Time", "cluster": "Messaging-Cross"},
    {"id": "overlap_inspect_screen", "query": "Inspect screen to view the buttons and active UI controls.", "expected": "Inspect Screen", "cluster": "Messaging-Cross"},
    {"id": "overlap_open_messaging_app", "query": "Fire up the telegram software.", "expected": "Open App", "cluster": "Messaging-Cross"},
]


def evaluate_suite(cases: List[Dict[str, Any]], suite_name: str) -> Dict[str, Any]:
    total = len(cases)
    bm25_top1_passes = 0
    bm25_top3_passes = 0
    struct_correct = 0
    clarified = 0
    silent_misroutes = 0

    evaluated_cases = []

    print(f"\n{'=' * 85}")
    print(f"SUITE: {suite_name} (N = {total} cases)")
    print(f"{'=' * 85}")

    for c in cases:
        qid = c["id"]
        q = c["query"]
        exp = c["expected"]
        cluster = c["cluster"]

        # 1. Pure Lexical BM25 Ranking
        raw_ranked = _bm25_rank(q)
        top1_tool, top1_score = raw_ranked[0] if raw_ranked else ("None", 0.0)
        top3_tools = [t for t, s in raw_ranked[:3] if s > 0]

        top1_ok = (top1_tool == exp)
        top3_ok = (exp in top3_tools)

        if top1_ok:
            bm25_top1_passes += 1
        if top3_ok:
            bm25_top3_passes += 1

        # 2. Structural Ranking & Confidence Gating
        struct_ranked, arg_info = rank_with_structure(q)
        chosen_action, is_confident, clar_prompt = resolve_intent_with_confidence(q, fallback_action="None")

        # Did it get the right action?
        action_match = (chosen_action == exp)
        is_clarification = (not is_confident and clar_prompt is not None)

        if action_match and is_confident:
            struct_correct += 1
            verdict = "CONFIDENT_CORRECT"
        elif is_clarification:
            clarified += 1
            verdict = "SAFELY_CLARIFIED"
        else:
            silent_misroutes += 1
            verdict = "SILENT_MISROUTE"

        print(f"[{qid}] '{q}'")
        print(f"  Expected   : '{exp}' [{cluster}]")
        print(f"  Pure BM25  : Top-1='{top1_tool}' ({top1_score:.2f}) [Match: {top1_ok}] | Top-3={[(t, round(s, 2)) for t, s in raw_ranked[:3]]} [In Top-3: {top3_ok}]")
        print(f"  Gated Action: '{chosen_action}' (Confident: {is_confident}) -> Verdict: {verdict}")
        if clar_prompt:
            print(f"  Clarification: {clar_prompt}")
        print("-" * 80)

        evaluated_cases.append({
            "id": qid,
            "query": q,
            "expected": exp,
            "cluster": cluster,
            "pure_bm25_top1": top1_tool,
            "pure_bm25_top1_score": float(top1_score),
            "pure_bm25_top1_match": top1_ok,
            "pure_bm25_top3": [(t, float(s)) for t, s in raw_ranked[:3]],
            "pure_bm25_top3_match": top3_ok,
            "chosen_action": chosen_action,
            "is_confident": is_confident,
            "clarification_prompt": clar_prompt,
            "verdict": verdict,
        })

    summary = {
        "suite_name": suite_name,
        "total": total,
        "bm25_top1_passes": bm25_top1_passes,
        "bm25_top1_pct": round(bm25_top1_passes / total * 100, 1),
        "bm25_top3_passes": bm25_top3_passes,
        "bm25_top3_pct": round(bm25_top3_passes / total * 100, 1),
        "struct_correct": struct_correct,
        "struct_correct_pct": round(struct_correct / total * 100, 1),
        "clarified": clarified,
        "clarified_pct": round(clarified / total * 100, 1),
        "silent_misroutes": silent_misroutes,
        "silent_misroutes_pct": round(silent_misroutes / total * 100, 1),
        "cases": evaluated_cases,
    }

    print(f"\nSUMMARY FOR {suite_name}:")
    print(f"  Pure BM25 Top-1 Accuracy: {bm25_top1_passes}/{total} ({summary['bm25_top1_pct']}%)")
    print(f"  Pure BM25 Top-3 Recall  : {bm25_top3_passes}/{total} ({summary['bm25_top3_pct']}%)")
    print(f"  Confident Correct Action: {struct_correct}/{total} ({summary['struct_correct_pct']}%)")
    print(f"  Safely Clarified Cases  : {clarified}/{total} ({summary['clarified_pct']}%)")
    print(f"  Dangerous Silent Misroutes: {silent_misroutes}/{total} ({summary['silent_misroutes_pct']}%)")
    print("=" * 85)

    return summary


def run_full_bm25_audit():
    print("=" * 85)
    print("CONSOLIDATED BM25 RE-AUDIT & SCALING ASSESSMENT (20 TOOLS)")
    print("Honoring Phase 10 Commitment: Real OOD Scaling Across Messaging + Existing Tools")
    print("=" * 85)
    print(f"Total Registered Tools : {len(TOOL_REGISTRY)}")
    print(f"Total Tool Descriptors : {len(TOOL_DESCRIPTORS)}")
    print(f"Confidence Thresholds  : CONFIDENCE_MARGIN={CONFIDENCE_MARGIN:.2f} | CONFIDENCE_RATIO={CONFIDENCE_RATIO:.2f}")

    s1 = evaluate_suite(INTER_CLUSTER_CASES, "Inter-Cluster Baseline Suite (Phase 8/9)")
    s2 = evaluate_suite(INTRA_CLUSTER_APP_CASES, "Intra-Cluster App Suite (Phase 10)")
    s3 = evaluate_suite(CROSS_DOMAIN_MESSAGING_CASES, "Cross-Domain Messaging & Overlap Suite (Phase 12/13)")

    total_cases = s1["total"] + s2["total"] + s3["total"]
    total_bm25_top1 = s1["bm25_top1_passes"] + s2["bm25_top1_passes"] + s3["bm25_top1_passes"]
    total_bm25_top3 = s1["bm25_top3_passes"] + s2["bm25_top3_passes"] + s3["bm25_top3_passes"]
    total_struct_correct = s1["struct_correct"] + s2["struct_correct"] + s3["struct_correct"]
    total_clarified = s1["clarified"] + s2["clarified"] + s3["clarified"]
    total_silent_misroutes = s1["silent_misroutes"] + s2["silent_misroutes"] + s3["silent_misroutes"]

    print("\n" + "=" * 85)
    print("FINAL CONSOLIDATED BM25 RE-AUDIT COMPARISON (PHASE 10 vs CURRENT 20 TOOLS)")
    print("=" * 85)
    print(f"{'Metric / Suite':<40} | {'Phase 10 (17 Tools)':<20} | {'Current v0.3 (20 Tools)':<20}")
    print("-" * 85)
    print(f"{'Inter-Cluster Top-1 Accuracy':<40} | {'100.0% (14/14)':<20} | {s1['bm25_top1_pct']:>5.1f}% ({s1['bm25_top1_passes']}/{s1['total']})")
    print(f"{'Inter-Cluster Top-3 Recall':<40} | {'100.0% (14/14)':<20} | {s1['bm25_top3_pct']:>5.1f}% ({s1['bm25_top3_passes']}/{s1['total']})")
    print(f"{'Intra-Cluster App Top-1 Accuracy':<40} | {' 87.5% (14/16)':<20} | {s2['bm25_top1_pct']:>5.1f}% ({s2['bm25_top1_passes']}/{s2['total']})")
    print(f"{'Intra-Cluster App Top-3 Recall':<40} | {' 93.8% (15/16)':<20} | {s2['bm25_top3_pct']:>5.1f}% ({s2['bm25_top3_passes']}/{s2['total']})")
    print(f"{'Cross-Domain Messaging Top-1 Acc':<40} | {' N/A (Untested)':<20} | {s3['bm25_top1_pct']:>5.1f}% ({s3['bm25_top1_passes']}/{s3['total']})")
    print(f"{'Cross-Domain Messaging Top-3 Rec':<40} | {' N/A (Untested)':<20} | {s3['bm25_top3_pct']:>5.1f}% ({s3['bm25_top3_passes']}/{s3['total']})")
    print("-" * 85)
    print(f"{'Overall Pure BM25 Top-1 Accuracy':<40} | {' 93.3% (28/30)':<20} | {total_bm25_top1/total_cases*100:>5.1f}% ({total_bm25_top1}/{total_cases})")
    print(f"{'Overall Pure BM25 Top-3 Recall':<40} | {' 96.7% (29/30)':<20} | {total_bm25_top3/total_cases*100:>5.1f}% ({total_bm25_top3}/{total_cases})")
    print(f"{'Overall Structured Confident Pass':<40} | {' 93.3% (28/30)':<20} | {total_struct_correct/total_cases*100:>5.1f}% ({total_struct_correct}/{total_cases})")
    print(f"{'Overall Safely Clarified (Gated)':<40} | {'  6.7% (2/30)':<20} | {total_clarified/total_cases*100:>5.1f}% ({total_clarified}/{total_cases})")
    print(f"{'Overall Silent Dangerous Misroutes':<40} | {'  0.0% (0/30)':<20} | {total_silent_misroutes/total_cases*100:>5.1f}% ({total_silent_misroutes}/{total_cases})")
    print("=" * 85)

    # Save artifact
    out_dir = Path("logs/phase13_consolidation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bm25_re_audit_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "inter_cluster": s1,
            "intra_cluster_app": s2,
            "cross_domain_messaging": s3,
            "totals": {
                "cases": total_cases,
                "pure_bm25_top1_accuracy": round(total_bm25_top1 / total_cases * 100, 1),
                "pure_bm25_top3_recall": round(total_bm25_top3 / total_cases * 100, 1),
                "structured_confident_accuracy": round(total_struct_correct / total_cases * 100, 1),
                "safely_clarified_rate": round(total_clarified / total_cases * 100, 1),
                "silent_misroute_rate": round(total_silent_misroutes / total_cases * 100, 1),
            }
        }, f, indent=2)
    print(f"\nSaved consolidated audit report to: {out_file}")


if __name__ == "__main__":
    run_full_bm25_audit()
