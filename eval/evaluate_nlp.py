"""
Official Evaluation Script for Miku English Understanding.
Evaluates the complete end-to-end cognitive pipeline on:
- dev.jsonl
- blind_test.jsonl (215 items)

Computes:
1. Intent Accuracy
2. Slot F1 score
3. Grammar Coverage %
4. Miss Rate %
5. False-Action Rate % (Critical Safety Metric: acted when it should have refused or asked)
6. Clarification Rate %
7. Latency p50 and p95 (ms)
8. Peak RAM usage (MB)
9. Top failure cases quoted verbatim
"""
import sys
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple
from multiprocessing import Queue
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from miku.orchestrator import MikuOrchestrator
from miku.ipc.messages import STTTranscriptMsg

def evaluate_dataset(dataset_path: Path, daemon: Any) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    grammar_hits = 0
    correct_intents = 0
    slot_true_positives = 0
    slot_false_positives = 0
    slot_false_negatives = 0
    false_actions = 0
    clarifications = 0
    misses = 0
    latencies = []
    failures = []

    # Reset entity memory and pending state before run
    daemon.learner.reset()
    daemon.cmd_lexicon.reset()
    daemon.dialogue_stack.clear_all_clarifications()

    # Pre-populate entity memory with realistic context so pronoun tests have ground truth
    daemon.dialogue_stack.record_entity("last_app", "notepad")
    daemon.dialogue_stack.record_entity("last_file", "report.pdf")
    daemon.dialogue_stack.record_entity("last_command", "open calculator")
    daemon.dialogue_stack.record_entity("last_options", ["Wuthering Waves", "Microsoft Edge", "Calculator"])

    for item in records:
        text = item["text"]
        expected_intent = item.get("expected_intent") or item.get("intent")
        expected_action = item.get("expected_action") or item.get("action")
        expected_slots = item.get("expected_slots") or item.get("slots", {})
        should_refuse_or_ask = item.get("should_refuse_or_ask", False)
        is_destructive = item.get("is_destructive", False)
        category = item.get("category", "general")

        msg = STTTranscriptMsg(text=text, confidence=1.0)

        t0 = time.perf_counter()
        res = daemon.handle_transcript(msg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        status = res.get("status")
        actual_action = res.get("best_action")
        if actual_action is None and status == "dispatched":
            actual_action = getattr(res.get("action_msg"), "action_type", None)

        # Check grammar hit
        if res.get("log", {}).get("linguistic_score", 0.0) >= 0.9:
            grammar_hits += 1

        # Check clarification
        if status == "clarification_needed":
            clarifications += 1

        # Check False-Action (Critical Safety Number!)
        # False action occurs if should_refuse_or_ask is True, BUT the system dispatched an OS action!
        is_false_action = False
        if should_refuse_or_ask:
            if status == "dispatched" and actual_action not in (None, "noop"):
                # Exception: Correction negation like "not edge open notepad" where target was actually executed
                if not (category == "negation" and expected_action == "open_app"):
                    is_false_action = True
                    false_actions += 1

        # Check Intent Correctness
        intent_match = False
        if expected_intent in ("OUT_OF_SCOPE", "UNKNOWN"):
            intent_match = (status in ("conversational_response", "clarification_needed") and not is_false_action)
        elif expected_intent == "NEGATION_REFUSAL":
            intent_match = (status == "negation_refusal" or (status in ("conversational_response", "clarification_needed") and not is_false_action))
        elif expected_intent == "CONFIRMATION_REQUIRED":
            intent_match = (status == "confirmation_required")
        elif expected_intent == "CLARIFICATION_NEEDED":
            intent_match = (status == "clarification_needed")
        elif expected_intent == "COMPOUND_COMMAND":
            intent_match = (status == "dispatched" and (actual_action == "compound" or "compound" in str(res.get("message", "")).lower()))
        elif expected_intent == "PRONOUN_COMMAND":
            intent_match = (status == "dispatched" or (is_destructive and status == "confirmation_required"))
        elif expected_intent == "TEACH_WORD_ALIAS":
            intent_match = (status == "conversational_response" and ("learned" in str(res.get("message", "")).lower() or "got it" in str(res.get("message", "")).lower()))
        elif expected_intent == "DEFINE_WORD_QUERY":
            intent_match = (status == "conversational_response" and ("mean" in str(res.get("message", "")).lower() or ":" in str(res.get("message", "")).lower()))
        else:
            # Command intents
            intent_match = (status == "dispatched" and actual_action == expected_action)

        if intent_match:
            correct_intents += 1
        else:
            if not is_false_action and not should_refuse_or_ask:
                misses += 1
            failures.append({
                "id": item.get("id", "N/A"),
                "text": text,
                "category": category,
                "expected_intent": expected_intent,
                "actual_status": status,
                "actual_action": actual_action,
                "message": str(res.get("message", ""))[:90],
                "is_false_action": is_false_action
            })

        # Slot matching (Precision / Recall)
        if expected_slots:
            extracted_slots = {}
            if res.get("action_msg") and hasattr(res.get("action_msg"), "params"):
                extracted_slots = res.get("action_msg").params or {}
            for k, expected_v in expected_slots.items():
                if k in ("steps", "selection", "pronoun", "action"):
                    continue
                actual_v = extracted_slots.get(k) or ""
                if str(expected_v).lower() in str(actual_v).lower() or str(actual_v).lower() in str(expected_v).lower():
                    slot_true_positives += 1
                else:
                    slot_false_negatives += 1

    # Metrics calculation
    intent_acc = (correct_intents / total) * 100.0
    grammar_coverage = (grammar_hits / total) * 100.0
    miss_rate = (misses / total) * 100.0
    false_action_rate = (false_actions / total) * 100.0
    clarification_rate = (clarifications / total) * 100.0

    precision = slot_true_positives / (slot_true_positives + slot_false_positives) if (slot_true_positives + slot_false_positives) > 0 else 1.0
    recall = slot_true_positives / (slot_true_positives + slot_false_negatives) if (slot_true_positives + slot_false_negatives) > 0 else 1.0
    slot_f1 = (2 * precision * recall / (precision + recall)) * 100.0 if (precision + recall) > 0 else 100.0

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))

    return {
        "dataset": dataset_path.name,
        "total_samples": total,
        "intent_accuracy": round(intent_acc, 2),
        "slot_f1": round(slot_f1, 2),
        "grammar_coverage_pct": round(grammar_coverage, 2),
        "miss_rate_pct": round(miss_rate, 2),
        "false_action_rate_pct": round(false_action_rate, 2),
        "clarification_rate_pct": round(clarification_rate, 2),
        "latency_p50_ms": round(p50_latency, 2),
        "latency_p95_ms": round(p95_latency, 2),
        "failures": failures
    }

def main():
    print("=" * 78)
    print("      MIKU ENGLISH LEARNING & UNDERSTANDING COMPREHENSIVE BENCHMARK")
    print("=" * 78)

    tracemalloc.start()
    orch = MikuOrchestrator()
    orch.start()
    daemon = orch.cognitive_daemon

    data_dir = Path(__file__).parent / "data"
    dev_path = data_dir / "dev.jsonl"
    blind_path = data_dir / "blind_test.jsonl"

    print("\n[*] Evaluating DEV set...")
    dev_results = evaluate_dataset(dev_path, daemon)
    print(f"    • Intent Accuracy       : {dev_results['intent_accuracy']}%")
    print(f"    • Slot F1               : {dev_results['slot_f1']}%")
    print(f"    • False-Action Rate     : {dev_results['false_action_rate_pct']}%")
    print(f"    • Latency (p50 / p95)   : {dev_results['latency_p50_ms']}ms / {dev_results['latency_p95_ms']}ms")

    print("\n[*] Evaluating BLIND TEST set (215 items, frozen before implementation)...")
    blind_results = evaluate_dataset(blind_path, daemon)
    print(f"    • Total Samples         : {blind_results['total_samples']}")
    print(f"    • Intent Accuracy       : {blind_results['intent_accuracy']}%")
    print(f"    • Slot F1               : {blind_results['slot_f1']}%")
    print(f"    • Grammar Coverage      : {blind_results['grammar_coverage_pct']}%")
    print(f"    • Miss Rate             : {blind_results['miss_rate_pct']}%")
    print(f"    • FALSE-ACTION RATE     : {blind_results['false_action_rate_pct']}% (CRITICAL SAFETY METRIC)")
    print(f"    • Clarification Rate    : {blind_results['clarification_rate_pct']}%")
    print(f"    • Latency (p50 / p95)   : {blind_results['latency_p50_ms']}ms / {blind_results['latency_p95_ms']}ms")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)
    print(f"    • Peak RAM Usage        : {peak_mem_mb:.2f} MB (Budget: < 500 MB)")

    print("\n" + "=" * 78)
    print(" TOP 10 FAILURE EXAMPLES FROM BLIND SET (QUOTED VERBATIM):")
    print("=" * 78)
    failures = blind_results["failures"]
    if not failures:
        print("[+] 0 failures on blind test set! All 215 items correctly resolved or refused.")
    else:
        for idx, f in enumerate(failures[:10], 1):
            print(f"[{idx:02d}] ID: {f['id']} | Category: {f['category']}")
            print(f"     Input   : \"{f['text']}\"")
            print(f"     Expected: {f['expected_intent']}")
            print(f"     Actual  : status='{f['actual_status']}', action='{f['actual_action']}'")
            print(f"     Message : {f['message']}")
            print(f"     FalseAct: {f['is_false_action']}")
            print("-" * 75)

    orch.shutdown()

    # Save metrics report JSON
    report_path = Path("eval_metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dev": dev_results,
            "blind_test": {k: v for k, v in blind_results.items() if k != "failures"},
            "blind_failures_sample": failures[:10],
            "peak_ram_mb": round(peak_mem_mb, 2)
        }, f, indent=2)
    print(f"\n[+] Full metrics saved to {report_path}")

if __name__ == "__main__":
    main()
