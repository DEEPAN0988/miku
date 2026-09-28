"""
evaluate_blind_v3.py — Locked one-shot evaluator for blind_v3.
Rules:
  (a) Reads blind_v3.jsonl only when run by the owner (MIKU_OWNER env var must be set).
  (b) Verifies SHA-256 of the file before running.
  (c) Writes an append-only run log to eval/blind_v3_run_log.jsonl.
  (d) Flags any second run on the same hash as "not blind".
"""
import sys
import os
import json
import hashlib
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.isolated_harness import evaluate_item_isolated

BLIND_V3_PATH = ROOT / "eval" / "data" / "blind_v3.jsonl"
RUN_LOG = ROOT / "eval" / "blind_v3_run_log.jsonl"
KNOWN_HASH_FILE = ROOT / "eval" / "data" / "blind_v3.sha256"

OWNER_ENV = "MIKU_OWNER"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_owner():
    if not os.environ.get(OWNER_ENV):
        print(f"ERROR: {OWNER_ENV} environment variable not set.")
        print("Only the dataset owner may run blind_v3. Set the variable and retry.")
        sys.exit(2)


def load_known_hash() -> str:
    if not KNOWN_HASH_FILE.exists():
        print(f"ERROR: {KNOWN_HASH_FILE} not found. Cannot verify dataset integrity.")
        sys.exit(3)
    return KNOWN_HASH_FILE.read_text().strip().lower()


def check_run_log(file_hash: str) -> bool:
    """Returns True if this hash has been run before (not blind)."""
    if not RUN_LOG.exists():
        return False
    with open(RUN_LOG, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("file_hash", "").lower() == file_hash.lower():
                    return True
            except json.JSONDecodeError:
                pass
    return False


def append_run_log(file_hash: str, results: dict):
    entry = {
        "timestamp": time.time(),
        "file_hash": file_hash,
        "intent_accuracy": results["intent_accuracy"],
        "false_action_rate_all": results["false_action_rate_all"],
        "false_action_rate_refusal_denom": results["false_action_rate_refusal_denom"],
        "clarification_rate": results["clarification_rate"],
        "per_category": results["per_category"],
    }
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def score_item(item: dict, res: dict) -> tuple[bool, bool]:
    """Returns (intent_match, is_false_action)."""
    expected_intent = item.get("expected_intent") or item.get("intent", "")
    expected_action = item.get("expected_action") or item.get("action", "")
    should_refuse_or_ask = item.get("should_refuse_or_ask", False)
    is_destructive = item.get("is_destructive", False)
    category = item.get("category", "general")

    status = res["status"]
    actual_action = res["actual_action"]
    msg = res["message"].lower()

    # False action check
    is_false_action = False
    if should_refuse_or_ask:
        if status == "dispatched" and actual_action not in (None, "noop"):
            if not (category == "negation" and expected_action == "open_app"):
                is_false_action = True

    # Intent match
    intent_match = False
    if expected_intent in ("OUT_OF_SCOPE", "UNKNOWN", "REFUSE"):
        intent_match = (status in ("conversational_response", "clarification_needed") and not is_false_action)
    elif expected_intent == "NEGATION_REFUSAL":
        intent_match = (status == "negation_refusal" or (
            status in ("conversational_response", "clarification_needed") and not is_false_action))
    elif expected_intent == "CONFIRMATION_REQUIRED":
        intent_match = (status == "confirmation_required")
    elif expected_intent == "CLARIFICATION_NEEDED":
        intent_match = (status == "clarification_needed")
    elif expected_intent == "COMPOUND_COMMAND":
        intent_match = (status == "dispatched" and (
            actual_action == "compound" or "compound" in str(res.get("message", "")).lower()))
    elif expected_intent == "PRONOUN_COMMAND":
        intent_match = (
            status == "dispatched"
            or (is_destructive and status == "confirmation_required")
            or status == "clarification_needed"
        )
    elif expected_intent == "TEACH_WORD_ALIAS":
        msg_lower = str(res.get("message", "")).lower()
        intent_match = (
            status == "confirmation_required"
            or (status == "conversational_response" and (
                "learned" in msg_lower or "got it" in msg_lower or "linked" in msg_lower))
        )
    elif expected_intent == "DEFINE_WORD_QUERY":
        intent_match = (status == "conversational_response" and (
            "mean" in msg or ":" in msg))
    else:
        intent_match = (status == "dispatched" and actual_action == expected_action)

    return intent_match, is_false_action


def run():
    # (a) Owner check
    check_owner()

    # (b) Hash check
    if not BLIND_V3_PATH.exists():
        print(f"ERROR: {BLIND_V3_PATH} not found.")
        sys.exit(4)

    known_hash = load_known_hash()
    actual_hash = sha256_of_file(BLIND_V3_PATH)

    if actual_hash.lower() != known_hash.lower():
        print(f"ERROR: SHA-256 mismatch.")
        print(f"  expected: {known_hash}")
        print(f"  actual  : {actual_hash}")
        sys.exit(5)
    print(f"[OK] SHA-256 verified: {actual_hash}")

    # (d) Second-run detection
    if check_run_log(actual_hash):
        print(f"\nWARNING: This hash ({actual_hash[:16]}...) has already been run.")
        print("This dataset is NO LONGER BLIND. Results will not be treated as a blind evaluation.")
        proceed = input("Continue anyway? (yes/no): ").strip().lower()
        if proceed != "yes":
            sys.exit(6)

    # Load dataset
    with open(BLIND_V3_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    correct = 0
    false_actions = 0
    clarifications = 0
    refuse_denom = 0  # items that should be CONFIRM, CLARIFY, or REFUSE

    per_cat_correct = {}
    per_cat_total = {}

    print(f"\nEvaluating {total} items from blind_v3...")
    for item in records:
        cat = item.get("category", "general")
        per_cat_total[cat] = per_cat_total.get(cat, 0) + 1

        should_refuse_or_ask = item.get("should_refuse_or_ask", False)
        if should_refuse_or_ask:
            refuse_denom += 1

        try:
            res = evaluate_item_isolated(item, item.get("previous_messages", []))
        except Exception:
            traceback.print_exc()
            continue

        if res["status"] == "clarification_needed":
            clarifications += 1

        intent_match, is_false_action = score_item(item, res)

        if is_false_action:
            false_actions += 1
        if intent_match:
            correct += 1
            per_cat_correct[cat] = per_cat_correct.get(cat, 0) + 1

    intent_acc = correct / total * 100
    far_all = false_actions / total * 100
    far_refusal = (false_actions / refuse_denom * 100) if refuse_denom else 0.0
    clarification_rate = clarifications / total * 100

    per_category = {}
    for cat, tot in sorted(per_cat_total.items()):
        corr = per_cat_correct.get(cat, 0)
        per_category[cat] = {"correct": corr, "total": tot, "accuracy": corr / tot * 100}

    results = {
        "intent_accuracy": intent_acc,
        "false_action_rate_all": far_all,
        "false_action_rate_refusal_denom": far_refusal,
        "clarification_rate": clarification_rate,
        "per_category": per_category,
    }

    print(f"\n{'='*60}")
    print(f"blind_v3 Results")
    print(f"{'='*60}")
    print(f"Total Items      : {total}")
    print(f"Intent Accuracy  : {intent_acc:.2f}%")
    print(f"FAR (all denom)  : {far_all:.2f}%  ({false_actions} false actions)")
    print(f"FAR (refusal den): {far_refusal:.2f}%  (n={refuse_denom})")
    print(f"Clarification %  : {clarification_rate:.2f}%")
    print(f"\nPer-Category Breakdown:")
    for cat, stats in sorted(per_category.items()):
        print(f"  {cat:<35} {stats['accuracy']:5.1f}%  errors={stats['total']-stats['correct']:3d}  total={stats['total']}")

    # (c) Append-only run log
    append_run_log(actual_hash, results)
    print(f"\n[OK] Run logged to {RUN_LOG}")


if __name__ == "__main__":
    run()
