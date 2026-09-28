"""
MIKU blind_v3 Locked Evaluator
================================
Rules enforced by this script:
  1. SHA-256 of blind_v3.jsonl must match the hash committed in blind_v3.sha256
     before evaluation begins. Refuses to run if changed.
  2. A second run on the same hash is automatically flagged "NOT BLIND" in the
     run log and refused.
  3. The run log (eval/run_log.jsonl) is append-only. Each entry contains time,
     git commit, file hash, and metric results. Existing entries are never
     overwritten.
  4. Every evaluation item runs against a FRESH, isolated daemon state.
     No learned data or dialogue state carries across items.
  5. This script does NOT tune, does NOT write back to any training file,
     and does NOT open blind_v3.jsonl for any purpose other than reading
     its lines sequentially.
"""
import sys
import json
import hashlib
import time
import subprocess
import tracemalloc
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── paths ────────────────────────────────────────────────────────────────────
EVAL_DIR   = Path(__file__).resolve().parent
DATA_DIR   = EVAL_DIR / "data"
BLIND_FILE = DATA_DIR / "blind_v3.jsonl"
HASH_FILE  = DATA_DIR / "blind_v3.sha256"
RUN_LOG    = EVAL_DIR / "run_log.jsonl"

ROOT_DIR = EVAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ── helpers ───────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def append_run_log(entry: dict):
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def already_run(file_hash: str) -> bool:
    if not RUN_LOG.exists():
        return False
    with open(RUN_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("file_hash") == file_hash and entry.get("verdict") not in ("ABORTED_HASH_MISMATCH", "NOT_BLIND_DUPLICATE_RUN"):
                    return True
            except Exception:
                pass
    return False


# ── isolation harness ─────────────────────────────────────────────────────────
def get_isolated_daemon():
    import tempfile
    import shutil
    from multiprocessing import Queue
    from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
    tmp = tempfile.mkdtemp()
    q = Queue()
    daemon = CognitiveDaemon(action_queue=q)
    daemon.learner.filepath = Path(tmp) / "miku_learned_memory.json"
    daemon.learner.data = {"custom_aliases": {}, "category_preferences": {}}
    daemon.lexicon.words_filepath = Path(tmp) / "miku_learned_words.json"
    daemon.lexicon.learned_words = {}
    return daemon, tmp


def eval_item(item: dict):
    import shutil
    from miku.ipc.messages import STTTranscriptMsg
    daemon, tmp = get_isolated_daemon()
    try:
        for prior in item.get("previous_messages", []):
            daemon.handle_transcript(STTTranscriptMsg(text=prior, confidence=1.0))

        msg = STTTranscriptMsg(text=item["text"], confidence=1.0)
        t0 = time.perf_counter()
        res = daemon.handle_transcript(msg)
        latency = (time.perf_counter() - t0) * 1000.0

        status        = res.get("status")
        actual_action = res.get("best_action")
        if actual_action is None and status == "dispatched":
            actual_action = getattr(res.get("action_msg"), "action_type", None)

        return {
            "status": status,
            "actual_action": actual_action,
            "latency": latency,
            "message": str(res.get("message", "")),
            "action_msg": res.get("action_msg"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── scorer ────────────────────────────────────────────────────────────────────
def score_item(item: dict, res: dict) -> dict:
    expected_intent = item.get("expected_intent") or item.get("intent", "")
    expected_action = item.get("expected_action") or item.get("action", "")
    should_refuse   = item.get("should_refuse_or_ask", False)
    is_destructive  = item.get("is_destructive", False)
    category        = item.get("category", "general")
    expected_slots  = item.get("expected_slots") or item.get("slots") or {}

    status        = res["status"]
    actual_action = res["actual_action"]
    msg_lower     = res["message"].lower()

    is_false_action = False
    if should_refuse:
        if status == "dispatched" and actual_action not in (None, "noop"):
            if not (category == "negation" and expected_action == "open_app"):
                is_false_action = True

    intent_match = False
    if expected_intent in ("OUT_OF_SCOPE", "UNKNOWN"):
        intent_match = (status in ("conversational_response", "clarification_needed") and not is_false_action)
    elif expected_intent == "NEGATION_REFUSAL":
        intent_match = (status == "negation_refusal"
                        or (status in ("conversational_response", "clarification_needed") and not is_false_action))
    elif expected_intent == "CONFIRMATION_REQUIRED":
        intent_match = (status == "confirmation_required")
    elif expected_intent == "CLARIFICATION_NEEDED":
        intent_match = (status == "clarification_needed")
    elif expected_intent == "COMPOUND_COMMAND":
        intent_match = (status == "dispatched"
                        and (actual_action == "compound"
                             or "compound" in res["message"].lower()))
    elif expected_intent == "PRONOUN_COMMAND":
        intent_match = (status == "dispatched"
                        or (is_destructive and status == "confirmation_required"))
    elif expected_intent == "TEACH_WORD_ALIAS":
        intent_match = (status == "conversational_response"
                        and ("learned" in msg_lower or "got it" in msg_lower))
    elif expected_intent == "DEFINE_WORD_QUERY":
        intent_match = (status == "conversational_response"
                        and ("mean" in msg_lower or ":" in msg_lower))
    else:
        intent_match = (status == "dispatched" and actual_action == expected_action)

    slot_tp = slot_fn = 0
    if expected_slots:
        extracted = {}
        if res.get("action_msg") and hasattr(res["action_msg"], "params"):
            extracted = res["action_msg"].params or {}
        for k, ev in expected_slots.items():
            if k in ("steps", "selection", "pronoun", "action"):
                continue
            av = extracted.get(k) or ""
            if str(ev).lower() in str(av).lower() or str(av).lower() in str(ev).lower():
                slot_tp += 1
            else:
                slot_fn += 1

    return {
        "intent_match":    intent_match,
        "is_false_action": is_false_action,
        "is_clarification": (status == "clarification_needed"),
        "slot_tp": slot_tp,
        "slot_fn": slot_fn,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  MIKU  blind_v3  Locked Evaluation")
    print("=" * 72)

    if not BLIND_FILE.exists():
        print(f"[ERROR] blind_v3.jsonl not found at {BLIND_FILE}")
        print("        Supply a file authored by someone other than the coding agent.")
        sys.exit(1)
    if not HASH_FILE.exists():
        print(f"[ERROR] blind_v3.sha256 not found at {HASH_FILE}")
        print("        Commit the hash file before running evaluation.")
        sys.exit(1)

    committed_hash = HASH_FILE.read_text(encoding="utf-8").strip().split()[0]
    actual_hash    = sha256_file(BLIND_FILE)

    print(f"[*] Committed hash : {committed_hash}")
    print(f"[*] Actual hash    : {actual_hash}")

    if actual_hash != committed_hash:
        print("[ABORT] Hash mismatch — blind_v3.jsonl has been modified since it was committed.")
        append_run_log({
            "time": datetime.now(timezone.utc).isoformat(),
            "commit": git_commit(),
            "file_hash": actual_hash,
            "committed_hash": committed_hash,
            "verdict": "ABORTED_HASH_MISMATCH",
        })
        sys.exit(2)

    if already_run(actual_hash):
        print("[ABORT] This hash has already been evaluated once.")
        print("        A second run on the same blind set is NOT BLIND — flagged in run log.")
        append_run_log({
            "time": datetime.now(timezone.utc).isoformat(),
            "commit": git_commit(),
            "file_hash": actual_hash,
            "verdict": "NOT_BLIND_DUPLICATE_RUN",
        })
        sys.exit(3)

    print("[OK] Hash verified. Starting isolated evaluation...\n")

    records = [json.loads(l) for l in BLIND_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    total   = len(records)
    print(f"[*] Total items: {total}")

    tracemalloc.start()
    correct = false_actions = clarifications = 0
    slot_tp = slot_fn = 0
    latencies = []
    failures  = []

    for i, item in enumerate(records, 1):
        if i % 50 == 0:
            print(f"    ... {i}/{total}")
        try:
            res = eval_item(item)
        except Exception as e:
            print(f"  [!] Item {i} ({item.get('id','?')}) error: {e}")
            traceback.print_exc()
            continue

        sc = score_item(item, res)
        latencies.append(res["latency"])

        if sc["intent_match"]:
            correct += 1
        else:
            failures.append({
                "id":           item.get("id", f"item-{i}"),
                "text":         item["text"],
                "category":     item.get("category", ""),
                "expected":     item.get("expected_intent") or item.get("intent", ""),
                "actual_status":  res["status"],
                "actual_action":  res["actual_action"],
                "message":        res["message"][:120],
                "is_false_action": sc["is_false_action"],
            })

        if sc["is_false_action"]:
            false_actions += 1
        if sc["is_clarification"]:
            clarifications += 1
        slot_tp += sc["slot_tp"]
        slot_fn += sc["slot_fn"]

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    intent_acc  = correct / total * 100
    far_total   = false_actions / total * 100
    refuse_n    = sum(1 for r in records if r.get("should_refuse_or_ask"))
    far_refusal = (false_actions / refuse_n * 100) if refuse_n else 0.0
    clar_rate   = clarifications / total * 100

    prec = slot_tp / (slot_tp) if slot_tp else 1.0
    rec  = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) else 1.0
    f1   = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) else 100.0

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    peak_mb = peak_mem / 1024 / 1024

    from collections import defaultdict
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)
    fail_ids    = {f["id"] for f in failures}
    for item in records:
        cat = item.get("category", "general")
        cat_total[cat] += 1
        if item.get("id", "") not in fail_ids:
            cat_correct[cat] += 1

    print("\n" + "=" * 72)
    print("  RESULTS  (as measured, not tuned)")
    print("=" * 72)
    print(f"  Intent Accuracy        : {intent_acc:.2f}%  ({correct}/{total})")
    print(f"  Slot F1                : {f1:.2f}%")
    print(f"  False-Action Rate      : {far_total:.2f}%  (all {total} items)")
    print(f"  False-Action Rate      : {far_refusal:.2f}%  (refusal/clarification items, n={refuse_n})")
    print(f"  Clarification Rate     : {clar_rate:.2f}%")
    print(f"  Latency p50/p95        : {p50:.2f} ms / {p95:.2f} ms")
    print(f"  Peak RAM               : {peak_mb:.1f} MB")

    print("\n  Per-category breakdown:")
    for cat in sorted(cat_total):
        acc = cat_correct[cat] / cat_total[cat] * 100
        print(f"    {cat:<32} {acc:5.1f}%  ({cat_correct[cat]}/{cat_total[cat]})")

    # Confused pairs (expected → actual_action)
    from collections import Counter
    confused = Counter(
        f"{f['expected']} → {f['actual_status']}:{f['actual_action']}"
        for f in failures
    )
    print("\n  Top 10 confused pairs:")
    for pair, count in confused.most_common(10):
        print(f"    {count:3d}x  {pair}")

    print("\n  10 worst failures (verbatim):")
    for i, f in enumerate(failures[:10], 1):
        print(f"  [{i:02d}] id={f['id']}  cat={f['category']}")
        print(f"        input      : \"{f['text']}\"")
        print(f"        expected   : {f['expected']}")
        print(f"        actual     : {f['actual_status']} / {f['actual_action']}")
        print(f"        message    : {f['message']}")
        print(f"        false_action: {f['is_false_action']}")
        print()

    run_entry = {
        "time":           datetime.now(timezone.utc).isoformat(),
        "commit":         git_commit(),
        "file_hash":      actual_hash,
        "verdict":        "BLIND_RUN",
        "total":          total,
        "intent_acc":     round(intent_acc,  2),
        "slot_f1":        round(f1,          2),
        "far_total":      round(far_total,   2),
        "far_refusal":    round(far_refusal, 2),
        "clar_rate":      round(clar_rate,   2),
        "p50_ms":         round(p50, 2),
        "p95_ms":         round(p95, 2),
        "peak_ram_mb":    round(peak_mb, 1),
        "false_actions":  false_actions,
        "failures_top10": failures[:10],
    }
    append_run_log(run_entry)
    print(f"[+] Run appended to {RUN_LOG}")
    print("    This was a BLIND RUN. Do not tune against these results.")


if __name__ == "__main__":
    main()
