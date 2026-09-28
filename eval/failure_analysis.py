"""Produce ranked failure table for dev_blind_v1 and dev_v2 under isolated harness."""
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.isolated_harness import evaluate_item_isolated

DATA_DIR = Path(__file__).resolve().parent / "data"

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def score(item, res):
    ei = item.get("expected_intent") or item.get("intent", "")
    ea = item.get("expected_action") or item.get("action", "")
    should_refuse = item.get("should_refuse_or_ask", False)
    is_dest = item.get("is_destructive", False)
    cat = item.get("category", "general")
    status = res["status"]
    aa = res["actual_action"]
    msg = res["message"].lower()

    is_fa = False
    if should_refuse and status == "dispatched" and aa not in (None, "noop"):
        if not (cat == "negation" and ea == "open_app"):
            is_fa = True

    if ei in ("OUT_OF_SCOPE","UNKNOWN"):
        ok = status in ("conversational_response","clarification_needed") and not is_fa
    elif ei == "NEGATION_REFUSAL":
        ok = status == "negation_refusal" or (status in ("conversational_response","clarification_needed") and not is_fa)
    elif ei == "CONFIRMATION_REQUIRED":
        ok = status == "confirmation_required"
    elif ei == "CLARIFICATION_NEEDED":
        ok = status == "clarification_needed"
    elif ei == "COMPOUND_COMMAND":
        ok = status == "dispatched" and (aa == "compound" or "compound" in res["message"].lower())
    elif ei == "PRONOUN_COMMAND":
        ok = (status == "dispatched" or (is_dest and status == "confirmation_required") or status == "clarification_needed")
    elif ei == "TEACH_WORD_ALIAS":
        msg_lower = res["message"].lower()
        ok = (
            status == "confirmation_required"
            or (status == "conversational_response" and ("learned" in msg_lower or "got it" in msg_lower or "linked" in msg_lower))
        )
    elif ei == "DEFINE_WORD_QUERY":
        ok = status == "conversational_response" and ("mean" in msg or ":" in msg)
    else:
        ok = status == "dispatched" and aa == ea

    return ok, is_fa

def analyse(path):
    records = load(path)
    cat_total   = defaultdict(int)
    cat_correct = defaultdict(int)
    failures = []
    false_actions = []
    refuse_n = 0

    for item in records:
        cat = item.get("category", "general")
        cat_total[cat] += 1
        if item.get("should_refuse_or_ask"):
            refuse_n += 1

        res = evaluate_item_isolated(item)
        ok, is_fa = score(item, res)

        if ok:
            cat_correct[cat] += 1
        else:
            failures.append({
                "id": item.get("id","?"),
                "text": item["text"],
                "category": cat,
                "expected": item.get("expected_intent") or item.get("intent",""),
                "actual_status": res["status"],
                "actual_action": res["actual_action"],
                "message": res["message"][:100],
                "is_fa": is_fa,
            })
        if is_fa:
            false_actions.append({"id": item.get("id","?"), "text": item["text"], "category": cat,
                                   "actual_action": res["actual_action"]})

    total = len(records)
    fa_total = len(false_actions) / total * 100
    fa_ref   = len(false_actions) / refuse_n * 100 if refuse_n else 0.0

    print(f"\n{'='*70}")
    print(f"Dataset: {Path(path).name}  ({total} items)")
    print(f"{'='*70}")
    acc = sum(cat_correct.values()) / total * 100
    print(f"Overall intent accuracy : {acc:.2f}%")
    print(f"FAR (all denom)         : {fa_total:.2f}%  ({len(false_actions)} items)")
    print(f"FAR (refusal denom)     : {fa_ref:.2f}%  (n={refuse_n})")

    print(f"\nPer-category (ranked by error count):")
    cat_errors = {c: cat_total[c] - cat_correct[c] for c in cat_total}
    for c, errs in sorted(cat_errors.items(), key=lambda x: -x[1]):
        acc_c = cat_correct[c] / cat_total[c] * 100
        print(f"  {c:<35} {acc_c:5.1f}%  errors={errs:3d}  total={cat_total[c]}")

    print(f"\nFailure table ({len(failures)} failures):")
    for f in failures:
        fa_tag = " [FALSE-ACTION]" if f["is_fa"] else ""
        print(f"  [{f['id']}] cat={f['category']}{fa_tag}")
        print(f"    text    : \"{f['text']}\"")
        print(f"    expected: {f['expected']}")
        print(f"    actual  : {f['actual_status']} / {f['actual_action']}")
        print(f"    message : {f['message']}")
        print()

    print(f"\nFalse actions ({len(false_actions)}):")
    for fa in false_actions:
        print(f"  [{fa['id']}] cat={fa['category']}  action={fa['actual_action']}  text=\"{fa['text']}\"")

    return failures, cat_errors, cat_total

print("Analysing dev_blind_v1...")
f1, ce1, ct1 = analyse(DATA_DIR / "dev_blind_v1.jsonl")
print("\n\nAnalysing dev_v2...")
f2, ce2, ct2 = analyse(DATA_DIR / "blind_v2.jsonl")

# Combined category ranking
print("\n\nCOMBINED category ranking:")
all_cats = set(ce1) | set(ce2)
rows = []
for c in all_cats:
    rows.append((c, ce1.get(c,0)+ce2.get(c,0), ct1.get(c,0)+ct2.get(c,0)))
for c, errs, tot in sorted(rows, key=lambda x: -x[1]):
    print(f"  {c:<35} errors={errs:3d}  total={tot}")
