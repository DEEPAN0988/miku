"""
Run the ORIGINAL (pre-Round-2) scorer on dev_blind_v1.
Reproduces evaluate_nlp.py @ 64c0b8b scoring rules exactly.
"""
import sys, json, time, tracemalloc
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from miku.orchestrator import MikuOrchestrator
from miku.ipc.messages import STTTranscriptMsg

try:
    import numpy as np
    def pct(a, n): return float(np.percentile(a, n))
except ImportError:
    def pct(a, n):
        a = sorted(a); idx = int(len(a)*n/100); return a[min(idx, len(a)-1)]

def score(records, daemon):
    correct = 0
    false_actions = 0
    latencies = []
    failures = []

    daemon.learner.reset()
    daemon.cmd_lexicon.reset()
    daemon.dialogue_stack.clear_all_clarifications()
    daemon.dialogue_stack.record_entity("last_app", "notepad")
    daemon.dialogue_stack.record_entity("last_file", "report.pdf")
    daemon.dialogue_stack.record_entity("last_command", "open calculator")
    daemon.dialogue_stack.record_entity("last_options", ["Wuthering Waves", "Microsoft Edge", "Calculator"])

    for item in records:
        text = item["text"]
        ei = item.get("expected_intent") or item.get("intent","")
        ea = item.get("expected_action") or item.get("action","")
        refuse = item.get("should_refuse_or_ask", False)
        destr  = item.get("is_destructive", False)
        cat    = item.get("category", "general")

        msg = STTTranscriptMsg(text=text, confidence=1.0)
        t0  = time.perf_counter()
        res = daemon.handle_transcript(msg)
        latencies.append((time.perf_counter()-t0)*1000.0)

        status = res.get("status")
        aa     = res.get("best_action")
        if aa is None and status == "dispatched":
            aa = getattr(res.get("action_msg"), "action_type", None)
        ml = str(res.get("message","")).lower()

        # ---- original false-action logic ----
        is_fa = False
        if refuse and status == "dispatched" and aa not in (None,"noop"):
            if not (cat == "negation" and ea == "open_app"):
                is_fa = True
                false_actions += 1

        # ---- original intent scoring (evaluate_nlp.py @ 64c0b8b) ----
        ok = False
        if   ei in ("OUT_OF_SCOPE","UNKNOWN"):
            ok = status in ("conversational_response","clarification_needed") and not is_fa
        elif ei == "NEGATION_REFUSAL":
            ok = status == "negation_refusal" or (status in ("conversational_response","clarification_needed") and not is_fa)
        elif ei == "CONFIRMATION_REQUIRED":
            ok = status == "confirmation_required"
        elif ei == "CLARIFICATION_NEEDED":
            ok = status == "clarification_needed"
        elif ei == "COMPOUND_COMMAND":
            ok = status == "dispatched" and (aa == "compound" or "compound" in str(res.get("message","")).lower())
        elif ei == "PRONOUN_COMMAND":
            ok = status == "dispatched" or (destr and status == "confirmation_required")
        elif ei == "TEACH_WORD_ALIAS":
            # ORIGINAL: only conversational_response with learned/got it
            ok = status == "conversational_response" and ("learned" in ml or "got it" in ml)
        elif ei == "DEFINE_WORD_QUERY":
            ok = status == "conversational_response" and ("mean" in ml or ":" in ml)
        else:
            # ORIGINAL: SYSTEM_STATUS, DELETE_FILE fall here → must dispatch with correct action
            ok = status == "dispatched" and aa == ea

        if ok:
            correct += 1
        else:
            failures.append({"id":item.get("id","?"),"text":text,"cat":cat,
                "ei":ei,"ea":ea,"status":status,"aa":aa,"ml":ml[:70],"is_fa":is_fa})

    total    = len(records)
    ref_n    = sum(1 for r in records if r.get("should_refuse_or_ask",False))
    return {
        "total":total,"correct":correct,
        "acc":round(correct/total*100,2),
        "false_actions":false_actions,
        "fa_total":round(false_actions/total*100,2),
        "fa_refusal":round(false_actions/ref_n*100,2) if ref_n else 0.0,
        "ref_n":ref_n,
        "p50":round(pct(latencies,50),2),
        "p95":round(pct(latencies,95),2),
        "failures":failures,
    }

def main():
    tracemalloc.start()
    orch = MikuOrchestrator(); orch.start()
    daemon = orch.cognitive_daemon

    path = ROOT_DIR / "eval" / "data" / "dev_blind_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Running ORIGINAL scorer on {path.name} ({len(records)} items) ...")
    r = score(records, daemon)
    orch.shutdown()

    print(f"\n{'='*60}")
    print("ORIGINAL SCORER — dev_blind_v1")
    print(f"{'='*60}")
    print(f"  Accuracy             : {r['acc']}%  ({r['correct']}/{r['total']})")
    print(f"  False actions        : {r['false_actions']}")
    print(f"  False-action (total) : {r['fa_total']}%")
    print(f"  False-action (ref)   : {r['fa_refusal']}%  ({r['false_actions']}/{r['ref_n']})")
    print(f"  Latency p50/p95      : {r['p50']} / {r['p95']} ms")
    print(f"\nFailed ({len(r['failures'])}):")
    for ei,n in Counter(f["ei"] for f in r["failures"]).most_common():
        print(f"  {ei:<30}: {n}")
    print("\nFull failure list:")
    for f in r["failures"]:
        tag = " [FALSE-ACTION]" if f["is_fa"] else ""
        print(f"  [{f['id']}][{f['cat']}]{tag} expected={f['ei']}/{f['ea']!r}")
        print(f"    \"{f['text']}\"")
        print(f"    got: status={f['status']} action={f['aa']}")
        print()

    with open(ROOT_DIR/"eval"/"original_scorer_results.json","w") as fh:
        json.dump(r, fh, indent=2)
    print("[+] Saved to eval/original_scorer_results.json")

if __name__ == "__main__":
    main()
