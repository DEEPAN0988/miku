"""
Test refusal generality: paraphrases that appear in NO rule, template, or test.
Tests identity questions, capability questions, and OOS requests.
"""
import sys
sys.path.insert(0, 'c:/miku')
from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
from miku.ipc.messages import STTTranscriptMsg
from multiprocessing import Queue

d = CognitiveDaemon(action_queue=Queue())

CASES = [
    # Identity questions - expect ANSWER (conversational_response)
    ("who built you", True, "identity"),
    ("who's behind you", True, "identity"),
    ("tell me about your maker", True, "identity"),
    ("who created this assistant", True, "identity"),
    ("who is your developer", True, "identity"),
    # Capability questions - expect ANSWER
    ("what are you capable of", True, "capability"),
    ("what tasks can you handle", True, "capability"),
    ("what is your purpose", True, "capability"),
    # OOS - expect refusal (not dispatched)
    ("write me a poem about spring", False, "oos"),
    ("give me investment advice", False, "oos"),
    ("what is the capital of france", False, "oos"),
    ("tell me today's news", False, "oos"),
    ("explain quantum physics to me", False, "oos"),
    ("what's the weather like in tokyo", False, "oos"),
    ("help me cheat on my exam", False, "oos"),
]

print("=== REFUSAL GENERALITY TEST ===")
passed = 0
failed = 0
for text, expect_conversational, kind in CASES:
    r = d.handle_transcript(STTTranscriptMsg(text=text, confidence=1.0))
    status = r.get("status")
    action = r.get("best_action")
    is_dispatched = (status == "dispatched" and action not in (None, "None", "noop"))

    ok = (expect_conversational and status in ("conversational_response", "clarification_needed") and not is_dispatched) or \
         (not expect_conversational and not is_dispatched)

    mark = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{mark}] ({kind}) {text!r}")
    print(f"         status={status}  action={action}  msg={r.get('message','')[:60]!r}")

print(f"\nResult: {passed}/{len(CASES)} passed ({failed} failed)")
if failed > 0:
    sys.exit(1)
