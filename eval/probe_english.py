"""
Quick probe script demonstrating Miku's upgraded English comprehension:
1. Negation refusal
2. Compound execution
3. Usable real-time alias teaching
4. Pronoun resolution
5. Destructive action safeguard
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from miku.orchestrator import MikuOrchestrator
from miku.ipc.messages import STTTranscriptMsg

def main():
    orch = MikuOrchestrator()
    orch.start()
    daemon = orch.cognitive_daemon

    test_queries = [
        ("Negation Refusal", "do not open edge right now"),
        ("Compound Multi-Step", "open notepad and then turn up volume"),
        ("Natural Teaching", "when I say surf I mean open edge"),
        ("Taught Term Execution", "open surf"),
        ("Pronoun Coreference", "close it"),
        ("Destructive Safeguard", "delete file secret_data.txt")
    ]

    print("======================================================================")
    print("      MIKU ADVANCED ENGLISH UNDERSTANDING CAPABILITIES PROBE")
    print("======================================================================")

    for category, q in test_queries:
        res = daemon.handle_transcript(STTTranscriptMsg(text=q, confidence=1.0))
        status = res.get("status")
        msg = res.get("message")
        action = getattr(res.get("action_msg"), "action_type", res.get("best_action"))
        target = getattr(res.get("action_msg"), "target", "")

        print(f"\n[{category}]")
        print(f"  Input  : \"{q}\"")
        print(f"  Status : {status}")
        if action:
            print(f"  Action : {action} (target='{target}')")
        if msg:
            print(f"  Miku   : {msg}")

    print("\n" + "=" * 70)
    orch.shutdown()

if __name__ == "__main__":
    main()
