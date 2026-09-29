import sys
import json
import time
import uuid
import tempfile
from pathlib import Path
from multiprocessing import Queue

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from miku.ipc.messages import STTTranscriptMsg

def get_isolated_daemon(temp_dir: str):
    from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
    from miku.core2_cognitive.realtime_learner import RealtimeLearner
    from miku.core2_cognitive.english_lexicon import EnglishLexicon

    # Patch paths to use temp directory
    orig_memory = RealtimeLearner.__init__.__defaults__
    orig_words = EnglishLexicon.__init__.__defaults__

    try:
        q = Queue()
        daemon = CognitiveDaemon(action_queue=q)
        daemon.learner.filepath = Path(temp_dir) / "miku_learned_memory.json"
        daemon.learner.data = {"custom_aliases": {}, "category_preferences": {}, "history": []}
        
        daemon.lexicon.words_filepath = Path(temp_dir) / "miku_learned_words.json"
        daemon.lexicon.learned_words = {}
        return daemon
    finally:
        pass

def evaluate_item_isolated(item: dict, previous_messages: list = None) -> dict:
    """Evaluates a single item in total isolation. If previous_messages is provided, 
    simulates a multi-turn conversation before the main item."""
    with tempfile.TemporaryDirectory() as temp_dir:
        daemon = get_isolated_daemon(temp_dir)
        
        if previous_messages:
            for pm in previous_messages:
                msg = STTTranscriptMsg(text=pm, confidence=1.0)
                daemon.handle_transcript(msg)

        msg = STTTranscriptMsg(text=item["text"], confidence=1.0)
        t0 = time.perf_counter()
        res = daemon.handle_transcript(msg)
        latency = (time.perf_counter() - t0) * 1000.0
        
        status = res.get("status")
        actual_action = res.get("best_action")
        if actual_action is None and status == "dispatched":
            actual_action = getattr(res.get("action_msg"), "action_type", None)
            
        return {
            "status": status,
            "actual_action": actual_action,
            "latency": latency,
            "message": res.get("message", ""),
            "action_msg": res.get("action_msg")
        }

def check_harness_isolation():
    print("[*] Running harness isolation tests...")
    # Test 1: Order independence
    item1 = {"text": "open edge"}
    item2 = {"text": "do it again"}
    
    # Run item1 then item2 in separate isolated envs
    res1 = evaluate_item_isolated(item1)
    res2 = evaluate_item_isolated(item2)
    
    if res2["status"] == "confirmation_required":
        print("[-] FAIL: 'do it again' failed isolation (it remembered 'open edge').")
        sys.exit(1)
        
    # Run item2 then item1 in separate isolated envs
    res2_rev = evaluate_item_isolated(item2)
    res1_rev = evaluate_item_isolated(item1)
    
    if res1["actual_action"] != res1_rev["actual_action"] or res2["status"] != res2_rev["status"]:
        print("[-] FAIL: Order dependence detected.")
        sys.exit(1)

    # Test 2: State leak test (Round 3 bug)
    # suggest an app -> open games
    res_leak = evaluate_item_isolated({"text": "open games"}, previous_messages=["suggest an app"])
    if res_leak["status"] != "clarification_needed" or "Which game" not in res_leak["message"]:
        print(f"[-] FAIL: State leak detected! 'open games' returned: {res_leak}")
        sys.exit(1)

    # Genuine answer test
    res_answer = evaluate_item_isolated({"text": "wuthering waves"}, previous_messages=["suggest an app"])
    if res_answer["status"] != "dispatched" or res_answer["actual_action"] != "open_app":
        print(f"[-] FAIL: Genuine answer failed! Returned: {res_answer}")
        sys.exit(1)

    print("[+] Harness isolation tests passed.")

if __name__ == "__main__":
    check_harness_isolation()
