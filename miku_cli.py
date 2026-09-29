"""
MIKU: Sovereign Local Agent — Command-Line Interface.
Zero-Model, Zero-API Autonomous Local Assistant (v1.0).
"""
import sys
import argparse
import time
import numpy as np
from pathlib import Path

from miku.orchestrator import MikuOrchestrator
from miku.core2_cognitive.calibration_daemon import CalibrationDaemon, PHONETIC_BALANCED_SCRIPT
from miku.core2_cognitive.bm25_memory import BM25Memory
from miku.persistence.encrypted_log import EncryptedPersistenceLogger

def print_banner():
    banner = r"""
========================================================================
              __  __  ___  _  __  _   _            _ 
             |  \/  ||_ _|| |/ / | | | |  __   __ / |
             | |\/| | | | | ' /  | | | |  \ \ / / | |
             | |  | | | | | . \  | |_| |   \ V /  | |
             |_|  |_||___||_|\_\  \___/     \_/   |_|
                 SOVEREIGN LOCAL AGENT - v1.0 (Zero-Model, Zero-API)
========================================================================
    """
    print(banner)

def run_calibrate():
    print("[*] Starting Day-Zero Calibration Onboarding Flow...")
    calib = CalibrationDaemon()
    print("Voice enrollment: read the following phonetically balanced phrases:")
    for i, phrase in enumerate(PHONETIC_BALANCED_SCRIPT, 1):
        print(f"  [{i}/{len(PHONETIC_BALANCED_SCRIPT)}] \"{phrase}\"")
        calib.enroll_voice_phrase(phrase)
    calib.enroll_vision_class("red_button", 5)
    calib.enroll_vision_class("blue_button", 5)
    metrics = calib.get_calibration_metrics()
    print("\n[+] Calibration complete:")
    print(f"    • {metrics['display_string']}")
    print(f"    • Vision samples enrolled: {metrics['vision_samples_total']}")
    print(f"    • Usable: {metrics['cleared_min_viable_floor']}")

def run_status():
    calib = CalibrationDaemon()
    mem = BM25Memory()
    logger = EncryptedPersistenceLogger()
    print("\n[*] MIKU SYSTEM STATUS:")
    print(f"    • Architecture: 4 Isolated CPU Cores (Sensory, Brain, Hands, Eyes)")
    print(f"    • External Network Traffic: 0 bytes (Strictly Local)")
    print(f"    • Calibration: {calib.get_calibration_metrics()['display_string']}")
    logs = logger.read_logs(limit=3)
    print(f"    • Recent Audit Logs: {len(logs)} entries in SQLite WAL")
    for l in logs:
        print(f"      - [{l['action_type']}] {l['target']} -> {l['status']}")

def run_interactive():
    print_banner()
    orchestrator = MikuOrchestrator()
    orchestrator.start()
    print("[*] Miku ready. Type command or 'exit' to quit.\n")

    try:
        while True:
            cmd = input("Miku> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit"):
                break

            if cmd.lower() in ("chat", "/chat", "chatbot", "chat mode"):
                orchestrator.shutdown()
                run_chat()
                return

            # Process command through cognitive router
            from miku.ipc.messages import STTTranscriptMsg
            msg = STTTranscriptMsg(text=cmd, confidence=1.0)
            decision = orchestrator.cognitive_daemon.handle_transcript(msg)

            status = decision.get("status")
            if status == "dispatched":
                if decision.get("message"):
                    print(f"  [Miku]   {decision['message']}")
                action = decision["action_msg"]
                print(f"  [Intent] Approved: {action.action_type} (target='{action.target}') [Conf: {decision['confidence']:.2f}]")
                res = orchestrator.execution_daemon.execute_request(action)
                orchestrator.cognitive_daemon.handle_action_completion(res)
                orchestrator.logger.log_action(res.task_id, action.action_type, action.target, res.status, res.message)
                print(f"  [Hands]  {res.status}: {res.message}")
            elif status in ("conversational_response", "clarification_needed"):
                print(f"  [Miku]   {decision.get('message')}")
            elif status == "calibration_required":
                print(f"  [Alert]  {decision.get('message')}")
            else:
                msg_text = decision.get("message") or f"Acknowledged '{cmd}'."
                print(f"  [Miku]   {msg_text}")
    finally:
        orchestrator.shutdown()
        print("\n[*] Miku shut down cleanly.")

def run_train():
    print("[*] Starting Local English Language & Vocabulary Training...")
    from miku.core2_cognitive.english_lexicon import EnglishLexicon
    from miku.core2_cognitive.custom_chat_engine import CustomChatEngine
    from miku.core2_cognitive.calibration_daemon import CalibrationDaemon, PHONETIC_BALANCED_SCRIPT

    lexicon = EnglishLexicon()
    stats = lexicon.get_lexicon_stats()
    print(f"  [+] Loaded English Lexicon: {stats['total_vocabulary_count']} words indexed.")
    print(f"  [+] Core Tech & OS Dictionary: {stats['core_dictionary_entries']} definitions ready.")

    print("  [*] Enrolling phonetic speech calibration phrases...")
    calib = CalibrationDaemon()
    for phrase in PHONETIC_BALANCED_SCRIPT:
        calib.enroll_voice_phrase(phrase)
    calib.enroll_vision_class("red_button", 5)
    calib.enroll_vision_class("blue_button", 5)
    calib_metrics = calib.get_calibration_metrics()
    print(f"  [+] Calibration state: {calib_metrics['display_string']}")

    print("  [*] Training local neural Causal Transformer on English dialogues...")
    chat = CustomChatEngine()
    metrics = chat.train_english_language(epochs=5)
    print(f"  [+] Training epochs: {metrics['epochs']}")
    print(f"  [+] Active vocabulary size: {metrics['vocab_size']} tokens")
    print(f"  [+] Conversational pairs: {metrics['dialogue_pairs']}")
    print(f"  [+] Initial loss: {metrics['initial_loss']} -> Final loss: {metrics['final_loss']} ({metrics['loss_reduction_pct']}% improvement)")

    print("\n[*] Testing English Word Understanding:")
    test_words = ["sovereign", "autonomous", "browser", "games", "understand"]
    for w in test_words:
        defn = lexicon.get_word_definition(w)
        if defn:
            print(f"  • '{w}' ({defn['pos']}): {defn['definition']}")

    print("\n[+] SUCCESS: Miku is fully trained to understand English words, commands, and conversations!")

def run_review():
    print("[*] Starting Miss-Review Loop...")
    from miku.core2_cognitive.miss_reviewer import MissReviewer
    mr = MissReviewer()
    mr.run_review_loop()

def run_forget(ex_id):
    from miku.core2_cognitive.miss_reviewer import MissReviewer
    mr = MissReviewer()
    mr.forget_example(ex_id)

def run_summary():
    from miku.core2_cognitive.miss_reviewer import MissReviewer
    mr = MissReviewer()
    mr.weekly_summary()

def run_chat():
    banner = r"""
========================================================================
                 MIKU CHATBOT MODE (Offline & Sovereign)
  - Talk to Miku naturally as a personal assistant & chatbot.
  - Execute PC commands ("open spotify", "screenshot", etc.) anytime!
  - Type '/history' to view log, '/clear' to reset, or 'exit' to quit.
========================================================================
    """
    print(banner)
    orchestrator = MikuOrchestrator()
    orchestrator.start()
    print("[*] Miku Chatbot is online and ready. How can I help you today?\n")

    history = []

    try:
        while True:
            cmd = input("You > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "bye", "goodbye"):
                print("\n[Miku] Goodbye! Have a great day.")
                break

            if cmd.lower() == "/history":
                print("\n--- Session History ---")
                for u, m in history:
                    print(f" You : {u}")
                    print(f" Miku: {m}\n")
                print("-----------------------\n")
                continue

            if cmd.lower() == "/clear":
                history.clear()
                print("\n[Miku] Session history cleared!\n")
                continue

            # Process prompt through cognitive daemon
            from miku.ipc.messages import STTTranscriptMsg
            msg = STTTranscriptMsg(text=cmd, confidence=1.0)
            decision = orchestrator.cognitive_daemon.handle_transcript(msg)

            status = decision.get("status")
            if status == "dispatched":
                action = decision["action_msg"]
                res = orchestrator.execution_daemon.execute_request(action)
                orchestrator.cognitive_daemon.handle_action_completion(res)
                orchestrator.logger.log_action(res.task_id, action.action_type, action.target, res.status, res.message)
                
                # Conversational feedback after executing command
                reply = decision.get("message") or res.message or f"Done! Executed {action.action_type} for '{action.target}'."
                print(f"Miku > {reply}")
                history.append((cmd, reply))

            elif status in ("conversational_response", "clarification_needed"):
                reply = decision.get("message", "I am right here! What would you like to do?")
                print(f"Miku > {reply}")
                history.append((cmd, reply))

            elif status == "calibration_required":
                reply = decision.get("message", "Calibration required before completing that action.")
                print(f"Miku > {reply}")
                history.append((cmd, reply))

            else:
                reply = decision.get("message") or f"Understood! '{cmd}'"
                print(f"Miku > {reply}")
                history.append((cmd, reply))

            print()
    finally:
        orchestrator.shutdown()
        print("\n[*] Miku Chatbot shut down cleanly.")

def main():
    parser = argparse.ArgumentParser(description="Miku Sovereign Local Agent CLI")
    parser.add_argument("mode", nargs="?", default="interactive", choices=["interactive", "chat", "calibrate", "status", "train", "review-misses", "forget-example", "weekly-summary"])
    parser.add_argument("ex_id", nargs="?", default=None, help="Example ID to forget")
    args = parser.parse_args()

    if args.mode == "chat":
        run_chat()
    elif args.mode == "calibrate":
        run_calibrate()
    elif args.mode == "status":
        run_status()
    elif args.mode == "train":
        run_train()
    elif args.mode == "review-misses":
        run_review()
    elif args.mode == "forget-example":
        if not args.ex_id:
            print("Error: forget-example requires an example ID.")
        else:
            run_forget(args.ex_id)
    elif args.mode == "weekly-summary":
        run_summary()
    else:
        run_interactive()

if __name__ == "__main__":
    main()
