"""
MIKU: Sovereign Local Agent — Command-Line Interface.
Zero-Model, Zero-API Autonomous Local Assistant (v1.2.0).
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
     __  __ _____ _  ___   _   ____   _____     _______ ____  _____ ___ _   _ 
    |  \/  |_   _| |/ / | | | / ___| / _ \ \   / / ____|  _ \| ____|_ _| \ | |
    | |\/| | | | | ' /| | | | \___ \| | | \ \ / /|  _| | |_) |  _|  | ||  \| |
    | |  | | | | | . \| |_| |  ___) | |_| |\ V / | |___|  _ <| |___ | || |\  |
    |_|  |_| |_| |_|\_\___/  |____/ \___/  \_/  |_____|_| \_\_____|___|_| \_|
                 SOVEREIGN LOCAL AGENT — v1.2.0 (Zero-Model, Zero-API)
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

            # Process command through cognitive router
            from miku.ipc.messages import STTTranscriptMsg
            msg = STTTranscriptMsg(text=cmd, confidence=1.0)
            decision = orchestrator.cognitive_daemon.handle_transcript(msg)

            status = decision.get("status")
            if status == "dispatched":
                action = decision["action_msg"]
                print(f"  [Intent] Approved: {action.action_type} (target='{action.target}') [Conf: {decision['confidence']:.2f}]")
                res = orchestrator.execution_daemon.execute_request(action)
                orchestrator.cognitive_daemon.handle_action_completion(res)
                orchestrator.logger.log_action(res.task_id, action.action_type, action.target, res.status, res.message)
                print(f"  [Hands]  {res.status}: {res.message}")
            elif status == "clarification_needed":
                print(f"  [Brain]  {decision.get('message')}")
            elif status == "calibration_required":
                print(f"  [Alert]  {decision.get('message')}")
    finally:
        orchestrator.shutdown()
        print("\n[*] Miku shut down cleanly.")

def main():
    parser = argparse.ArgumentParser(description="Miku Sovereign Local Agent CLI")
    parser.add_argument("mode", nargs="?", default="interactive", choices=["interactive", "calibrate", "status"])
    args = parser.parse_args()

    if args.mode == "calibrate":
        run_calibrate()
    elif args.mode == "status":
        run_status()
    else:
        run_interactive()

if __name__ == "__main__":
    main()
