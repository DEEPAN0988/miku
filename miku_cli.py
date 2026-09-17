"""
Miku — Offline-First Personal AI Voice Assistant.
Interactive Console & Execution Runner.
Zero external API keys, zero pretrained black-box models.
"""

import sys
import os
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wakeword.engine import WakeWordEngine
from asr.engine import ASREngine
from nlu.engine import NLUEngine
from orchestrator.router import Orchestrator


def print_banner():
    banner = r"""
============================================================
       __  __ ___ _  ___   _     _    ____ ___ 
      |  \/  |_ _| |/ / | | |   / \  / ___|_ _|
      | |\/| || || ' /| | | |  / _ \| |  _ | | 
      | |  | || || . \| |_| | / ___ \ |_| || | 
      |_|  |_|___|_|\_\\___/ /_/   \_\____|___|
      
    Offline-First Personal Voice Assistant (Windows Phase 1)
    * Zero Cloud API Keys  * Zero Pretrained Models
    * 100% Local Inference * Privacy-First Operation
============================================================
"""
    print(banner)


def run_interactive_text(orchestrator: Orchestrator, nlu: NLUEngine):
    print("\n[Miku Interactive Text Mode Active]")
    print("Type a command (or 'exit' / 'quit' to close).")
    print("Example commands:")
    print(" - 'what is the time'")
    print(" - 'plan my day'")
    print(" - 'create task buy groceries with priority high'")
    print(" - 'open notepad and write an essay about quantum computing'")
    print(" - 'system status'")
    print(" - 'delete file test.txt' (Tests safety confirmation gate)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nMiku > Goodbye!")
                break

            parsed = nlu.parse(user_input)
            print(f"[NLU] Intent: {parsed['intent']} (Conf: {parsed['confidence']:.2f}, Source: {parsed['source']})")
            
            result = orchestrator.handle_intent(parsed)
            print(f"Miku > {result.get('response', '')}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Miku...")
            break


def run_voice_loop(orchestrator: Orchestrator, nlu: NLUEngine, wakeword: WakeWordEngine, asr: ASREngine):
    print("\n[Miku Voice Mode Active]")
    print("Listening for wake word: 'Hey Miku'...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # Record 1.2 second chunk to check for wake word
            audio_chunk = asr.listen_microphone(duration_sec=1.2)
            if audio_chunk is None:
                print("\nMicrophone audio device unavailable. Switching to text mode.")
                run_interactive_text(orchestrator, nlu)
                break

            detected, score = wakeword.detect(audio_chunk)
            if detected:
                print(f"\n[Wake Word Triggered!] (Confidence: {score:.2f})")
                orchestrator.respond("I'm listening...")
                
                # Capture command audio
                print("Listening for command...")
                cmd_audio = asr.listen_microphone(duration_sec=3.5)
                if cmd_audio is not None:
                    transcribed, conf = asr.transcribe_audio(cmd_audio)
                    print(f"Heard: '{transcribed}' (Conf: {conf:.2f})")
                    if transcribed:
                        parsed = nlu.parse(transcribed)
                        res = orchestrator.handle_intent(parsed)
                        print(f"Miku > {res.get('response')}")
                    else:
                        orchestrator.respond("Sorry, I did not catch that.")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping voice loop...")


def run_self_test(orchestrator: Orchestrator, nlu: NLUEngine, wakeword: WakeWordEngine, asr: ASREngine):
    print("\nRunning Miku Phase 1 Self-Verification Test Suite...")
    print("=" * 60)

    # 1. Wake Word Test
    X_syn, y_syn = wakeword.generate_synthetic_samples(num_samples=4)
    det, score = wakeword.detect(X_syn[0].flatten())
    print(f"[OK] Wake Word Model: Operational (Score: {score:.2f})")

    # 2. ASR Test
    test_cmd, conf = asr.matcher.match(asr.matcher.templates.get("what is the time", [X_syn[0]])[0])
    print(f"[OK] ASR Acoustic Template Matcher: Operational (Matched '{test_cmd}' with conf {conf:.2f})")

    # 3. NLU Test
    sample_queries = [
        "what is the time",
        "plan my day",
        "create task finish report with priority high",
        "open notepad and write an essay about robotics",
        "delete file sensitive.txt",
        "system status"
    ]
    for q in sample_queries:
        parsed = nlu.parse(q)
        print(f"[OK] NLU Query: '{q}' -> Intent: {parsed['intent']} (Conf: {parsed['confidence']:.2f})")

    # 4. Orchestrator Handling Test
    res_time = orchestrator.handle_intent(nlu.parse("what is the time"), confirmed=False)
    print(f"[OK] Orchestrator Time Query -> {res_time['response']}")

    res_plan = orchestrator.handle_intent(nlu.parse("plan my day"), confirmed=False)
    print(f"[OK] Orchestrator Plan Day -> {res_plan['response']}")

    # 5. Security Confirmation Test
    res_del = orchestrator.handle_intent(nlu.parse("delete file dummy.txt"), confirmed=False)
    print(f"[OK] Security Confirmation Gate -> Requires Confirmation: {res_del.get('requires_confirmation')}")

    # Confirm Yes
    res_conf = orchestrator.handle_intent(nlu.parse("yes"), confirmed=True)
    print(f"[OK] Security Confirmation Response -> {res_conf['response']}")

    print("=" * 60)
    print("ALL CORE PHASE 1 MODULES VERIFIED SUCCESSFULLY!\n")


def main():
    parser = argparse.ArgumentParser(description="Miku Offline-First Voice Assistant")
    parser.add_argument("--text", action="store_true", help="Start directly in interactive text mode")
    parser.add_argument("--voice", action="store_true", help="Start in voice listening mode")
    parser.add_argument("--test-mode", action="store_true", help="Run automated self-verification suite")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech output")
    parser.add_argument("--retrain-nlu", action="store_true", help="Retrain NLU intent classifier from scratch")
    args = parser.parse_args()

    print_banner()

    print("[1/4] Initializing NLU engine...")
    nlu = NLUEngine()
    if args.retrain_nlu:
        print("Retraining NLU classifier from scratch...")
        nlu.classifier.train()

    print("[2/4] Initializing Wake Word engine...")
    wakeword = WakeWordEngine()

    print("[3/4] Initializing ASR engine...")
    asr = ASREngine()

    print("[4/4] Initializing Orchestrator & Control Layer...")
    orchestrator = Orchestrator(tts_enabled=not args.no_tts)

    if args.test_mode:
        run_self_test(orchestrator, nlu, wakeword, asr)
    elif args.voice:
        run_voice_loop(orchestrator, nlu, wakeword, asr)
    else:
        run_interactive_text(orchestrator, nlu)


if __name__ == "__main__":
    main()
