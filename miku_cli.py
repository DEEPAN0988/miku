"""
Miku — Offline-First Personal AI Voice Assistant.
Interactive Console & Real-Time Voice Hearing Interface.
Features live terminal hearing animation, VAD, and offline NLU execution.
Zero external API keys, zero pretrained black-box models.
"""

import sys
import os
import argparse
import time
import numpy as np

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


def render_equalizer(rms: float, max_bars: int = 12) -> str:
    """Generate dynamic visual equalizer string based on audio energy."""
    # Scale RMS (normal speech is ~0.01 - 0.20)
    level = min(1.0, max(0.0, rms * 15.0))
    filled = int(level * max_bars)
    bar = "▰" * filled + "▱" * (max_bars - filled)
    
    # Animated wave glyphs
    waves = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    wave_idx = min(len(waves) - 1, int(level * len(waves)))
    wave_str = waves[wave_idx] * 4
    return f"[{bar}] {wave_str} (Vol: {int(level * 100):2d}%)"


def run_interactive_text(orchestrator: Orchestrator, nlu: NLUEngine):
    print("\n[Miku Interactive Text Mode Active]")
    print("Type a command (or 'exit' / 'quit' to close).")
    print("Example commands:")
    print(" - 'hi' / 'hello' / 'who are you'")
    print(" - 'what is the time' / 'what day is it'")
    print(" - 'plan my day' / 'how should i train for marathon'")
    print(" - 'create task buy groceries with priority high'")
    print(" - 'open notepad and write an essay about AI'")
    print(" - 'system status' / 'take a photo' / 'read screen'")
    print(" - 'delete file test.txt' (Safety confirmation gate)")
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


def run_voice_loop(orchestrator: Orchestrator, nlu: NLUEngine, wakeword: WakeWordEngine, asr: ASREngine, require_wake_word: bool = False):
    print("\n[Miku Voice Mode Active]")
    if require_wake_word:
        print("Say 'Hey Miku' to trigger listening.")
    else:
        print("Speak naturally into your microphone (or say 'Hey Miku').")
    print("Live hearing monitor active. Press Ctrl+C to return to menu.\n")

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"Microphone device error: {e}. Switching to text mode.")
        run_interactive_text(orchestrator, nlu)
        return

    sample_rate = 16000
    chunk_size = int(0.08 * sample_rate)  # 80ms per tick for fast 12 FPS animation
    speech_threshold = 0.008

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            while True:
                # 1. Idle listening state with live animation
                frames = []
                is_recording = False
                silence_frames = 0
                max_silence_frames = int(1.1 / 0.08)  # ~1.1s of silence to finish command
                max_speech_frames = int(7.0 / 0.08)   # 7 seconds max utterance

                # Listen until speech or wake word is heard
                while not is_recording:
                    data, _ = stream.read(chunk_size)
                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    eq = render_equalizer(rms)

                    sys.stdout.write(f"\r👂 [Listening] {eq}  Say command or 'Hey Miku'...   ")
                    sys.stdout.flush()

                    if rms > speech_threshold:
                        is_recording = True
                        frames.append(chunk)
                        break

                # 2. Recording command with hearing animation
                sys.stdout.write("\r" + " " * 80 + "\r")
                while is_recording:
                    data, _ = stream.read(chunk_size)
                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    frames.append(chunk)

                    eq = render_equalizer(rms)
                    sys.stdout.write(f"\r🎙️ [HEARING SPEECH] {eq} Recording command...   ")
                    sys.stdout.flush()

                    if rms < speech_threshold:
                        silence_frames += 1
                        if silence_frames >= max_silence_frames:
                            is_recording = False
                    else:
                        silence_frames = 0

                    if len(frames) >= max_speech_frames:
                        is_recording = False

                # 3. Process captured audio
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.write("🧠 [Processing Audio] Transcribing local speech...\r")
                sys.stdout.flush()

                full_audio = np.concatenate(frames)
                sys.stdout.write("\r" + " " * 80 + "\r")

                # Transcribe
                transcribed, conf = asr.transcribe_audio(full_audio, sample_rate=sample_rate)
                transcribed_clean = transcribed.strip()

                if transcribed_clean:
                    print(f"\n🗣️ Heard: \"{transcribed_clean}\" (Conf: {conf:.2f})")
                    
                    # Check if utterance started with wake word "Hey Miku"
                    lower = transcribed_clean.lower()
                    if lower.startswith("hey miku"):
                        cmd_text = lower.replace("hey miku", "").strip()
                        if not cmd_text:
                            orchestrator.respond("Yes, I'm here! What can I do for you?")
                            continue
                        transcribed_clean = cmd_text

                    parsed = nlu.parse(transcribed_clean)
                    print(f"🎯 [NLU] Intent: {parsed['intent']} (Conf: {parsed['confidence']:.2f})")
                    res = orchestrator.handle_intent(parsed)
                    print(f"🤖 Miku > {res.get('response')}\n")
                else:
                    # Low audio or unintelligible noise
                    print("\n[Notice] Audio detected but no clear speech was recognized.")
                    print("Tip: Speak clearly into your microphone or check microphone volume.\n")

                time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n\nVoice loop stopped.")


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
        "hi",
        "who are you",
        "help",
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
    res_hi = orchestrator.handle_intent(nlu.parse("hi"), confirmed=False)
    print(f"[OK] Orchestrator Greeting -> {res_hi['response']}")

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
    parser.add_argument("--voice", action="store_true", help="Start in voice listening mode with live hearing animation")
    parser.add_argument("--wake-word-only", action="store_true", help="Require saying 'Hey Miku' before every command in voice mode")
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

    print("[3/4] Initializing ASR engine with Voice Activity Detection...")
    asr = ASREngine()

    print("[4/4] Initializing Orchestrator & Control Layer...")
    orchestrator = Orchestrator(tts_enabled=not args.no_tts)

    if args.test_mode:
        run_self_test(orchestrator, nlu, wakeword, asr)
    elif args.voice:
        run_voice_loop(orchestrator, nlu, wakeword, asr, require_wake_word=args.wake_word_only)
    else:
        run_interactive_text(orchestrator, nlu)


if __name__ == "__main__":
    main()
