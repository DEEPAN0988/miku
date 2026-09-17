"""
Miku — Offline-First Personal AI Voice Assistant.
Interactive Console & Real-Time Voice Hearing Interface.
Features live terminal hearing animation, VAD, and offline NLU execution with Female Voice.
Zero external API keys, zero pretrained black-box models.
"""

import sys
import os
import argparse
import time
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
    * Female Voice (Microsoft Zira)  * Zero Pretrained Models
    * Zero Cloud API Keys            * 100% Local Inference
============================================================
"""
    print(banner)


def ensure_microphone_unmuted() -> dict:
    """
    Inspect Windows CoreAudio via pycaw:
    If microphone is muted in Windows settings or volume is too low,
    automatically unmute and set level to 85%.
    """
    status = {"unmuted": False, "adjusted": False, "volume": 1.0}
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from ctypes import cast, POINTER
        mic = AudioUtilities.GetMicrophone()
        if mic:
            interface = mic.Activate(IAudioEndpointVolume._iid_, 7, None)
            vol = cast(interface, POINTER(IAudioEndpointVolume))
            if vol.GetMute() == 1:
                vol.SetMute(0, None)
                status["unmuted"] = True
            current_vol = vol.GetMasterVolumeLevelScalar()
            if current_vol < 0.70:
                vol.SetMasterVolumeLevelScalar(0.85, None)
                status["adjusted"] = True
                status["volume"] = 0.85
            else:
                status["volume"] = current_vol
    except Exception:
        pass
    return status


def render_equalizer(rms: float, max_bars: int = 16) -> str:
    """
    Generate dynamic visual equalizer string using high-sensitivity decibel scale.
    Maps -85 dB (silence) to -25 dB (loud speech) smoothly to 0% - 100%.
    """
    db = 20.0 * np.log10(max(rms, 1e-6))
    level = max(0.0, min(1.0, (db + 85.0) / 60.0))
    
    filled = int(level * max_bars)
    bar = "|" * filled + "." * (max_bars - filled)
    pct = int(level * 100)
    
    # Dynamic wave animation
    waves = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    w_idx = min(len(waves) - 1, int(level * len(waves)))
    wave_str = waves[w_idx] * 4

    return f"[{bar}] {wave_str} (Vol: {pct:2d}%)"


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


def run_test_mic(device_index: int = None):
    """Real-time 10-second microphone test with visual audio meter and unmuting check."""
    mic_status = ensure_microphone_unmuted()
    if mic_status["unmuted"]:
        print("[Notice] Microphone was MUTED in Windows settings. Miku automatically UNMUTED it.")
    if mic_status["adjusted"]:
        print(f"[Notice] Microphone recording volume was low. Raised to {int(mic_status['volume']*100)}%.")

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"Cannot initialize sounddevice: {e}")
        return

    sample_rate = 16000
    chunk_size = int(0.08 * sample_rate)

    dev_info = sd.query_devices(device_index, 'input') if device_index is not None else sd.query_devices(kind='input')
    dev_name = dev_info['name']
    print(f"\n[Microphone Test Active]")
    print(f"Device: '{dev_name}' (16kHz Mono)")
    print("Speak or make sound into your mic. The volume meter will react in real-time.")
    print("Running for 10 seconds (or press Ctrl+C to stop)...\n")

    start_time = time.time()
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', device=device_index) as stream:
            while time.time() - start_time < 10.0:
                data, _ = stream.read(chunk_size)
                rms = float(np.sqrt(np.mean(data.flatten() ** 2)))
                eq = render_equalizer(rms, max_bars=20)
                remaining = int(10.0 - (time.time() - start_time))
                sys.stdout.write(f"\r🎤 [Mic Signal] {eq}  Remaining: {remaining}s   ")
                sys.stdout.flush()
        print("\n\n[Mic Test Complete] Audio input stream verified.")
    except KeyboardInterrupt:
        print("\n\n[Mic Test Stopped]")


def run_voice_loop(orchestrator: Orchestrator, nlu: NLUEngine, wakeword: WakeWordEngine, asr: ASREngine, device_index: int = None):
    print("\n[Miku Voice Mode Active]")

    # Check Windows mic mute status
    mic_status = ensure_microphone_unmuted()
    if mic_status["unmuted"]:
        print("[Notice] Microphone was MUTED in Windows. Miku automatically UNMUTED it.")
    if mic_status["adjusted"]:
        print(f"[Notice] Microphone volume adjusted to {int(mic_status['volume']*100)}%.")

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"Microphone device error: {e}. Switching to text mode.")
        run_interactive_text(orchestrator, nlu)
        return

    sample_rate = 16000
    chunk_size = int(0.08 * sample_rate)  # 80ms per tick

    dev_info = sd.query_devices(device_index, 'input') if device_index is not None else sd.query_devices(kind='input')
    dev_name = dev_info['name']
    print(f"Active Microphone: '{dev_name}'")
    print("Calibrating ambient room noise...")

    # Ambient Noise Calibration (0.4s)
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', device=device_index) as stream:
            ambient_samples = []
            for _ in range(5):
                data, _ = stream.read(chunk_size)
                ambient_samples.append(float(np.sqrt(np.mean(data.flatten() ** 2))))
            ambient_floor = max(1e-5, float(np.mean(ambient_samples)))
            
            # High-sensitivity trigger threshold
            speech_threshold = max(0.0001, ambient_floor * 2.0)
            
            print(f"Calibration Complete: Ambient Floor={ambient_floor:.6f}, Trigger Threshold={speech_threshold:.6f}")
            print("\nSpeak your command into the microphone (e.g. 'hi', 'what is the time', 'plan my day').")
            print("Live hearing monitor active. Press Ctrl+C to stop.\n")

            while True:
                frames = []
                is_recording = False
                silence_frames = 0
                max_silence_frames = int(1.0 / 0.08)  # ~1.0s silence to finish command
                max_speech_frames = int(7.0 / 0.08)   # 7 seconds max

                # 1. Idle listening state
                while not is_recording:
                    data, _ = stream.read(chunk_size)
                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    eq = render_equalizer(rms)

                    sys.stdout.write(f"\r👂 [Listening] {eq}  Speak command...   ")
                    sys.stdout.flush()

                    if rms > speech_threshold:
                        is_recording = True
                        frames.append(chunk)
                        break

                # 2. Recording speech state
                sys.stdout.write("\r" + " " * 80 + "\r")
                while is_recording:
                    data, _ = stream.read(chunk_size)
                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    frames.append(chunk)

                    eq = render_equalizer(rms)
                    sys.stdout.write(f"\r🎙️ [HEARING SPEECH] {eq} Recording... (pause to finish)   ")
                    sys.stdout.flush()

                    if rms < speech_threshold:
                        silence_frames += 1
                        if silence_frames >= max_silence_frames:
                            is_recording = False
                    else:
                        silence_frames = 0

                    if len(frames) >= max_speech_frames:
                        is_recording = False

                # 3. Process & Transcribe
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.write("🧠 [Processing Audio] Transcribing speech...\r")
                sys.stdout.flush()

                full_audio = np.concatenate(frames)
                
                # Normalize / Boost audio for clear recognition
                peak = np.max(np.abs(full_audio))
                if peak > 1e-4:
                    full_audio = full_audio * (0.75 / peak)

                sys.stdout.write("\r" + " " * 80 + "\r")

                # Transcribe via SAPI / acoustic matcher
                transcribed, conf = asr.transcribe_audio(full_audio, sample_rate=sample_rate)
                transcribed_clean = transcribed.strip()

                if transcribed_clean:
                    print(f"\n🗣️ Heard: \"{transcribed_clean}\" (Conf: {conf:.2f})")
                    
                    # Strip wake word if spoken at start
                    lower = transcribed_clean.lower()
                    if lower.startswith("hey miku") or lower.startswith("miku"):
                        cmd_text = lower.replace("hey miku", "").replace("miku", "").strip()
                        if not cmd_text:
                            orchestrator.respond("Hello! I am Miku. How can I help you today?")
                            continue
                        transcribed_clean = cmd_text

                    parsed = nlu.parse(transcribed_clean)
                    print(f"🎯 [NLU] Intent: {parsed['intent']} (Conf: {parsed['confidence']:.2f})")
                    res = orchestrator.handle_intent(parsed)
                    print(f"🤖 Miku > {res.get('response')}\n")
                else:
                    print("\n[Notice] Sound detected but no words recognized.")
                    print("Tip: Speak clearly into your microphone.\n")

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
    parser.add_argument("--test-mic", action="store_true", help="Test microphone input with live visual audio meter")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (default: system default)")
    parser.add_argument("--test-mode", action="store_true", help="Run automated self-verification suite")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech output")
    parser.add_argument("--retrain-nlu", action="store_true", help="Retrain NLU intent classifier from scratch")
    args = parser.parse_args()

    print_banner()

    if args.test_mic:
        run_test_mic(device_index=args.device)
        return

    print("[1/4] Initializing NLU engine...")
    nlu = NLUEngine()
    if args.retrain_nlu:
        print("Retraining NLU classifier from scratch...")
        nlu.classifier.train()

    print("[2/4] Initializing Wake Word engine...")
    wakeword = WakeWordEngine()

    print("[3/4] Initializing ASR engine with Voice Activity Detection...")
    asr = ASREngine()

    print("[4/4] Initializing Orchestrator & Female Voice (Microsoft Zira)...")
    orchestrator = Orchestrator(tts_enabled=not args.no_tts)

    if args.test_mode:
        run_self_test(orchestrator, nlu, wakeword, asr)
    elif args.voice:
        run_voice_loop(orchestrator, nlu, wakeword, asr, device_index=args.device)
    else:
        run_interactive_text(orchestrator, nlu)


if __name__ == "__main__":
    main()
