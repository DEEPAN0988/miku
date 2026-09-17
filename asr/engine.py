"""
Offline Speech Recognition Engine for Miku.
Features real-time Voice Activity Detection (VAD), live volume metering,
Windows native SAPI speech transcription, and acoustic DTW template matching.
Zero cloud APIs, zero external black-box weights.
"""

import os
import pickle
import time
import wave
import subprocess
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable
from wakeword.audio_features import extract_mfcc
from .acoustic import AcousticTemplateMatcher

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEMPLATES_PATH = os.path.join(DATA_DIR, "asr_templates.pkl")
TEMP_WAV_PATH = os.path.join(DATA_DIR, "temp_mic.wav")
SAPI_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "sapi_transcribe.ps1")

TARGET_COMMANDS = [
    "hi",
    "hello",
    "help",
    "open notepad",
    "open calculator",
    "open browser",
    "what is the time",
    "what is the date",
    "plan my day",
    "list my tasks",
    "take a photo",
    "read the screen",
    "system status",
    "scan bluetooth",
    "confirm yes",
    "confirm no"
]


class ASREngine:
    def __init__(self, templates_path: str = TEMPLATES_PATH):
        self.templates_path = templates_path
        self.matcher = AcousticTemplateMatcher()
        self._load_or_seed_templates()

    def _load_or_seed_templates(self):
        if os.path.exists(self.templates_path):
            try:
                with open(self.templates_path, "rb") as f:
                    self.matcher.templates = pickle.load(f)
                return
            except Exception:
                pass
        self._seed_synthetic_templates()

    def _seed_synthetic_templates(self):
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        for i, cmd in enumerate(TARGET_COMMANDS):
            base_f = 200 + (i * 60)
            synth = (
                np.sin(2 * np.pi * base_f * t)
                + 0.5 * np.sin(2 * np.pi * (base_f * 2.1) * t)
                + 0.25 * np.sin(2 * np.pi * (base_f * 3.7) * t)
            )
            syllables = max(1, len(cmd.split()))
            envelope = np.zeros_like(t)
            for s in range(syllables):
                center = (s + 0.5) / syllables * duration
                envelope += np.exp(-((t - center) ** 2) / 0.04)
            audio = synth * envelope
            mfcc = extract_mfcc(audio, sample_rate=sample_rate, target_frames=99)
            self.matcher.register_template(cmd, mfcc)

        self.save_templates()

    def save_templates(self):
        os.makedirs(os.path.dirname(self.templates_path), exist_ok=True)
        with open(self.templates_path, "wb") as f:
            pickle.dump(self.matcher.templates, f)

    def enroll_user_command(self, command_label: str, audio_samples: List[np.ndarray], sample_rate: int = 16000):
        for audio in audio_samples:
            mfcc = extract_mfcc(audio, sample_rate=sample_rate)
            self.matcher.register_template(command_label, mfcc)
        self.save_templates()

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribe audio using Windows native SAPI speech engine,
        falling back to local acoustic DTW template matching.
        """
        if len(audio) < int(sample_rate * 0.3):
            return "", 0.0

        # Method 1: Windows native SAPI recognizer via temporary WAV
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            # Normalize and clamp audio
            clamped = np.clip(audio, -1.0, 1.0)
            int16_data = (clamped * 32767).astype(np.int16)
            
            with wave.open(TEMP_WAV_PATH, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_data.tobytes())

            if os.path.exists(SAPI_SCRIPT_PATH):
                cmd = [
                    "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", SAPI_SCRIPT_PATH,
                    "-WavPath", TEMP_WAV_PATH
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                recognized = res.stdout.strip()
                if recognized:
                    return recognized, 0.95
        except Exception:
            pass

        # Method 2: Acoustic DTW template matching fallback
        mfcc = extract_mfcc(audio, sample_rate=sample_rate)
        cmd, conf = self.matcher.match(mfcc)
        return cmd, conf

    def listen_microphone(self, duration_sec: float = 3.0, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Standard fixed-duration recording."""
        try:
            import sounddevice as sd
            recording = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            return recording.flatten()
        except Exception:
            return None

    def listen_with_vad(
        self,
        level_callback: Optional[Callable[[float, bool, str], None]] = None,
        max_duration: float = 6.0,
        silence_timeout: float = 1.0,
        sample_rate: int = 16000,
        energy_threshold: float = 0.008
    ) -> Optional[np.ndarray]:
        """
        Stream microphone input with Voice Activity Detection (VAD).
        Automatically starts buffering when speech is detected,
        and finishes when silence is detected for silence_timeout seconds.
        Calls level_callback(rms, is_speech, status_message) for visual animation.
        """
        try:
            import sounddevice as sd
        except Exception:
            return None

        chunk_size = int(0.1 * sample_rate)  # 100ms per frame
        frames = []
        speech_detected = False
        silence_start = None
        start_time = time.time()

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
                while True:
                    # Check max recording limit
                    elapsed = time.time() - start_time
                    if elapsed > max_duration:
                        break

                    data, _ = stream.read(chunk_size)
                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    is_speech_chunk = rms > energy_threshold

                    if is_speech_chunk:
                        speech_detected = True
                        silence_start = None
                        frames.append(chunk)
                        msg = "Speaking..."
                    else:
                        if speech_detected:
                            frames.append(chunk)
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start >= silence_timeout:
                                # Silence threshold met after speech
                                break
                            msg = f"Silence... ({silence_timeout - (time.time() - silence_start):.1f}s)"
                        else:
                            # Keep sliding window of pre-speech audio (~0.3s)
                            frames.append(chunk)
                            if len(frames) > 3:
                                frames.pop(0)
                            msg = "Listening..."

                    if level_callback:
                        level_callback(rms, speech_detected, msg)

            if frames:
                return np.concatenate(frames)
            return None

        except Exception:
            return None
