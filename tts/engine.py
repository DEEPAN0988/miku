"""
Local Offline Text-To-Speech (TTS) Engine for Miku.
Uses Windows SAPI5 / pyttsx3.
100% offline, zero cloud API, zero external black-box models.
"""

import threading
import os
from typing import Optional, List, Dict, Any


class TTSEngine:
    def __init__(self, rate: int = 175, volume: float = 0.9, voice_index: int = 0):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._lock = threading.Lock()
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init('sapi5')
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            voices = self._engine.getProperty('voices')
            if voices and len(voices) > self.voice_index:
                self._engine.setProperty('voice', voices[self.voice_index].id)
        except Exception as e:
            # Fallback if pyttsx3 fails to initialize COM
            self._engine = None

    def get_available_voices(self) -> List[Dict[str, Any]]:
        voices_list = []
        try:
            import pyttsx3
            eng = pyttsx3.init('sapi5')
            voices = eng.getProperty('voices')
            for i, v in enumerate(voices):
                voices_list.append({
                    "index": i,
                    "id": v.id,
                    "name": v.name,
                    "languages": getattr(v, 'languages', [])
                })
        except Exception:
            pass
        return voices_list

    def speak(self, text: str, block: bool = True):
        """
        Speak text through the system audio device.
        """
        if not text or not text.strip():
            return

        def _do_speak():
            with self._lock:
                try:
                    import pyttsx3
                    # Re-init engine per thread to prevent COM concurrency conflicts on Windows
                    engine = pyttsx3.init('sapi5')
                    engine.setProperty('rate', self.rate)
                    engine.setProperty('volume', self.volume)
                    voices = engine.getProperty('voices')
                    if voices and len(voices) > self.voice_index:
                        engine.setProperty('voice', voices[self.voice_index].id)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    # Fallback to PowerShell speech synthesis if COM is locked
                    safe_text = text.replace("'", "''").replace('"', '\"')
                    os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{safe_text}\')" >nul 2>&1')

        if block:
            _do_speak()
        else:
            t = threading.Thread(target=_do_speak, daemon=True)
            t.start()

    def save_to_file(self, text: str, output_path: str) -> bool:
        """Save synthesized speech to a local .wav file."""
        with self._lock:
            try:
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                return os.path.exists(output_path)
            except Exception:
                return False
