"""
Local Offline Text-To-Speech (TTS) Engine for Miku.
Uses Windows SAPI5 / pyttsx3 with Female Voice by default.
100% offline, zero cloud API, zero external black-box models.
"""

import threading
import os
from typing import Optional, List, Dict, Any


class TTSEngine:
    def __init__(self, rate: int = 175, volume: float = 0.95, prefer_female: bool = True):
        self.rate = rate
        self.volume = volume
        self.prefer_female = prefer_female
        self.voice_id = None
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
            
            # Find female voice (e.g. Zira, Hazel, Eva, or any marked female)
            female_voice = None
            if self.prefer_female and voices:
                for v in voices:
                    name_lower = v.name.lower()
                    if any(k in name_lower for k in ["zira", "female", "eva", "hazel", "catherine", "susan"]):
                        female_voice = v.id
                        break
                # If not matched by name, check index 1 (standard Windows female voice slot)
                if not female_voice and len(voices) > 1:
                    female_voice = voices[1].id
                elif not female_voice and len(voices) > 0:
                    female_voice = voices[0].id

            self.voice_id = female_voice or (voices[0].id if voices else None)
            if self.voice_id:
                self._engine.setProperty('voice', self.voice_id)
        except Exception:
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
        Speak text through system audio speaker using female voice.
        """
        if not text or not text.strip():
            return

        def _do_speak():
            with self._lock:
                try:
                    import pyttsx3
                    engine = pyttsx3.init('sapi5')
                    engine.setProperty('rate', self.rate)
                    engine.setProperty('volume', self.volume)
                    if self.voice_id:
                        engine.setProperty('voice', self.voice_id)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception:
                    # Fallback to PowerShell SpeechSynthesizer with Female hint
                    safe_text = text.replace("'", "''").replace('"', '\"')
                    ps_cmd = f"""
                    Add-Type -AssemblyName System.Speech;
                    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
                    try {{ $synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female) }} catch {{}};
                    $synth.Speak('{safe_text}')
                    """
                    os.system(f'powershell -NoProfile -Command "{ps_cmd}" >nul 2>&1')

        if block:
            _do_speak()
        else:
            t = threading.Thread(target=_do_speak, daemon=True)
            t.start()

    def save_to_file(self, text: str, output_path: str) -> bool:
        """Save synthesized speech to local .wav file."""
        with self._lock:
            try:
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                if self.voice_id:
                    engine.setProperty('voice', self.voice_id)
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                return os.path.exists(output_path)
            except Exception:
                return False
