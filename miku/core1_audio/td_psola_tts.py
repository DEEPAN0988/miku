"""
TD-PSOLA (Time-Domain Pitch-Synchronous Overlap and Add) Concatenative Synthesizer.
Core 1: Audio Daemon (Sensory & Voice)
"""
import numpy as np
from typing import Dict, List, Tuple

class TDPSOLASynthesizer:
    def __init__(self, sample_rate: int = 16000, base_f0: float = 140.0):
        self.sample_rate = sample_rate
        self.base_f0 = base_f0
        self.pitch_period = int(self.sample_rate / self.base_f0)  # e.g., ~114 samples
        self.phone_inventory = self._build_diphone_inventory()

    def _build_diphone_inventory(self) -> Dict[str, np.ndarray]:
        """
        Builds a deterministic acoustic inventory of vowel/consonant formants.
        Formants modeled with harmonic series + resonant filters.
        """
        inventory = {}
        # Vowels (F1, F2, F3 frequencies)
        vowel_formants = {
            "a": (800, 1200, 2500),
            "e": (500, 1800, 2600),
            "i": (300, 2300, 3000),
            "o": (500, 900, 2400),
            "u": (350, 800, 2200),
            "sil": (0, 0, 0)
        }

        duration_samples = int(self.sample_rate * 0.15)  # 150ms per phoneme
        t = np.linspace(0, 0.15, duration_samples, endpoint=False)

        for phone, (f1, f2, f3) in vowel_formants.items():
            if phone == "sil":
                inventory[phone] = np.zeros(int(self.sample_rate * 0.05), dtype=np.float32)
                continue

            # Harmonic excitation at base_f0
            excitation = np.sin(2 * np.pi * self.base_f0 * t)
            # Add harmonics
            for h in range(2, 8):
                excitation += (1.0 / h) * np.sin(2 * np.pi * (self.base_f0 * h) * t)

            # Resonance amplification
            resonances = (
                0.5 * np.sin(2 * np.pi * f1 * t) +
                0.3 * np.sin(2 * np.pi * f2 * t) +
                0.2 * np.sin(2 * np.pi * f3 * t)
            )
            wave = excitation * resonances
            # Fade in/out to prevent clicks
            window = np.hanning(duration_samples)
            inventory[phone] = (wave * window / (np.max(np.abs(wave)) + 1e-6)).astype(np.float32)

        return inventory

    def _find_pitch_marks(self, signal: np.ndarray) -> List[int]:
        """
        Finds pitch-synchronous marker points spaced approximately at the pitch period.
        """
        marks = []
        cur = self.pitch_period
        while cur < len(signal) - self.pitch_period:
            # Search within a small window around cur for local peak
            window_start = max(0, cur - int(self.pitch_period * 0.3))
            window_end = min(len(signal), cur + int(self.pitch_period * 0.3))
            peak_idx = window_start + int(np.argmax(np.abs(signal[window_start:window_end])))
            marks.append(peak_idx)
            cur = peak_idx + self.pitch_period
        return marks

    def psola_resynthesize(
        self,
        signal: np.ndarray,
        pitch_factor: float = 1.0,
        time_factor: float = 1.0
    ) -> np.ndarray:
        """
        Performs TD-PSOLA algorithm:
        pitch_factor: >1.0 increases pitch, <1.0 decreases pitch
        time_factor: >1.0 slows down duration, <1.0 speeds up duration
        """
        if len(signal) < self.pitch_period * 3:
            return signal

        pitch_marks = self._find_pitch_marks(signal)
        if len(pitch_marks) < 2:
            return signal

        new_period = int(self.pitch_period / pitch_factor)
        out_len = int(len(signal) * time_factor)
        output = np.zeros(out_len + new_period * 4, dtype=np.float32)

        out_idx = 0
        in_idx_f = 0.0
        step_in = 1.0 / time_factor

        while out_idx < out_len:
            # Find closest pitch mark in input
            in_mark_idx = int(round(in_idx_f))
            if in_mark_idx >= len(pitch_marks):
                break
            m = pitch_marks[in_mark_idx]

            # Extract window around mark m
            w_size = 2 * self.pitch_period
            w_start = max(0, m - self.pitch_period)
            w_end = min(len(signal), m + self.pitch_period)
            chunk = signal[w_start:w_end]

            if len(chunk) > 0:
                hanning_win = np.hanning(len(chunk))
                windowed_chunk = chunk * hanning_win

                # Overlap and add to output
                o_start = out_idx
                o_end = o_start + len(windowed_chunk)
                if o_end <= len(output):
                    output[o_start:o_end] += windowed_chunk

            out_idx += new_period
            in_idx_f += step_in

        # Normalize output amplitude
        max_val = np.max(np.abs(output))
        if max_val > 1e-4:
            output = output / max_val * 0.8
        return output[:out_len].astype(np.float32)

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Deterministic grapheme-to-phoneme map for response vocabulary.
        """
        mapping = {
            "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
            "b": "sil", "c": "sil", "d": "sil", "f": "sil", "g": "sil",
            "h": "sil", "j": "sil", "k": "sil", "l": "e", "m": "o",
            "n": "o", "p": "sil", "q": "sil", "r": "a", "s": "sil",
            "t": "sil", "v": "sil", "w": "u", "x": "sil", "y": "i", "z": "sil",
            " ": "sil"
        }
        res = []
        for ch in text.lower():
            p = mapping.get(ch, "sil")
            res.append(p)
        return res

    def synthesize(self, text: str, pitch_factor: float = 1.0, speed_factor: float = 1.0) -> np.ndarray:
        """
        Synthesizes speech audio waveform directly from text string.
        """
        phonemes = self.text_to_phonemes(text)
        segments = []
        for p in phonemes:
            seg = self.phone_inventory.get(p, self.phone_inventory["sil"])
            segments.append(seg)

        if not segments:
            return np.zeros(self.sample_rate // 4, dtype=np.float32)

        concat_signal = np.concatenate(segments)
        # Apply PSOLA modification
        return self.psola_resynthesize(concat_signal, pitch_factor=pitch_factor, time_factor=speed_factor)
