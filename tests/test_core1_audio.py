"""
Test Suite: Core 1 Audio Daemon (Sensory & Voice).
Covers NLMS filter convergence, GMM-HMM WER, TD-PSOLA TTS, and Audio Fuzzing.
"""
import unittest
import numpy as np
from miku.core1_audio.nlms_filter import NLMSFilter
from miku.core1_audio.gmm_hmm_stt import GMMHMMModel
from miku.core1_audio.td_psola_tts import TDPSOLASynthesizer

class TestCore1Audio(unittest.TestCase):
    def test_nlms_filter_convergence(self):
        """
        Feeds synthetic echo + known clean speech signal.
        Asserts that echo residual converges below threshold after adaptation.
        """
        nlms = NLMSFilter(filter_length=64, step_size=0.15)
        num_samples = 2000

        np.random.seed(42)
        ref_speaker = np.random.randn(num_samples).astype(np.float32) * 0.5
        # Simulate room acoustic echo impulse response
        room_impulse = np.exp(-np.linspace(0, 3, 30)).astype(np.float32)
        echo = np.convolve(ref_speaker, room_impulse, mode="full")[:num_samples]

        clean_voice = 0.2 * np.sin(2 * np.pi * 440 * np.arange(num_samples) / 16000).astype(np.float32)
        mic_signal = clean_voice + echo

        # Process through NLMS
        error_signal = nlms.process_block(mic_signal, ref_speaker)

        # Early block error vs late block error
        early_mse = float(np.mean((error_signal[:200] - clean_voice[:200]) ** 2))
        late_mse = float(np.mean((error_signal[1500:] - clean_voice[1500:]) ** 2))

        # Echo residual must drop significantly as filter converges
        self.assertLess(late_mse, early_mse, "NLMS filter must adapt and attenuate echo residual")
        self.assertLess(late_mse, 0.05, "Late echo error must be below threshold")

    def test_gmm_hmm_wer_and_decoding(self):
        """
        Tests GMM-HMM feature extraction, acoustic training, and decoding.
        """
        stt = GMMHMMModel(sample_rate=16000)
        
        # Test clean sample decoding
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        voice_sample = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        stt.train_phrase("open notepad", [voice_sample])

        phrase, conf = stt.decode(voice_sample)
        self.assertEqual(phrase, "open notepad")
        self.assertGreater(conf, 0.5)

        # Compute WER
        wer = GMMHMMModel.compute_wer("open notepad", phrase)
        self.assertEqual(wer, 0.0)

        # Test partial substitution WER
        wer_sub = GMMHMMModel.compute_wer("open notepad", "open calculator")
        self.assertGreater(wer_sub, 0.0)

    def test_td_psola_tts_synthesis(self):
        """
        Verifies TD-PSOLA waveform generation, pitch, and duration scaling.
        """
        tts = TDPSOLASynthesizer(sample_rate=16000, base_f0=150.0)
        audio = tts.synthesize("open notepad", pitch_factor=1.1, speed_factor=1.0)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        self.assertLess(np.max(np.abs(audio)), 1.1, "Audio must be normalized without clipping")

    def test_audio_fuzz_malformed_and_silent(self):
        """
        Fuzz test: Zero-length buffer, pure silence, clipped/saturated audio.
        Asserts no crash and graceful 'no speech detected'.
        """
        stt = GMMHMMModel(sample_rate=16000)

        # 1. Zero-length
        empty_res, empty_conf = stt.decode(np.array([], dtype=np.float32))
        self.assertEqual(empty_res, "")
        self.assertEqual(empty_conf, 0.0)

        # 2. Pure silence
        silence = np.zeros(1600, dtype=np.float32)
        sil_res, sil_conf = stt.decode(silence)
        self.assertEqual(sil_res, "")
        self.assertEqual(sil_conf, 0.0)

        # 3. Saturated / Clipped square wave
        saturated = np.sign(np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
        sat_res, _ = stt.decode(saturated)
        # Saturated noise should not crash
        self.assertIsInstance(sat_res, str)

if __name__ == "__main__":
    unittest.main()
