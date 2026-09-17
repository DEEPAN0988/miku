import unittest
import sys
import os
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wakeword.audio_features import extract_mfcc
from wakeword.engine import WakeWordEngine


class TestWakeWord(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_ww.pth")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mfcc_extraction(self):
        signal = np.random.normal(0, 0.1, 16000)
        mfcc = extract_mfcc(signal, sample_rate=16000, target_frames=99)
        self.assertEqual(mfcc.shape, (99, 13))

    def test_wakeword_detection_flow(self):
        engine = WakeWordEngine(model_path=self.model_path)
        
        # Verify synthetic sample generation
        X, y = engine.generate_synthetic_samples(num_samples=10)
        self.assertEqual(len(X), 10)
        self.assertEqual(len(y), 10)

        # Detection inference
        sample_signal = np.random.normal(0, 0.1, 16000)
        detected, score = engine.detect(sample_signal)
        self.assertIsInstance(detected, bool)
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
