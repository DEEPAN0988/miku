"""
Test Suite: End-to-End Miku Sovereign Local Agent Pipeline.
Verifies multi-core execution:
Audio Voice Input -> Core 1 STT -> Core 2 CDSH Router -> Core 3 OS Action -> Persistence Log.
Vision Camera Detection -> Core 4 -> Core 2 Fusion.
"""
import unittest
import numpy as np
from pathlib import Path
import tempfile
import time

from miku.orchestrator import MikuOrchestrator
from miku.ipc.messages import STTTranscriptMsg

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.orchestrator = MikuOrchestrator()
        self.orchestrator.start()

    def tearDown(self):
        self.orchestrator.shutdown()

    def test_voice_command_to_action_and_audit_log(self):
        """
        Tests voice command flow through entire pipeline:
        1. Enroll voice phrase.
        2. Ingest audio through Core 1.
        3. Core 2 parses intent and computes CDSH score.
        4. Core 3 executes action (e.g. adjust volume).
        5. Encrypted persistence layer logs the audit record.
        """
        # Synthesize audio for command 'volume up'
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        voice_pcm = (0.4 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

        # Enroll phrase into GMM-HMM
        self.orchestrator.audio_daemon.stt.train_phrase("volume up", [voice_pcm])

        # Also enroll in calibration daemon to clear floor
        self.orchestrator.cognitive_daemon.calibration.state["min_viable_floor"] = 0.1
        self.orchestrator.cognitive_daemon.calibration.enroll_voice_phrase("volume up")

        # Mock OS volume call in test
        self.orchestrator.execution_daemon.os_exec.adjust_volume = lambda d: (True, f"Volume set {d}")

        decision = self.orchestrator.process_voice_input(voice_pcm)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["status"], "dispatched")
        self.assertEqual(decision["best_action"], "volume")

        # Check encrypted audit log
        logs = self.orchestrator.logger.read_logs(limit=1)
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]["action_type"], "volume")
        self.assertEqual(logs[0]["status"], "completed")

    def test_vision_to_cognitive_fusion(self):
        """
        Tests Core 4 camera detection feeding into Core 2 CDSH fusion.
        """
        # Red button frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[30:50, 40:60] = [255, 0, 0]

        detections = self.orchestrator.process_vision_frame(frame)
        self.assertGreater(len(detections), 0)
        self.assertEqual(detections[0].label, "red_region")

        # Core 2 now has latest vision detection
        self.assertIsNotNone(self.orchestrator.cognitive_daemon.latest_vision_detection)
        self.assertEqual(self.orchestrator.cognitive_daemon.latest_vision_detection.label, "red_region")

if __name__ == "__main__":
    unittest.main()
