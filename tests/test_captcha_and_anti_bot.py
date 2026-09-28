"""
Test Suite: Custom On-Device CAPTCHA Solver & Anti-Bot Stealth Subsystem.
Verifies custom text/slider/audio CAPTCHA solving and stealth profile generation.
"""
import unittest
import numpy as np
from multiprocessing import Queue

from miku.core3_execution.captcha_solver import CustomCaptchaSolver
from miku.core3_execution.anti_bot import AntiBotProfile
from miku.core3_execution.execution_daemon import ExecutionDaemon
from miku.ipc.messages import ActionRequestMsg

class TestCaptchaAndAntiBot(unittest.TestCase):
    def test_text_captcha_solving(self):
        """
        Verifies template matching text CAPTCHA solver.
        """
        solver = CustomCaptchaSolver()
        # Synthetic 32x100 white text on dark background
        img = np.zeros((32, 100), dtype=np.uint8)
        img[6:26, 10:25] = 255
        img[6:26, 35:50] = 255
        img[6:26, 60:75] = 255

        result = solver.solve_text_captcha(img)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_slider_puzzle_gap_detection(self):
        """
        Verifies edge detection and horizontal localization of slider puzzle notch.
        """
        solver = CustomCaptchaSolver()
        bg = np.full((100, 200), 120, dtype=np.uint8)
        # Create puzzle notch gap at x=110..150
        bg[30:70, 110:150] = 20

        notch_x = solver.solve_slider_puzzle_gap(bg, piece_width=40)
        self.assertGreaterEqual(notch_x, 50)
        self.assertLessEqual(notch_x, 180)

    def test_audio_captcha_digits(self):
        """
        Verifies spoken digits energy burst decoding.
        """
        solver = CustomCaptchaSolver()
        # Create 4 energy bursts (digits)
        sig = np.zeros(16000, dtype=np.float32)
        for i in range(4):
            start = i * 4000 + 500
            sig[start : start + 1500] = 0.5 * np.sin(np.linspace(0, 50, 1500))

        digits = solver.solve_audio_digits(sig)
        self.assertEqual(len(digits), 4)

    def test_anti_bot_stealth_profile(self):
        """
        Verifies CDP stealth injection payload and humanized inputs.
        """
        anti_bot = AntiBotProfile()
        cdp_payload = anti_bot.get_cdp_stealth_payload()
        self.assertEqual(cdp_payload["method"], "Page.addScriptToEvaluateOnNewDocument")
        self.assertIn("webdriver", cdp_payload["params"]["source"])

        # Humanized typing
        keystrokes = anti_bot.generate_humanized_keystrokes("test input!")
        self.assertEqual(len(keystrokes), len("test input!"))
        delays = [d for _, d in keystrokes]
        # Verify natural variation
        self.assertGreater(max(delays), min(delays))

    def test_execution_daemon_captcha_action(self):
        """
        Verifies ExecutionDaemon handling of solve_captcha and bypass_bot_check.
        """
        act_q = Queue()
        res_q = Queue()
        daemon = ExecutionDaemon(act_q, res_q)

        req1 = ActionRequestMsg(
            action_type="solve_captcha",
            target="captcha",
            params={"captcha_type": "text"},
            task_id="c1"
        )
        res1 = daemon.execute_request(req1)
        self.assertTrue(res1.success)
        self.assertIn("Solved text CAPTCHA", res1.message)

        req2 = ActionRequestMsg(
            action_type="bypass_bot_check",
            target="browser",
            task_id="c2"
        )
        res2 = daemon.execute_request(req2)
        self.assertTrue(res2.success)
        self.assertIn("stealth", res2.message)

if __name__ == "__main__":
    unittest.main()
