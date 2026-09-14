"""
eval/test_humanizer_curves.py — Unit tests for Miku's Win32 Humanizer Bézier curves and typing delays
"""

import unittest
from unittest.mock import patch, MagicMock, call
import time

from tools.screen_inspector import (
    human_mouse_move,
    get_current_cursor_pos,
    dispatch_real_click,
    dispatch_real_double_click,
    REAL_CLICK_ENABLED,
)
from tools.typing_automation import (
    dispatch_typing_payload,
    dispatch_human_keystrokes,
    REAL_TYPE_ENABLED,
)


class TestHumanizerCurves(unittest.TestCase):

    def test_get_current_cursor_pos(self):
        pos = get_current_cursor_pos()
        self.assertIsInstance(pos, tuple)
        self.assertEqual(len(pos), 2)
        self.assertIsInstance(pos[0], int)
        self.assertIsInstance(pos[1], int)

    @patch("tools.screen_inspector.user32.SetCursorPos")
    @patch("tools.screen_inspector.time.sleep")
    def test_human_mouse_move_trajectory(self, mock_sleep, mock_set_pos):
        human_mouse_move(0, 0, 500, 300, duration=0.20)
        # Should have called SetCursorPos multiple times along the Bézier curve
        self.assertGreater(mock_set_pos.call_count, 10)
        # Last call must be exactly the destination
        mock_set_pos.assert_called_with(500, 300)

    @patch("tools.typing_automation.user32.SendInput")
    @patch("tools.typing_automation.time.sleep")
    def test_dispatch_typing_payload_cadence(self, mock_sleep, mock_send_input):
        text = "Miku"
        count = dispatch_typing_payload(text, min_delay_sec=0.015, max_delay_sec=0.035)
        self.assertEqual(count, len(text))
        # Each character sends an input block
        self.assertEqual(mock_send_input.call_count, 4)
        # Micro-delay called after each character
        self.assertEqual(mock_sleep.call_count, 4)

    @patch("tools.screen_inspector.REAL_CLICK_ENABLED", False)
    def test_dispatch_real_double_click_circuit_breaker(self):
        res = dispatch_real_double_click(12345, (100, 100))
        self.assertFalse(res.success)
        self.assertEqual(res.status, "DRY_RUN_PENDING_CIRCUIT_BREAKER")
        self.assertFalse(res.real_input_dispatched)

    @patch("tools.screen_inspector.REAL_CLICK_ENABLED", True)
    @patch("tools.screen_inspector.verify_element_clickable")
    @patch("tools.screen_inspector.request_live_human_click_confirmation", return_value=True)
    @patch("tools.screen_inspector.user32.GetForegroundWindow", return_value=12345)
    @patch("tools.screen_inspector.user32.GetAncestor", return_value=12345)
    @patch("tools.screen_inspector.user32.mouse_event")
    @patch("tools.screen_inspector.human_mouse_move")
    def test_dispatch_real_double_click_success(
        self, mock_move, mock_mouse_event, mock_anc, mock_fg, mock_confirm, mock_verify
    ):
        mock_verif = MagicMock()
        mock_verif.is_safe = True
        mock_verif.details = {"window_text": "Target Window"}
        mock_verify.return_value = mock_verif

        res = dispatch_real_double_click(12345, (250, 350))
        self.assertTrue(res.success)
        self.assertEqual(res.status, "DOUBLE_CLICK_SUCCESS")
        self.assertTrue(res.real_input_dispatched)
        # Mouse moved via Bézier glide
        mock_move.assert_called_once()
        # Double click requires 4 mouse_event calls (down, up, down, up)
        self.assertEqual(mock_mouse_event.call_count, 4)


if __name__ == "__main__":
    unittest.main()
