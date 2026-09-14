"""
eval/test_human_fallback_launcher.py — Test Suite for Human Fallback Launcher

Validates:
1. Payload sanitization & rejection of illegal control characters.
2. Safety circuit breaker gating (fails closed / operates in simulated mode when flags are missing).
3. Keyboard launch sequence: VK_LWIN -> Unicode SendInput -> VK_RETURN.
4. Vision & cursor fallback sequence: capture_window_base64 -> gpt-6-astra query -> dispatch_real_click.
5. Error handling across capture failures and schema violations.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.human_fallback_launcher import (
    human_launch_app,
    dispatch_unicode_text_to_foreground,
    HumanLaunchResult,
    VK_LWIN,
    VK_RETURN,
)
from tools.orchestrator import AstraVisionClient
from tools.screen_inspector import ClickDispatchResult


class TestPayloadSanitization(unittest.TestCase):
    """Verifies target name sanitization and forbidden control char rejection."""

    def test_reject_empty_payload(self):
        res = human_launch_app("")
        self.assertFalse(res.success)
        self.assertIn("REJECT_EMPTY_TEXT", res.status)

    def test_reject_forbidden_control_characters(self):
        res = human_launch_app("Wuthering\x00Waves")
        self.assertFalse(res.success)
        self.assertIn("REJECT_FORBIDDEN_CONTROL_CHAR", res.status)

        res2 = human_launch_app("Wuthering\x1BWaves")
        self.assertFalse(res2.success)
        self.assertIn("REJECT_FORBIDDEN_CONTROL_CHAR", res2.status)


class TestCircuitBreakerEnforcement(unittest.TestCase):
    """Verifies physical input is blocked when circuit breakers are disabled."""

    def setUp(self):
        self.orig_live = os.environ.get("MIKU_LIVE_EXECUTION")
        self.orig_auto = os.environ.get("MIKU_AUTONOMOUS_MODE")

    def tearDown(self):
        if self.orig_live is not None:
            os.environ["MIKU_LIVE_EXECUTION"] = self.orig_live
        else:
            os.environ.pop("MIKU_LIVE_EXECUTION", None)

        if self.orig_auto is not None:
            os.environ["MIKU_AUTONOMOUS_MODE"] = self.orig_auto
        else:
            os.environ.pop("MIKU_AUTONOMOUS_MODE", None)

    @patch("ctypes.windll.user32.keybd_event")
    @patch("ctypes.windll.user32.SendInput")
    def test_unauthorized_runs_in_simulated_mode(self, mock_send_input, mock_keybd_event):
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

        res = human_launch_app("Calculator", use_mouse=False, sleep_fn=lambda s: None)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "SIMULATED_KEYBOARD_LAUNCH_READY")
        self.assertEqual(res.mode, "DRY_RUN_SIMULATED")
        self.assertIn("SIMULATED_VK_LWIN", res.steps_executed)
        self.assertIn("SIMULATED_TYPE:Calculator", res.steps_executed)

        # Ensure NO physical hardware inputs were dispatched
        mock_keybd_event.assert_not_called()
        mock_send_input.assert_not_called()

    @patch("ctypes.windll.user32.keybd_event")
    @patch("ctypes.windll.user32.SendInput")
    def test_authorized_dispatches_real_hardware_events(self, mock_send_input, mock_keybd_event):
        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

        res = human_launch_app("Calculator", use_mouse=False, sleep_fn=lambda s: None)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "SUCCESS_KEYBOARD_LAUNCH")
        self.assertEqual(res.mode, "KEYBOARD_ENTER")

        # Verify VK_LWIN was pressed & released, and VK_RETURN was pressed & released
        calls = mock_keybd_event.call_args_list
        # Call 1: VK_LWIN down, Call 2: VK_LWIN up, Call 3: VK_RETURN down, Call 4: VK_RETURN up
        self.assertGreaterEqual(len(calls), 4)
        self.assertEqual(calls[0][0][0], VK_LWIN)
        self.assertEqual(calls[0][0][2], 0)  # down
        self.assertEqual(calls[1][0][0], VK_LWIN)
        self.assertEqual(calls[1][0][2], 2)  # up
        self.assertEqual(calls[2][0][0], VK_RETURN)
        self.assertEqual(calls[2][0][2], 0)  # down
        self.assertEqual(calls[3][0][0], VK_RETURN)
        self.assertEqual(calls[3][0][2], 2)  # up

        # Verify SendInput was called for typing
        self.assertEqual(mock_send_input.call_count, len("Calculator"))


class TestVisionAndCursorFallback(unittest.TestCase):
    """Verifies screen capture, gpt-6-astra coordinate reasoning, and dispatch_real_click."""

    def setUp(self):
        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

    def tearDown(self):
        os.environ.pop("MIKU_LIVE_EXECUTION", None)
        os.environ.pop("MIKU_AUTONOMOUS_MODE", None)

    @patch("ctypes.windll.user32.keybd_event")
    @patch("tools.human_fallback_launcher.dispatch_unicode_text_to_foreground")
    @patch("tools.human_fallback_launcher.dispatch_real_click")
    def test_vision_mouse_launch_success(self, mock_click, mock_type, mock_keybd):
        def mock_capturer(hwnd):
            return ("data:image/png;base64,TEST_DESKTOP", (0, 0, 1920, 1080))

        def mock_api_runner(payload):
            return '{"action": "click", "x": 450, "y": 280}'

        mock_click.return_value = ClickDispatchResult(
            success=True,
            status="REAL_CLICK_SUCCESS",
            verification=None,
            action_log="[PHYSICAL CLICK DISPATCHED] Clicked at (450, 280)",
            target_hwnd=65552,
            element_name="search_item",
            coordinate=(450, 280),
            button="left",
            real_input_dispatched=True,
        )

        client = AstraVisionClient(api_runner=mock_api_runner)
        res = human_launch_app(
            target_name="Wuthering Waves",
            use_mouse=True,
            client=client,
            capturer=mock_capturer,
            sleep_fn=lambda s: None,
        )

        self.assertTrue(res.success)
        self.assertEqual(res.status, "SUCCESS_VISION_LAUNCH")
        self.assertEqual(res.mode, "VISION_CURSOR")
        self.assertIn("DISPATCH_REAL_CLICK_AT:(450,280)", res.steps_executed)
        mock_click.assert_called_once()
        args, kwargs = mock_click.call_args
        self.assertEqual(args[1], (450, 280))

    def test_vision_mouse_launch_simulated_when_unauthorized(self):
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

        res = human_launch_app("Wuthering Waves", use_mouse=True, sleep_fn=lambda s: None)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "SIMULATED_VISION_CLICK_READY")
        self.assertEqual(res.mode, "DRY_RUN_SIMULATED")
        self.assertIn("VISION_GROUNDING_CURSOR_FALLBACK", res.steps_executed)

    @patch("ctypes.windll.user32.keybd_event")
    @patch("tools.human_fallback_launcher.dispatch_unicode_text_to_foreground")
    def test_vision_mouse_capture_failure(self, mock_type, mock_keybd):
        res = human_launch_app(
            target_name="Wuthering Waves",
            use_mouse=True,
            capturer=lambda h: None,
            sleep_fn=lambda s: None,
        )
        self.assertFalse(res.success)
        self.assertEqual(res.status, "ABORT_CAPTURE_FAILED")


if __name__ == "__main__":
    unittest.main()
