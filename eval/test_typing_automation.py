"""
eval/test_typing_automation.py — Comprehensive Test Suite for GUI Edit Control Typing Automation

Tests:
  1. Circuit Breaker Invariant: REAL_TYPE_ENABLED is False by default; blocks physical dispatch.
  2. Payload Sanitizer: Length, empty, control characters, safe multiline.
  3. Pre-Typing Editability Verification:
     - Rejects non-editable controls (Button, Pane, Text, CheckBox).
     - Rejects disabled controls.
     - Rejects invalid HWNDs.
  4. Simulated Typing Dry-Run:
     - Guarantees real_input_dispatched is False under all scenarios.
     - Verifies correct simulation logging and status reporting.
  5. Live Confirmation Gate:
     - Fails closed in non-interactive mode.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from tools.screen_inspector import UIElement
from tools.typing_automation import (
    REAL_TYPE_ENABLED,
    MAX_STAGED_TEXT_LENGTH,
    VALID_EDIT_CONTROL_TYPES,
    sanitize_typing_payload,
    verify_element_editable,
    simulate_typing,
    dispatch_real_typing,
    request_live_human_type_confirmation,
)


class TestTypingAutomation(unittest.TestCase):

    def test_circuit_breaker_default_false(self):
        """Constitutional Safety Invariant: REAL_TYPE_ENABLED must be False by default."""
        self.assertFalse(
            REAL_TYPE_ENABLED,
            "CRITICAL SAFETY VIOLATION: REAL_TYPE_ENABLED must be False by default."
        )

    def test_dispatch_blocked_by_circuit_breaker(self):
        """Real typing dispatch must fail closed when REAL_TYPE_ENABLED is False."""
        el = UIElement(
            control_type="Edit",
            name="SearchBox",
            automation_id="txtSearch",
            rect=(100, 100, 300, 130),
            center=(200, 115),
            is_enabled=True,
        )
        res = dispatch_real_typing(12345, el, "test string")
        self.assertFalse(res.success)
        self.assertEqual(res.status, "DRY_RUN_PENDING_CIRCUIT_BREAKER")
        self.assertFalse(res.real_input_dispatched)
        self.assertEqual(res.characters_typed, 0)

    def test_payload_sanitizer(self):
        """Tests all sanitization paths: empty, overflow, control characters, and valid text."""
        # 1. Empty text
        ok, text, reason = sanitize_typing_payload("")
        self.assertFalse(ok)
        self.assertEqual(reason, "REJECT_EMPTY_TEXT")

        # 2. Non-string
        ok, text, reason = sanitize_typing_payload(None)  # type: ignore
        self.assertFalse(ok)
        self.assertEqual(reason, "REJECT_NOT_STRING")

        # 3. Exceeds max length -> Gracefully truncated
        long_text = "a" * (MAX_STAGED_TEXT_LENGTH + 50)
        ok, text, reason = sanitize_typing_payload(long_text)
        self.assertTrue(ok)
        self.assertEqual(len(text), MAX_STAGED_TEXT_LENGTH)
        self.assertEqual(reason, "PAYLOAD_TRUNCATED")

        # 4. Control characters (NULL byte, Escape, Bell)
        ok, text, reason = sanitize_typing_payload("hello\x00world")
        self.assertFalse(ok)
        self.assertEqual(reason, "REJECT_FORBIDDEN_CONTROL_CHAR_0x00")

        ok, text, reason = sanitize_typing_payload("command\x1b[2J")
        self.assertFalse(ok)
        self.assertEqual(reason, "REJECT_FORBIDDEN_CONTROL_CHAR_0x1B")

        # 5. Valid text with safe whitespace (\n, \t) and unicode
        ok, text, reason = sanitize_typing_payload("Hello, Miku!\nLine 2\tTabbed: \u2713")
        self.assertTrue(ok)
        self.assertEqual(reason, "SAFE")
        self.assertEqual(text, "Hello, Miku!\nLine 2\tTabbed: \u2713")

    def test_editability_control_type_rejection(self):
        """Non-editable controls (Button, Pane, Text, etc.) must be rejected."""
        btn = UIElement(
            control_type="Button",
            name="SubmitButton",
            automation_id="btn_submit",
            rect=(50, 50, 150, 90),
            center=(100, 70),
            is_enabled=True,
        )
        verif = verify_element_editable(0, btn)
        # Invalid HWND or NOT_EDITABLE_TYPE
        self.assertFalse(verif.is_safe)
        self.assertIn(verif.reason, ("INVALID_HWND", "NOT_EDITABLE_TYPE"))

    def test_disabled_element_rejection(self):
        """Disabled elements must fail editable verification."""
        # Using desktop window or dummy HWND
        edit_disabled = UIElement(
            control_type="Edit",
            name="ReadOnlyInput",
            automation_id="txtDisabled",
            rect=(50, 50, 250, 80),
            center=(150, 65),
            is_enabled=False,  # Disabled
        )
        verif = verify_element_editable(0, edit_disabled)
        self.assertFalse(verif.is_safe)

    def test_simulate_typing_rejections(self):
        """Simulated typing correctly reports payload and element rejections."""
        btn = UIElement(
            control_type="Button",
            name="CancelButton",
            automation_id="btn_cancel",
            rect=(50, 50, 150, 90),
            center=(100, 70),
            is_enabled=True,
        )
        # Test bad payload
        res_payload = simulate_typing(12345, btn, "bad\x00text")
        self.assertFalse(res_payload.success)
        self.assertEqual(res_payload.status, "ABORT_PAYLOAD_REJECT_FORBIDDEN_CONTROL_CHAR_0x00")
        self.assertFalse(res_payload.real_input_dispatched)

        # Test non-editable element with valid payload
        res_el = simulate_typing(12345, btn, "valid text")
        self.assertFalse(res_el.success)
        self.assertTrue(res_el.status.startswith("ABORT_"))
        self.assertFalse(res_el.real_input_dispatched)

    def test_human_confirmation_fails_closed_in_ci(self):
        """Live human confirmation must fail closed when stdin is not a TTY."""
        old_live = os.environ.pop("MIKU_LIVE_EXECUTION", None)
        old_auto = os.environ.pop("MIKU_AUTONOMOUS_MODE", None)
        old_appr = os.environ.pop("MIKU_AUTO_APPROVE", None)
        old_pass = os.environ.pop("MIKU_FAST_CONFIRM_PASSTHROUGH", None)
        try:
            confirmed = request_live_human_type_confirmation("Notepad", "Document", "Hello World")
            self.assertFalse(confirmed)
        finally:
            if old_live: os.environ["MIKU_LIVE_EXECUTION"] = old_live
            if old_auto: os.environ["MIKU_AUTONOMOUS_MODE"] = old_auto
            if old_appr: os.environ["MIKU_AUTO_APPROVE"] = old_appr
            if old_pass: os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = old_pass


if __name__ == "__main__":
    unittest.main()
