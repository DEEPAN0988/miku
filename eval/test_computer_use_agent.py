"""
eval/test_computer_use_agent.py — Test Suite for Generalized Multi-Step Computer Use Agent

Constitutional Verification:
  1. Keystroke Mapping: Verifies mapping of common string keys to Win32 Virtual-Key Codes.
  2. Action Schema Dispatch: Verifies execution of click, double_click, type, press_key, wait, terminate.
  3. Zero Safety Regressions: Verifies verify_element_clickable intercepts spatial clicks.
  4. Circuit Breakers: Verifies MIKU_LIVE_EXECUTION and MIKU_AUTONOMOUS_MODE enforcement and approval gate.
  5. Safeguard: Verifies MAX_STEPS = 15 halts infinite execution.
  6. Multi-Step Loop: Verifies full execution loop accumulating history and terminating on objective completion.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.computer_use_agent import (
    MAX_STEPS,
    KEY_MAPPING,
    resolve_virtual_key,
    is_autonomous_authorized,
    request_step_approval,
    dispatch_computer_use_action,
    run_computer_use_task,
    ComputerUseStepResult,
    ComputerUseTaskResult,
)
from tools.screen_inspector import ClickVerificationResult
from tools.orchestrator import AstraVisionClient


class TestKeystrokeMapping(unittest.TestCase):
    """Verifies string key resolution to Win32 Virtual-Key Codes."""

    def test_common_named_keys(self):
        self.assertEqual(resolve_virtual_key("enter"), 0x0D)
        self.assertEqual(resolve_virtual_key("return"), 0x0D)
        self.assertEqual(resolve_virtual_key("tab"), 0x09)
        self.assertEqual(resolve_virtual_key("esc"), 0x1B)
        self.assertEqual(resolve_virtual_key("escape"), 0x1B)
        self.assertEqual(resolve_virtual_key("win"), 0x5B)
        self.assertEqual(resolve_virtual_key("windows"), 0x5B)
        self.assertEqual(resolve_virtual_key("space"), 0x20)
        self.assertEqual(resolve_virtual_key("backspace"), 0x08)
        self.assertEqual(resolve_virtual_key("delete"), 0x2E)
        self.assertEqual(resolve_virtual_key("ctrl"), 0x11)
        self.assertEqual(resolve_virtual_key("shift"), 0x10)
        self.assertEqual(resolve_virtual_key("alt"), 0x12)
        self.assertEqual(resolve_virtual_key("up"), 0x26)
        self.assertEqual(resolve_virtual_key("down"), 0x28)
        self.assertEqual(resolve_virtual_key("left"), 0x25)
        self.assertEqual(resolve_virtual_key("right"), 0x27)
        self.assertEqual(resolve_virtual_key("f1"), 0x70)
        self.assertEqual(resolve_virtual_key("f5"), 0x74)
        self.assertEqual(resolve_virtual_key("f12"), 0x7B)

    def test_alphanumeric_keys(self):
        self.assertEqual(resolve_virtual_key("a"), ord("A"))
        self.assertEqual(resolve_virtual_key("Z"), ord("Z"))
        self.assertEqual(resolve_virtual_key("1"), ord("1"))
        self.assertEqual(resolve_virtual_key("9"), ord("9"))

    def test_invalid_keys(self):
        self.assertIsNone(resolve_virtual_key("nonexistent_unknown_key_xyz"))
        self.assertIsNone(resolve_virtual_key(None))


class TestActionDispatcherAndSafety(unittest.TestCase):
    """Verifies Win32 action dispatching and verify_element_clickable interception."""

    @patch("tools.computer_use_agent.verify_element_clickable")
    def test_click_safety_gate_rejection(self, mock_verify):
        mock_verify.return_value = ClickVerificationResult(
            is_safe=False,
            reason="OCCLUDED_AT_POINT",
            details={"error": "Point occluded by banner"},
        )
        action = {"action": "click", "x": 300, "y": 400}
        res = dispatch_computer_use_action(action, real_execution=False, target_hwnd=123)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "ABORT_OCCLUDED_AT_POINT")
        self.assertIn("LOCAL SAFETY GATE REJECTION", res.output)
        mock_verify.assert_called_once_with(123, (300, 400))

    @patch("tools.computer_use_agent.verify_element_clickable")
    @patch("tools.computer_use_agent.simulate_click")
    def test_click_safety_gate_allowed(self, mock_sim, mock_verify):
        mock_verify.return_value = ClickVerificationResult(is_safe=True, reason="SAFE")
        mock_sim.return_value = MagicMock(success=True, status="SUCCESS", action_log="Click passed")

        action = {"action": "click", "x": 300, "y": 400}
        res = dispatch_computer_use_action(action, real_execution=False, target_hwnd=123)
        self.assertTrue(res.success)
        self.assertIn("SIMULATED CLICK", res.output)
        mock_verify.assert_called_once_with(123, (300, 400))

    @patch("tools.computer_use_agent.verify_element_clickable")
    def test_double_click_safety_gate(self, mock_verify):
        mock_verify.return_value = ClickVerificationResult(
            is_safe=False,
            reason="NOT_FOREGROUND",
            details={"error": "Target window not foreground"},
        )
        action = {"action": "double_click", "x": 500, "y": 200}
        res = dispatch_computer_use_action(action, real_execution=False, target_hwnd=456)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "ABORT_NOT_FOREGROUND")
        mock_verify.assert_called_once_with(456, (500, 200))

    def test_type_payload_sanitization(self):
        # Valid typing
        action = {"action": "type", "text": "hello computer"}
        res = dispatch_computer_use_action(action, real_execution=False)
        self.assertTrue(res.success)
        self.assertIn("SIMULATED TYPING", res.output)

        # Forbidden control character (\x00 NULL)
        action_bad = {"action": "type", "text": "bad\x00input"}
        res_bad = dispatch_computer_use_action(action_bad, real_execution=False)
        self.assertFalse(res_bad.success)
        self.assertIn("ABORT_PAYLOAD", res_bad.status)

    def test_press_key_dispatch(self):
        action = {"action": "press_key", "key": "win"}
        res = dispatch_computer_use_action(action, real_execution=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "SIMULATED_KEY_SUCCESS")
        self.assertEqual(res.details["vk_code"], 0x5B)

    def test_wait_dispatch(self):
        action = {"action": "wait", "seconds": 0.01}
        res = dispatch_computer_use_action(action, real_execution=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "WAIT_COMPLETED")

    def test_terminate_dispatch(self):
        action = {"action": "terminate", "reason": "Completed"}
        res = dispatch_computer_use_action(action, real_execution=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "TERMINATED")


class TestCircuitBreakersAndApproval(unittest.TestCase):
    """Verifies circuit breaker enforcement and step approval gating."""

    def test_circuit_breaker_disabled_by_default(self):
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"
        self.assertFalse(is_autonomous_authorized())

    def test_circuit_breaker_enabled_only_when_both_flags_true(self):
        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "true"
        self.assertTrue(is_autonomous_authorized())

        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"
        self.assertFalse(is_autonomous_authorized())

        # Cleanup
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

    def test_step_approval_via_runner(self):
        action = {"action": "press_key", "key": "enter"}
        # User approves
        self.assertTrue(request_step_approval(1, action, prompt_runner=lambda p: True))
        # User denies
        self.assertFalse(request_step_approval(1, action, prompt_runner=lambda p: False))

    def test_non_interactive_approval_fails_closed(self):
        action = {"action": "click", "x": 10, "y": 10}
        with patch("sys.stdin.isatty", return_value=False):
            self.assertFalse(request_step_approval(1, action, prompt_runner=None))


class TestComputerUseExecutionLoop(unittest.TestCase):
    """Verifies the multi-step execution loop, history recording, safeguards, and termination."""

    def test_max_steps_safeguard(self):
        """Loop must terminate at max_steps if model never outputs terminate."""
        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

        def mock_capturer(hwnd):
            return ("data:image/png;base64,SCREEN", (0, 0, 1920, 1080))

        # Model keeps outputting wait action endlessly
        def mock_api_runner(payload):
            return '{"action": "wait", "seconds": 0.01}'

        client = AstraVisionClient(api_runner=mock_api_runner)

        res = run_computer_use_task(
            objective="Wait forever",
            max_steps=3,
            client=client,
            real_execution=False,
            screenshot_capturer=mock_capturer,
            delay_between_steps=0,
        )

        self.assertFalse(res.success)
        self.assertEqual(res.final_status, "MAX_STEPS_EXCEEDED")
        self.assertEqual(res.total_steps, 3)
        self.assertIn("Exceeded maximum allowable steps", res.error)

    def test_user_rejection_halts_loop(self):
        """When autonomous mode is off and user rejects, loop terminates safely."""
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

        def mock_capturer(hwnd):
            return ("data:image/png;base64,SCREEN", (0, 0, 1920, 1080))

        def mock_api_runner(payload):
            return '{"action": "type", "text": "calc"}'

        client = AstraVisionClient(api_runner=mock_api_runner)

        # User denies step approval
        res = run_computer_use_task(
            objective="Launch Calc",
            client=client,
            real_execution=False,
            screenshot_capturer=mock_capturer,
            prompt_runner=lambda prompt: False,
            delay_between_steps=0,
        )

        self.assertFalse(res.success)
        self.assertEqual(res.final_status, "USER_REJECTED")
        self.assertEqual(res.total_steps, 1)

    def test_multi_step_nominal_flow_with_history(self):
        """Executes a 4-step sequence (press_key -> type -> press_key -> terminate) and verifies history."""
        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

        def mock_capturer(hwnd):
            return ("data:image/png;base64,SCREEN", (0, 0, 1920, 1080))

        step_counter = 0
        received_histories = []

        def mock_api_runner(payload):
            nonlocal step_counter
            step_counter += 1
            # Check user content history
            user_text = payload["messages"][1]["content"][0]["text"]
            received_histories.append(user_text)

            if step_counter == 1:
                return '{"action": "press_key", "key": "win"}'
            elif step_counter == 2:
                return '{"action": "type", "text": "notepad"}'
            elif step_counter == 3:
                return '{"action": "press_key", "key": "enter"}'
            else:
                return '{"action": "terminate", "reason": "Notepad opened"}'

        client = AstraVisionClient(api_runner=mock_api_runner)

        res = run_computer_use_task(
            objective="Open Notepad via Start Menu",
            max_steps=MAX_STEPS,
            client=client,
            real_execution=False,
            screenshot_capturer=mock_capturer,
            delay_between_steps=0,
        )

        self.assertTrue(res.success)
        self.assertEqual(res.final_status, "TERMINATED")
        self.assertEqual(res.total_steps, 4)
        self.assertEqual(len(res.steps), 4)
        self.assertEqual(len(res.action_history), 4)

        # Verify step actions
        self.assertEqual(res.steps[0].action["action"], "press_key")
        self.assertEqual(res.steps[0].action["key"], "win")
        self.assertEqual(res.steps[1].action["action"], "type")
        self.assertEqual(res.steps[1].action["text"], "notepad")
        self.assertEqual(res.steps[2].action["action"], "press_key")
        self.assertEqual(res.steps[2].action["key"], "enter")
        self.assertEqual(res.steps[3].action["action"], "terminate")

        # Verify history was accumulated and passed to model
        self.assertIn("History of previously executed actions:", received_histories[1])
        self.assertIn("press_key", received_histories[1])
        self.assertIn("notepad", received_histories[2])

        # Test dictionary serialization
        task_dict = res.to_dict()
        self.assertTrue(task_dict["success"])
        self.assertEqual(task_dict["total_steps"], 4)

        # Reset env
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"


if __name__ == "__main__":
    unittest.main()
