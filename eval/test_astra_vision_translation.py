"""
eval/test_astra_vision_translation.py — Test Suite for Astra Vision Client & Action Translation Layer

Constitutional Verification:
  1. No Fake AI / Evidence-based: Tests client payload formatting for multimodal models.
  2. Triple Circuit Breaker: Verifies physical input is blocked when REAL_CLICK_ENABLED=False.
  3. Safety Gate Interception: Verifies verify_element_clickable intercepts invalid HWND / out-of-bounds
     coordinates before any OS interaction.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.orchestrator import (
    AstraVisionClient,
    parse_astra_action,
    dispatch_astra_ui_action,
    execute_with_astra_vision,
)
from tools.screen_inspector import (
    ClickVerificationResult,
    SimulatedClickResult,
    REAL_CLICK_ENABLED,
)
from tools.vision_grounder import capture_window_base64


class TestAstraVisionClient(unittest.TestCase):
    """Verifies Astra multimodal API client initialization and message payload assembly."""

    def test_client_defaults(self):
        client = AstraVisionClient()
        self.assertEqual(client.model, "openai/gpt-6-astra")
        self.assertIn("api.openai.com", client.base_url)

    def test_client_custom_config(self):
        client = AstraVisionClient(
            model="custom/multimodal-v1",
            api_key="test_key_123",
            base_url="https://custom.endpoint/v1",
        )
        self.assertEqual(client.model, "custom/multimodal-v1")
        self.assertEqual(client.api_key, "test_key_123")
        self.assertEqual(client.base_url, "https://custom.endpoint/v1")

    def test_build_vision_payload(self):
        client = AstraVisionClient(model="openai/gpt-6-astra")
        dummy_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        query = "Click on the Submit button"

        payload = client.build_vision_payload(query=query, base64_image_url=dummy_b64)

        self.assertEqual(payload["model"], "openai/gpt-6-astra")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

        messages = payload["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[0]["text"], query)
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(user_content[1]["image_url"]["url"], dummy_b64)

    def test_query_action_via_mock_runner(self):
        recorded_payloads = []

        def mock_runner(payload):
            recorded_payloads.append(payload)
            return '{"action": "click", "x": 527, "y": 349}'

        client = AstraVisionClient(api_runner=mock_runner)
        dummy_b64 = "data:image/png;base64,TEST"
        resp = client.query_action("Click submit", dummy_b64)

        self.assertEqual(resp, '{"action": "click", "x": 527, "y": 349}')
        self.assertEqual(len(recorded_payloads), 1)
        self.assertEqual(recorded_payloads[0]["model"], "openai/gpt-6-astra")


class TestActionTranslationLayer(unittest.TestCase):
    """Verifies structured JSON action parsing and safety-intercepted dispatch."""

    def test_parse_clean_json(self):
        raw = '{"action": "click", "x": 527, "y": 349, "button": "left"}'
        parsed = parse_astra_action(raw)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["action"], "click")
        self.assertEqual(parsed["x"], 527)
        self.assertEqual(parsed["y"], 349)
        self.assertEqual(parsed["button"], "left")

    def test_parse_markdown_code_block(self):
        raw = """```json
{
  "action": "click",
  "x": 1024,
  "y": 768
}
```"""
        parsed = parse_astra_action(raw)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["action"], "click")
        self.assertEqual(parsed["x"], 1024)
        self.assertEqual(parsed["y"], 768)

    def test_parse_type_action(self):
        raw = '{"action": "type", "text": "hello miku", "x": 200, "y": 150}'
        parsed = parse_astra_action(raw)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["action"], "type")
        self.assertEqual(parsed["text"], "hello miku")
        self.assertEqual(parsed["x"], 200)
        self.assertEqual(parsed["y"], 150)

    def test_build_vision_payload_with_history(self):
        client = AstraVisionClient(model="openai/gpt-6-astra")
        dummy_b64 = "data:image/png;base64,TEST"
        history = [
            {"action": {"action": "click", "x": 100, "y": 200}, "success": True, "status": "SUCCESS"},
            {"action": {"action": "press_key", "key": "enter"}, "success": True, "status": "SUCCESS"},
        ]
        payload = client.build_vision_payload(
            query="Open Settings",
            base64_image_url=dummy_b64,
            history=history,
        )
        user_content = payload["messages"][1]["content"]
        self.assertIn("History of previously executed actions:", user_content[0]["text"])
        self.assertIn("click", user_content[0]["text"])
        self.assertIn("press_key", user_content[0]["text"])

    def test_parse_expanded_actions(self):
        # double_click
        parsed_dc = parse_astra_action('{"action": "double_click", "x": 400, "y": 300}')
        self.assertTrue(parsed_dc["valid"])
        self.assertEqual(parsed_dc["action"], "double_click")
        self.assertEqual(parsed_dc["x"], 400)
        self.assertEqual(parsed_dc["y"], 300)

        # press_key
        parsed_pk = parse_astra_action('{"action": "press_key", "key": "enter"}')
        self.assertTrue(parsed_pk["valid"])
        self.assertEqual(parsed_pk["action"], "press_key")
        self.assertEqual(parsed_pk["key"], "enter")

        # wait
        parsed_w = parse_astra_action('{"action": "wait", "seconds": 2.5}')
        self.assertTrue(parsed_w["valid"])
        self.assertEqual(parsed_w["action"], "wait")
        self.assertEqual(parsed_w["seconds"], 2.5)

        # terminate
        parsed_term = parse_astra_action('{"action": "terminate", "reason": "Objective reached"}')
        self.assertTrue(parsed_term["valid"])
        self.assertEqual(parsed_term["action"], "terminate")
        self.assertEqual(parsed_term["reason"], "Objective reached")

    def test_parse_invalid_payloads(self):
        # Missing action
        self.assertFalse(parse_astra_action('{"x": 10, "y": 20}')["valid"])
        # Unsupported verb
        self.assertFalse(parse_astra_action('{"action": "destroy", "x": 10, "y": 20}')["valid"])
        # Non-integer coordinates
        self.assertFalse(parse_astra_action('{"action": "click", "x": "top", "y": 10}')["valid"])
        # Malformed text
        self.assertFalse(parse_astra_action('not a json object')["valid"])
        # Invalid press_key (missing/empty key)
        self.assertFalse(parse_astra_action('{"action": "press_key", "key": ""}')["valid"])
        # Negative wait
        self.assertFalse(parse_astra_action('{"action": "wait", "seconds": -5}')["valid"])

    def test_dispatch_non_spatial_actions(self):
        # press_key dispatch
        res_pk = dispatch_astra_ui_action(
            target_hwnd=0,
            action_dict={"valid": True, "action": "press_key", "key": "win"},
            real_execution=False,
        )
        self.assertTrue(res_pk["success"])
        self.assertEqual(res_pk["status"], "PRESS_KEY_DISPATCHED")

        # wait dispatch
        res_w = dispatch_astra_ui_action(
            target_hwnd=0,
            action_dict={"valid": True, "action": "wait", "seconds": 0.01},
            real_execution=False,
        )
        self.assertTrue(res_w["success"])
        self.assertEqual(res_w["status"], "WAIT_COMPLETED")

        # terminate dispatch
        res_term = dispatch_astra_ui_action(
            target_hwnd=0,
            action_dict={"valid": True, "action": "terminate", "reason": "Goal done"},
            real_execution=False,
        )
        self.assertTrue(res_term["success"])
        self.assertEqual(res_term["status"], "TERMINATED")

    @patch("tools.orchestrator.verify_element_clickable")
    def test_safety_gate_intercepts_unclickable_element(self, mock_verify):
        """Astra coordinates MUST be intercepted by verify_element_clickable before click."""
        mock_verify.return_value = ClickVerificationResult(
            is_safe=False,
            reason="OCCLUDED_AT_POINT",
            details="Point is occluded by modal overlay",
        )

        action = {"valid": True, "action": "click", "x": 500, "y": 300, "button": "left"}
        res = dispatch_astra_ui_action(target_hwnd=12345, action_dict=action, real_execution=False)

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "ABORT_OCCLUDED_AT_POINT")
        self.assertIn("LOCAL SAFETY GATE REJECTION", res["output"])
        mock_verify.assert_called_once_with(12345, (500, 300))

    @patch("tools.orchestrator.verify_element_clickable")
    @patch("tools.orchestrator.simulate_click")
    def test_safety_gate_allows_safe_coordinate(self, mock_simulate, mock_verify):
        """When verify_element_clickable reports SAFE, routes to simulate_click."""
        mock_verify.return_value = ClickVerificationResult(
            is_safe=True,
            reason="SAFE",
            details="Window is foreground and point is unoccluded",
        )
        mock_simulate.return_value = SimulatedClickResult(
            success=True,
            status="SUCCESS",
            verification=mock_verify.return_value,
            action_log="[SIMULATED CLICK SUCCESS] Click dispatched",
            target_hwnd=12345,
            element_name="raw_coordinate",
            coordinate=(500, 300),
            button="left",
            real_input_dispatched=False,
        )

        action = {"valid": True, "action": "click", "x": 500, "y": 300, "button": "left"}
        res = dispatch_astra_ui_action(target_hwnd=12345, action_dict=action, real_execution=False)

        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "SIMULATED_CLICK")
        mock_verify.assert_called_once_with(12345, (500, 300))
        mock_simulate.assert_called_once_with(12345, (500, 300), button="left")

    def test_circuit_breaker_invariant_real_click_blocked(self):
        """Circuit Breaker: REAL_CLICK_ENABLED must remain False by default."""
        self.assertFalse(REAL_CLICK_ENABLED)


class TestAstraVisionEndToEnd(unittest.TestCase):
    """End-to-end execution flow with mocked vision capture and model query."""

    @patch("tools.orchestrator.verify_element_clickable")
    @patch("tools.orchestrator.simulate_click")
    def test_execute_with_astra_vision_flow(self, mock_simulate, mock_verify):
        mock_verify.return_value = ClickVerificationResult(is_safe=True, reason="SAFE")
        mock_simulate.return_value = SimulatedClickResult(
            success=True,
            status="SUCCESS",
            verification=mock_verify.return_value,
            action_log="[SIMULATED CLICK] at (527, 349)",
            target_hwnd=9999,
            element_name="raw_coordinate",
            coordinate=(527, 349),
            button="left",
            real_input_dispatched=False,
        )

        def mock_capturer(hwnd):
            return ("data:image/png;base64,DUMMY_IMAGE", (0, 0, 1920, 1080))

        def mock_api_runner(payload):
            return '{"action": "click", "x": 527, "y": 349}'

        client = AstraVisionClient(api_runner=mock_api_runner)

        res = execute_with_astra_vision(
            target_hwnd=9999,
            query="Click the Submit button",
            client=client,
            real_execution=False,
            screenshot_capturer=mock_capturer,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "SIMULATED_CLICK")
        self.assertEqual(res["coordinate"], (527, 349))
        self.assertEqual(res["parsed_action"]["action"], "click")


if __name__ == "__main__":
    unittest.main()
