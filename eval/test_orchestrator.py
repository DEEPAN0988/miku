"""
eval/test_orchestrator.py — Exhaustive Unit Test Suite for CompoundActionEngine

Tests:
  1. Nominal multi-step simulated workflow (Plan -> Focus -> Inspect -> Ground -> Click -> Audit -> Complete)
  2. Nominal workflow with valid text typing (into Edit control)
  3. Router yield on invalid/empty task parameters
  4. Router yield on dangerous/malformed typing payload
  5. Router yield on application focus/launch failure
  6. Router yield on empty UI inspection tree
  7. Router yield on missing UI target element
  8. Router yield on unclickable target element
  9. Router yield on typing into non-editable element (Button)
 10. Router yield on audit verification failure
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("."))

from tools.orchestrator import (
    CompoundActionEngine,
    CompoundTask,
    PipelineState,
    OrchestratorResult,
)
from tools.screen_inspector import (
    UIElement,
    ScreenSnapshot,
    SimulatedClickResult,
    ClickVerificationResult,
)
from tools.typing_automation import (
    SimulatedTypingResult,
    EditableVerificationResult,
)


def make_sim_click(
    success: bool = True,
    status: str = "SIMULATED_CLICK_SUCCESS",
    action_log: str = "Simulated click",
    hwnd: int = 12345,
    name: str = "Clear",
    coord: tuple = (150, 125),
) -> SimulatedClickResult:
    return SimulatedClickResult(
        success=success,
        status=status,
        verification=ClickVerificationResult(is_safe=success, reason="SAFE" if success else "FAILED"),
        action_log=action_log,
        target_hwnd=hwnd,
        element_name=name,
        coordinate=coord,
        button="left",
        real_input_dispatched=False,
    )


def make_sim_type(
    success: bool = True,
    status: str = "SIMULATED_TYPE_SUCCESS",
    action_log: str = "Simulated typing",
    hwnd: int = 12345,
    name: str = "Search",
    text: str = "test query",
) -> SimulatedTypingResult:
    return SimulatedTypingResult(
        success=success,
        status=status,
        verification=None,
        sanitized_text=text,
        action_log=action_log,
        target_hwnd=hwnd,
        element_name=name,
        real_input_dispatched=False,
    )


class TestCompoundActionEngineUnit(unittest.TestCase):
    """Exhaustive deterministic tests for CompoundActionEngine state machine."""

    def setUp(self):
        self.engine = CompoundActionEngine()
        self.mock_button = UIElement(
            control_type="Button",
            name="Clear",
            automation_id="clearButton",
            rect=(100, 100, 200, 150),
            center=(150, 125),
            is_enabled=True,
            is_focusable=True,
            source="uia",
        )
        self.mock_edit = UIElement(
            control_type="Edit",
            name="Search",
            automation_id="searchBox",
            rect=(100, 50, 300, 90),
            center=(200, 70),
            is_enabled=True,
            is_focusable=True,
            source="uia",
        )
        self.mock_snapshot = ScreenSnapshot(
            hwnd=12345,
            title="Mock App",
            class_name="MockClass",
            process_name="mock.exe",
            window_rect=(0, 0, 800, 600),
            elements=[self.mock_button, self.mock_edit],
            interactive_count=2,
            total_scanned=2,
            latency_ms=12.5,
        )

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    @patch("tools.orchestrator.find_element")
    @patch("tools.orchestrator.simulate_click")
    def test_nominal_simulated_workflow(self, mock_click, mock_find, mock_inspect, mock_focus):
        """Full nominal progression: Plan -> Focus -> Inspect -> Ground -> Click -> Audit -> Complete."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = self.mock_snapshot
        mock_find.return_value = self.mock_button
        mock_click.return_value = make_sim_click(success=True, status="SIMULATED_CLICK_SUCCESS", coord=(150, 125))

        task = CompoundTask(
            app_name="mockapp",
            target_element_query="Clear",
            control_type="Button",
            click_button="left",
            type_text=None,
            audit_element_query="Clear",
            use_vision_fallback=True,
            real_execution=False,
        )

        res = self.engine.execute(task)

        self.assertTrue(res.success, f"Nominal task should succeed: {res.action_log}")
        self.assertEqual(res.final_state, PipelineState.COMPLETE)
        self.assertIsNone(res.yield_reason)
        self.assertEqual(res.target_hwnd, 12345)
        self.assertEqual(res.grounded_element, self.mock_button)

        step_names = [t.step for t in res.telemetry]
        self.assertEqual(
            step_names,
            [
                PipelineState.PLAN,
                PipelineState.FOCUS_APP,
                PipelineState.INSPECT_UI,
                PipelineState.GROUND_COORDINATE,
                PipelineState.CLICK,
                PipelineState.AUDIT,
            ],
        )

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    @patch("tools.orchestrator.find_element")
    @patch("tools.orchestrator.simulate_click")
    @patch("tools.orchestrator.simulate_typing")
    def test_nominal_workflow_with_typing(self, mock_type, mock_click, mock_find, mock_inspect, mock_focus):
        """Nominal progression with typing into an Edit control."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = self.mock_snapshot
        mock_find.return_value = self.mock_edit
        mock_click.return_value = make_sim_click(success=True, status="SIMULATED_CLICK_SUCCESS", name="Search", coord=(200, 70))
        mock_type.return_value = make_sim_type(success=True, status="SIMULATED_TYPE_SUCCESS", name="Search", text="test query")

        task = CompoundTask(
            app_name="mockapp",
            target_element_query="Search",
            control_type="Edit",
            type_text="test query",
            real_execution=False,
        )

        res = self.engine.execute(task)

        self.assertTrue(res.success)
        self.assertEqual(res.final_state, PipelineState.COMPLETE)
        step_names = [t.step for t in res.telemetry]
        self.assertIn(PipelineState.TYPE, step_names)

    def test_yield_on_invalid_parameters(self):
        """Engine yields to router if required parameters are missing."""
        task = CompoundTask(app_name="", target_element_query="")
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "INVALID_TASK_PLAN_PARAMETERS")

    def test_yield_on_malformed_payload(self):
        """Engine yields to router at PLAN step if payload contains illegal control characters."""
        task = CompoundTask(
            app_name="mockapp",
            target_element_query="Search",
            type_text="hello\x00world",
        )
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertTrue(res.yield_reason.startswith("PAYLOAD_REJECTED_"))

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.launch_app")
    def test_yield_on_focus_and_launch_failure(self, mock_launch, mock_focus):
        """Engine yields to router if app cannot be focused or launched."""
        mock_focus.return_value = {"found": False}
        mock_launch.return_value = {"status": "FAILED"}

        task = CompoundTask(app_name="nonexistent_app", target_element_query="Button")
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "APP_FOCUS_AND_LAUNCH_FAILED")

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    def test_yield_on_empty_ui_inspection(self, mock_inspect, mock_focus):
        """Engine yields to router if UI tree contains zero elements."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = ScreenSnapshot(
            hwnd=12345,
            title="Mock App",
            class_name="MockClass",
            process_name="mock.exe",
            window_rect=(0, 0, 800, 600),
            elements=[],
            interactive_count=0,
            total_scanned=0,
            error="Empty UI tree",
        )

        task = CompoundTask(app_name="mockapp", target_element_query="Button")
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "UI_INSPECTION_EMPTY")

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    @patch("tools.orchestrator.find_element")
    def test_yield_on_missing_element(self, mock_find, mock_inspect, mock_focus):
        """Engine yields to router if target element cannot be located."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = self.mock_snapshot
        mock_find.return_value = None

        task = CompoundTask(app_name="mockapp", target_element_query="NonExistent")
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "ELEMENT_NOT_LOCATED")

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    @patch("tools.orchestrator.find_element")
    @patch("tools.orchestrator.simulate_click")
    def test_yield_on_click_failure(self, mock_click, mock_find, mock_inspect, mock_focus):
        """Engine yields to router if click simulation or dispatch fails."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = self.mock_snapshot
        mock_find.return_value = self.mock_button
        mock_click.return_value = make_sim_click(
            success=False,
            status="ABORT_DISABLED",
            action_log="Element is disabled",
            coord=(150, 125),
        )

        task = CompoundTask(app_name="mockapp", target_element_query="Clear")
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "CLICK_FAILED_ABORT_DISABLED")

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    @patch("tools.orchestrator.find_element")
    @patch("tools.orchestrator.simulate_click")
    @patch("tools.orchestrator.simulate_typing")
    def test_yield_on_typing_into_non_editable_element(self, mock_type, mock_click, mock_find, mock_inspect, mock_focus):
        """Engine yields to router if typing is attempted on a Button."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = self.mock_snapshot
        mock_find.return_value = self.mock_button
        mock_click.return_value = make_sim_click(success=True, status="SIMULATED_CLICK_SUCCESS", coord=(150, 125))
        mock_type.return_value = make_sim_type(
            success=False,
            status="ABORT_NOT_EDITABLE",
            action_log="Element is not editable",
            name="Clear",
            text="12345",
        )

        task = CompoundTask(
            app_name="mockapp",
            target_element_query="Clear",
            type_text="12345",
        )
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "TYPE_FAILED_ABORT_NOT_EDITABLE")

    @patch("tools.orchestrator.focus_app")
    @patch("tools.orchestrator.inspect_screen_with_vision_fallback")
    @patch("tools.orchestrator.find_element")
    @patch("tools.orchestrator.simulate_click")
    def test_yield_on_audit_failure(self, mock_click, mock_find, mock_inspect, mock_focus):
        """Engine yields to router if post-action audit element cannot be found."""
        mock_focus.return_value = {"found": True, "hwnd": 12345, "title": "Mock App"}
        mock_inspect.return_value = self.mock_snapshot
        # First call finds the target button, second call (audit) returns None
        mock_find.side_effect = [self.mock_button, None]
        mock_click.return_value = make_sim_click(success=True, status="SIMULATED_CLICK_SUCCESS", coord=(150, 125))

        task = CompoundTask(
            app_name="mockapp",
            target_element_query="Clear",
            audit_element_query="ExpectedResultLabel",
        )
        res = self.engine.execute(task)

        self.assertFalse(res.success)
        self.assertEqual(res.final_state, PipelineState.YIELD_TO_ROUTER)
        self.assertEqual(res.yield_reason, "POST_ACTION_AUDIT_TARGET_MISSING")


if __name__ == "__main__":
    unittest.main()
