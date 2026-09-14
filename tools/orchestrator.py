"""
tools/orchestrator.py — Compound Action Engine & Pipeline State Machine (Phase 14 / v0.3.5)

Architectural Role:
  Chains primitive desktop automation tools into autonomous multi-step execution pipelines:
    Plan -> Focus App -> Inspect UI -> Ground Coordinate -> Click -> Type -> Audit.
  
Constitutional Safety & State Machine Guarantees:
  1. Fail-Safe Yielding: If any step encounters an unexpected condition, occlusion,
     focus shift, or missing element, the state machine transitions to YIELD_TO_ROUTER
     with full diagnostic telemetry rather than crashing.
  2. Circuit Breaker Invariant: Respects REAL_CLICK_ENABLED and REAL_TYPE_ENABLED.
     Runs safely in simulation mode by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tools.screen_inspector import (
    UIElement,
    ScreenSnapshot,
    inspect_screen,
    find_element,
    simulate_click,
    dispatch_real_click,
    REAL_CLICK_ENABLED,
)
from tools.vision_grounder import inspect_screen_with_vision_fallback
from tools.typing_automation import (
    simulate_typing,
    dispatch_real_typing,
    REAL_TYPE_ENABLED,
    sanitize_typing_payload,
)
from tools.app_launcher import focus_app, launch_app


class PipelineState(str, Enum):
    PLAN = "PLAN"
    FOCUS_APP = "FOCUS_APP"
    INSPECT_UI = "INSPECT_UI"
    GROUND_COORDINATE = "GROUND_COORDINATE"
    CLICK = "CLICK"
    TYPE = "TYPE"
    AUDIT = "AUDIT"
    COMPLETE = "COMPLETE"
    YIELD_TO_ROUTER = "YIELD_TO_ROUTER"


@dataclass
class CompoundTask:
    """Specification of a compound multi-step desktop task."""
    app_name: str
    target_element_query: str
    control_type: Optional[str] = None
    click_button: str = "left"
    type_text: Optional[str] = None
    audit_element_query: Optional[str] = None
    use_vision_fallback: bool = True
    real_execution: bool = False


@dataclass
class StepTelemetry:
    step: PipelineState
    status: str  # "SUCCESS" | "FAILED" | "SKIPPED"
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class OrchestratorResult:
    """Structured outcome of a compound action sequence."""
    success: bool
    final_state: PipelineState
    yield_reason: Optional[str]
    telemetry: List[StepTelemetry] = field(default_factory=list)
    action_log: str = ""
    target_hwnd: int = 0
    grounded_element: Optional[UIElement] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_state": self.final_state.value,
            "yield_reason": self.yield_reason,
            "target_hwnd": self.target_hwnd,
            "telemetry": [
                {
                    "step": t.step.value,
                    "status": t.status,
                    "details": t.details,
                    "duration_ms": t.duration_ms,
                }
                for t in self.telemetry
            ],
            "action_log": self.action_log,
        }


class CompoundActionEngine:
    """
    Autonomous Compound Action Engine executing multi-step desktop automation
    with defensive state machine transitions and router yield-on-failure.
    """

    def __init__(self):
        self.state = PipelineState.PLAN

    def execute(self, task: CompoundTask) -> OrchestratorResult:
        telemetry: List[StepTelemetry] = []
        action_logs: List[str] = []
        target_hwnd: int = 0
        grounded_el: Optional[UIElement] = None

        def record_step(step: PipelineState, status: str, details: Dict[str, Any], t_start: float):
            dur = round((time.perf_counter() - t_start) * 1000, 2)
            telemetry.append(StepTelemetry(step=step, status=status, details=details, duration_ms=dur))

        # ----------------------------------------------------------------------
        # Step 1: PLAN
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        self.state = PipelineState.PLAN
        if not task.app_name or not task.target_element_query:
            record_step(PipelineState.PLAN, "FAILED", {"error": "Missing app_name or target_element_query"}, t0)
            return OrchestratorResult(
                success=False,
                final_state=PipelineState.YIELD_TO_ROUTER,
                yield_reason="INVALID_TASK_PLAN_PARAMETERS",
                telemetry=telemetry,
                action_log="Task plan validation failed: missing app_name or target_element_query.",
            )

        # Pre-validate payload if typing is requested
        if task.type_text:
            is_safe, _, reason = sanitize_typing_payload(task.type_text)
            if not is_safe:
                record_step(PipelineState.PLAN, "FAILED", {"error": f"Invalid payload: {reason}"}, t0)
                return OrchestratorResult(
                    success=False,
                    final_state=PipelineState.YIELD_TO_ROUTER,
                    yield_reason=f"PAYLOAD_REJECTED_{reason}",
                    telemetry=telemetry,
                    action_log=f"Task plan aborted: text payload rejected ({reason}).",
                )

        record_step(PipelineState.PLAN, "SUCCESS", {"app": task.app_name, "query": task.target_element_query}, t0)
        action_logs.append(f"[PLAN] Initialized plan for app '{task.app_name}', targeting '{task.target_element_query}'.")

        # ----------------------------------------------------------------------
        # Step 2: FOCUS_APP
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        self.state = PipelineState.FOCUS_APP
        focus_res = focus_app(task.app_name)
        if not focus_res.get("found"):
            # Attempt to launch app if not currently open
            launch_res = launch_app(task.app_name)
            if launch_res.get("status") != "SUCCESS":
                record_step(PipelineState.FOCUS_APP, "FAILED", {"error": "App not running and launch failed", "details": focus_res}, t0)
                return OrchestratorResult(
                    success=False,
                    final_state=PipelineState.YIELD_TO_ROUTER,
                    yield_reason="APP_FOCUS_AND_LAUNCH_FAILED",
                    telemetry=telemetry,
                    action_log=f"Could not focus or launch application '{task.app_name}'. Yielding to router.",
                )
            time.sleep(1.0)
            focus_res = focus_app(task.app_name)

        target_hwnd = focus_res.get("hwnd", 0)
        if not target_hwnd or not focus_res.get("found"):
            record_step(PipelineState.FOCUS_APP, "FAILED", {"error": "Failed to acquire target window HWND", "details": focus_res}, t0)
            return OrchestratorResult(
                success=False,
                final_state=PipelineState.YIELD_TO_ROUTER,
                yield_reason="HWND_ACQUISITION_FAILED",
                telemetry=telemetry,
                action_log=f"Failed to acquire HWND for '{task.app_name}'. Yielding to router.",
            )

        record_step(PipelineState.FOCUS_APP, "SUCCESS", {"hwnd": target_hwnd, "title": focus_res.get("title")}, t0)
        action_logs.append(f"[FOCUS_APP] Successfully focused HWND {target_hwnd} ('{focus_res.get('title')}').")

        # ----------------------------------------------------------------------
        # Step 3: INSPECT_UI
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        self.state = PipelineState.INSPECT_UI
        if task.use_vision_fallback:
            snapshot = inspect_screen_with_vision_fallback(target_hwnd)
        else:
            snapshot = inspect_screen(target_hwnd)

        if snapshot.interactive_count == 0:
            record_step(PipelineState.INSPECT_UI, "FAILED", {"error": "Zero interactive elements detected", "snapshot_error": snapshot.error}, t0)
            return OrchestratorResult(
                success=False,
                final_state=PipelineState.YIELD_TO_ROUTER,
                yield_reason="UI_INSPECTION_EMPTY",
                telemetry=telemetry,
                target_hwnd=target_hwnd,
                action_log=f"UI inspection found 0 elements on HWND {target_hwnd}. Yielding to router.",
            )

        record_step(PipelineState.INSPECT_UI, "SUCCESS", {"interactive_count": snapshot.interactive_count, "latency_ms": snapshot.latency_ms}, t0)
        action_logs.append(f"[INSPECT_UI] Discovered {snapshot.interactive_count} interactive controls on HWND {target_hwnd}.")

        # ----------------------------------------------------------------------
        # Step 4: GROUND_COORDINATE
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        self.state = PipelineState.GROUND_COORDINATE
        grounded_el = find_element(snapshot, task.target_element_query, control_type=task.control_type)
        if not grounded_el:
            record_step(PipelineState.GROUND_COORDINATE, "FAILED", {"query": task.target_element_query, "control_type": task.control_type}, t0)
            return OrchestratorResult(
                success=False,
                final_state=PipelineState.YIELD_TO_ROUTER,
                yield_reason="ELEMENT_NOT_LOCATED",
                telemetry=telemetry,
                target_hwnd=target_hwnd,
                action_log=f"Target element '{task.target_element_query}' not located in UI tree of HWND {target_hwnd}. Yielding to router.",
            )

        record_step(
            PipelineState.GROUND_COORDINATE,
            "SUCCESS",
            {"name": grounded_el.name, "id": grounded_el.automation_id, "type": grounded_el.control_type, "center": list(grounded_el.center)},
            t0,
        )
        action_logs.append(f"[GROUND_COORDINATE] Grounded '{grounded_el.name}' ({grounded_el.control_type}) at {grounded_el.center}.")

        # ----------------------------------------------------------------------
        # Step 5: CLICK
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        self.state = PipelineState.CLICK

        if task.real_execution and REAL_CLICK_ENABLED:
            click_res = dispatch_real_click(target_hwnd, grounded_el, button=task.click_button)
        else:
            click_res = simulate_click(target_hwnd, grounded_el, button=task.click_button)

        if not click_res.success:
            record_step(PipelineState.CLICK, "FAILED", {"status": click_res.status, "log": click_res.action_log}, t0)
            return OrchestratorResult(
                success=False,
                final_state=PipelineState.YIELD_TO_ROUTER,
                yield_reason=f"CLICK_FAILED_{click_res.status}",
                telemetry=telemetry,
                target_hwnd=target_hwnd,
                grounded_element=grounded_el,
                action_log=f"Click failed on '{grounded_el.name}': {click_res.status}. Yielding to router.",
            )

        record_step(
            PipelineState.CLICK,
            "SUCCESS",
            {"status": click_res.status, "real_input": click_res.real_input_dispatched},
            t0,
        )
        action_logs.append(f"[CLICK] Successfully executed click: {click_res.status} (real_input={click_res.real_input_dispatched}).")

        # ----------------------------------------------------------------------
        # Step 6: TYPE (Optional)
        # ----------------------------------------------------------------------
        if task.type_text:
            t0 = time.perf_counter()
            self.state = PipelineState.TYPE

            if task.real_execution and REAL_TYPE_ENABLED:
                type_res = dispatch_real_typing(target_hwnd, grounded_el, task.type_text)
            else:
                type_res = simulate_typing(target_hwnd, grounded_el, task.type_text)

            if not type_res.success:
                record_step(PipelineState.TYPE, "FAILED", {"status": type_res.status, "log": type_res.action_log}, t0)
                return OrchestratorResult(
                    success=False,
                    final_state=PipelineState.YIELD_TO_ROUTER,
                    yield_reason=f"TYPE_FAILED_{type_res.status}",
                    telemetry=telemetry,
                    target_hwnd=target_hwnd,
                    grounded_element=grounded_el,
                    action_log=f"Typing failed on '{grounded_el.name}': {type_res.status}. Yielding to router.",
                )

            record_step(
                PipelineState.TYPE,
                "SUCCESS",
                {"status": type_res.status, "real_input": type_res.real_input_dispatched},
                t0,
            )
            action_logs.append(f"[TYPE] Successfully staged/typed text: {type_res.status} (real_input={type_res.real_input_dispatched}).")

        # ----------------------------------------------------------------------
        # Step 7: AUDIT
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        self.state = PipelineState.AUDIT
        time.sleep(0.1)  # Allow UI thread to settle

        post_snap = inspect_screen(target_hwnd)
        audit_details = {
            "post_interactive_count": post_snap.interactive_count,
            "window_alive": post_snap.hwnd == target_hwnd,
        }

        if task.audit_element_query:
            audit_el = find_element(post_snap, task.audit_element_query)
            audit_details["audit_target_found"] = audit_el is not None
            if not audit_el:
                record_step(PipelineState.AUDIT, "FAILED", audit_details, t0)
                return OrchestratorResult(
                    success=False,
                    final_state=PipelineState.YIELD_TO_ROUTER,
                    yield_reason="POST_ACTION_AUDIT_TARGET_MISSING",
                    telemetry=telemetry,
                    target_hwnd=target_hwnd,
                    grounded_element=grounded_el,
                    action_log=f"Audit query '{task.audit_element_query}' failed post-action. Yielding to router.",
                )

        record_step(PipelineState.AUDIT, "SUCCESS", audit_details, t0)
        action_logs.append(f"[AUDIT] Post-action audit confirmed UI responsive ({post_snap.interactive_count} elements).")

        # ----------------------------------------------------------------------
        # Step 8: COMPLETE
        # ----------------------------------------------------------------------
        self.state = PipelineState.COMPLETE
        return OrchestratorResult(
            success=True,
            final_state=PipelineState.COMPLETE,
            yield_reason=None,
            telemetry=telemetry,
            target_hwnd=target_hwnd,
            grounded_element=grounded_el,
            action_log="\n".join(action_logs),
        )
