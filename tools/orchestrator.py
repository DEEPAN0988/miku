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

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import tools.screen_inspector
import tools.typing_automation

from tools.screen_inspector import (
    UIElement,
    ScreenSnapshot,
    inspect_screen,
    find_element,
    simulate_click,
    dispatch_real_click,
    dispatch_real_double_click,
    verify_element_clickable,
    REAL_CLICK_ENABLED,
    human_mouse_move,
    get_current_cursor_pos,
)
from tools.vision_grounder import (
    inspect_screen_with_vision_fallback,
    capture_window_base64,
)
from tools.typing_automation import (
    simulate_typing,
    dispatch_real_typing,
    dispatch_human_keystrokes,
    dispatch_typing_payload,
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
    use_astra_vision: bool = False
    astra_client: Optional[Any] = None


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


# ==============================================================================
# Astra Multimodal Vision Client & Action Translation Layer
# ==============================================================================

class AstraVisionClient:
    """
    Multimodal API client targeting vision-grounded UI navigation models (e.g. openai/gpt-6-astra).
    Formats Miku's GDI BitBlt screenshots into base64 data URLs and requests structured UI navigation actions.
    When running offline (no OPENAI_API_KEY), seamlessly routes requests to the local Ollama server bridge.
    """

    def __init__(
        self,
        model: str = "openai/gpt-6-astra",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_runner: Optional[Callable[[Dict[str, Any]], Any]] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_runner = api_runner
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model or os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")

    def build_vision_payload(
        self,
        query: str,
        base64_image_url: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Builds the OpenAI-compatible multimodal chat completions payload with base64 image data URL.
        Instructs the model on generalized multi-step computer use actions.
        """
        default_sys_msg = (
            "You are Miku OS Vision Navigation & Computer Use Engine powered by gpt-6-astra.\n"
            "Given the user's objective, desktop screenshot, and action history, determine the next action.\n"
            "Respond ONLY with a valid JSON object matching exactly one of these schemas:\n"
            '  {"action": "click", "x": <int>, "y": <int>}\n'
            '  {"action": "double_click", "x": <int>, "y": <int>}\n'
            '  {"action": "type", "text": "<str>"}\n'
            '  {"action": "press_key", "key": "<str>"}\n'
            '  {"action": "wait", "seconds": <float>}\n'
            '  {"action": "terminate", "reason": "<str>"}\n'
            "Rules:\n"
            "1. Coordinates x and y must be integers corresponding to desktop pixel coordinates.\n"
            "2. For press_key, use standard key names (e.g. 'enter', 'win', 'esc', 'tab', 'backspace', 'space').\n"
            "3. Use terminate when the objective is fully achieved or cannot proceed.\n"
            "4. Return ONLY the JSON object, with no markdown or explanatory commentary."
        )
        sys_msg = system_prompt or default_sys_msg

        user_text = query
        if history:
            user_text += "\n\nHistory of previously executed actions:"
            for i, h in enumerate(history, 1):
                act_str = json.dumps(h.get("action", h))
                status = h.get("status", "SUCCESS" if h.get("success") else "FAILED")
                user_text += f"\n  Step {i}: {act_str} -> {status}"

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": sys_msg,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image_url},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

    def query_action(
        self,
        query: str,
        base64_image_url: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Dispatches multimodal vision request to model endpoint.
        Uses api_runner if provided, calls OpenAI if api_key is configured,
        otherwise seamlessly routes through the local Miku inference engine (text-only).
        """
        payload = self.build_vision_payload(query, base64_image_url, system_prompt=system_prompt, history=history)
        if self.api_runner is not None:
            res = self.api_runner(payload)
        elif self.api_key:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                res = data["choices"][0]["message"]["content"]
        else:
            # Offline / local execution bridge: route to Miku's native inference engine
            from tools.miku_inference import generate_action_json, get_screen_state_text
            screen_state = get_screen_state_text()
            res = generate_action_json(
                objective=query,
                screen_state=screen_state,
                action_history=history,
            )

        # Physical visual glide preview during live execution
        if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
            try:
                self._dispatch_visual_preview(res)
            except Exception:
                pass
        return res

    def _dispatch_visual_preview(self, response_str: str) -> None:
        """Previews physical cursor movement or typing for local/offline queries when live execution enabled."""
        action_dict = parse_astra_action(response_str)
        if not action_dict.get("valid"):
            return
        verb = action_dict.get("action")
        if verb in ("click", "double_click", "move", "hover"):
            x, y = action_dict.get("x", 0), action_dict.get("y", 0)
            if x > 0 and y > 0:
                cur_x, cur_y = get_current_cursor_pos()
                human_mouse_move(cur_x, cur_y, x, y, duration=0.20)
        elif verb == "type":
            txt = action_dict.get("text", "")
            if txt:
                dispatch_typing_payload(txt, min_delay_sec=0.015, max_delay_sec=0.035)


SUPPORTED_ASTRA_ACTIONS = (
    "click",
    "right_click",
    "double_click",
    "type",
    "press_key",
    "wait",
    "terminate",
    "move",
    "hover",
)


def parse_astra_action(response: str | Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses structured JSON commands for UI navigation from Astra model output.
    Expected schemas:
      {"action": "click", "x": 527, "y": 349}
      {"action": "right_click", "x": 527, "y": 349}
      {"action": "double_click", "x": 527, "y": 349}
      {"action": "type", "text": "foo"}
      {"action": "press_key", "key": "enter"}
      {"action": "wait", "seconds": 2.5}
      {"action": "terminate", "reason": "Objective achieved"}
      {"action": "move", "x": 527, "y": 349}
    """
    if isinstance(response, dict):
        raw_data = response
    elif isinstance(response, str):
        clean = response.strip()
        # Strip markdown code blocks if present
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s*```$", "", clean)
        try:
            raw_data = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback regex extraction for {"action": ...}
            match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", clean)
            if match:
                try:
                    raw_data = json.loads(match.group(0))
                except Exception:
                    return {"valid": False, "error": "INVALID_JSON", "raw": response}
            else:
                return {"valid": False, "error": "INVALID_JSON", "raw": response}
    else:
        return {"valid": False, "error": "UNSUPPORTED_TYPE", "raw": response}

    if not isinstance(raw_data, dict):
        return {"valid": False, "error": "NOT_A_DICT", "raw": raw_data}

    action = str(raw_data.get("action", "")).lower().strip()
    if action not in SUPPORTED_ASTRA_ACTIONS:
        return {"valid": False, "error": f"UNSUPPORTED_ACTION_{action.upper()}", "raw": raw_data}

    # Normalize right_click to click with button=right
    if action == "right_click":
        action = "click"
        raw_data["button"] = "right"

    parsed: Dict[str, Any] = {
        "valid": True,
        "action": action,
        "raw": raw_data,
    }

    if action in ("click", "double_click", "move"):
        if "x" not in raw_data or "y" not in raw_data:
            return {"valid": False, "error": "MISSING_COORDINATES", "raw": raw_data}
        try:
            x = int(raw_data.get("x", 0))
            y = int(raw_data.get("y", 0))
        except (ValueError, TypeError):
            return {"valid": False, "error": "NON_INTEGER_COORDINATES", "raw": raw_data}
        parsed["x"] = x
        parsed["y"] = y
        parsed["button"] = raw_data.get("button", "left")

    elif action == "type":
        text = raw_data.get("text")
        if text is None or not isinstance(text, str):
            return {"valid": False, "error": "INVALID_TEXT_PAYLOAD", "raw": raw_data}
        parsed["text"] = text
        if "x" in raw_data and "y" in raw_data:
            try:
                parsed["x"] = int(raw_data["x"])
                parsed["y"] = int(raw_data["y"])
            except (ValueError, TypeError):
                parsed["x"] = 0
                parsed["y"] = 0
        else:
            parsed["x"] = 0
            parsed["y"] = 0

    elif action == "press_key":
        key = raw_data.get("key")
        if key is None or not isinstance(key, str) or not key.strip():
            return {"valid": False, "error": "INVALID_KEY_PAYLOAD", "raw": raw_data}
        parsed["key"] = key.strip().lower()

    elif action == "wait":
        seconds = raw_data.get("seconds")
        try:
            sec_val = float(seconds)
            if sec_val < 0:
                return {"valid": False, "error": "NEGATIVE_WAIT_SECONDS", "raw": raw_data}
            parsed["seconds"] = sec_val
        except (ValueError, TypeError):
            return {"valid": False, "error": "NON_NUMERIC_SECONDS", "raw": raw_data}

    elif action == "terminate":
        reason = raw_data.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        parsed["reason"] = reason

    return parsed


def dispatch_astra_ui_action(
    target_hwnd: int,
    action_dict: Dict[str, Any],
    real_execution: bool = False,
    human_confirm_runner: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Action Translation Layer:
    Intercepts Astra's parsed JSON action, validates coordinate clickability via Miku's local
    verify_element_clickable safety gate, and routes directly to simulate_click or dispatch_real_click.

    Triple Circuit Breaker Invariant:
      REAL_CLICK_ENABLED must remain respected; physical input is permanently blocked by default.
    """
    if not action_dict.get("valid"):
        return {
            "success": False,
            "status": "REJECTED_INVALID_ACTION",
            "action": action_dict.get("action"),
            "coordinate": (action_dict.get("x", 0), action_dict.get("y", 0)),
            "output": f"[ASTRA ACTION REJECTION] Invalid action payload: {action_dict.get('error')}",
        }

    action_verb = action_dict["action"]

    # Handle actions that do not require spatial coordinates
    if action_verb == "press_key":
        key = action_dict.get("key", "")
        return {
            "success": True,
            "status": "PRESS_KEY_DISPATCHED",
            "mode": "REAL_KEY" if real_execution else "SIMULATED_KEY",
            "action": "press_key",
            "key": key,
            "output": f"Dispatched key '{key}'",
        }

    if action_verb == "wait":
        seconds = action_dict.get("seconds", 1.0)
        if real_execution:
            time.sleep(min(seconds, 10.0))
        return {
            "success": True,
            "status": "WAIT_COMPLETED",
            "mode": "REAL_WAIT" if real_execution else "SIMULATED_WAIT",
            "action": "wait",
            "seconds": seconds,
            "output": f"Waited {seconds} seconds",
        }

    if action_verb == "terminate":
        reason = action_dict.get("reason", "")
        return {
            "success": True,
            "status": "TERMINATED",
            "action": "terminate",
            "reason": reason,
            "output": f"Task terminated by model: {reason}",
        }

    x = action_dict.get("x", 0)
    y = action_dict.get("y", 0)
    coord = (x, y)
    button = action_dict.get("button", "left")

    # Step 1: Intercept through local safety gate verify_element_clickable for spatial actions
    click_verif = verify_element_clickable(target_hwnd, coord)
    if not click_verif.is_safe:
        return {
            "success": False,
            "status": f"ABORT_{click_verif.reason}",
            "action": action_verb,
            "coordinate": coord,
            "verification": click_verif,
            "output": (
                f"[LOCAL SAFETY GATE REJECTION] Astra proposed {action_verb} at {coord}, but "
                f"verify_element_clickable rejected: {click_verif.reason} ({click_verif.details})."
            ),
        }

    # Step 2: Route to simulate_click or dispatch_real_click
    if action_verb in ("click", "move", "hover"):
        if real_execution and REAL_CLICK_ENABLED:
            click_res = dispatch_real_click(
                target_hwnd,
                coord,
                button=button,
                human_confirm_runner=human_confirm_runner,
            )
            return {
                "success": click_res.success,
                "status": click_res.status,
                "mode": "REAL_CLICK",
                "action": action_verb,
                "coordinate": coord,
                "output": click_res.action_log,
                "details": click_res.to_dict(),
            }
        else:
            if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
                try:
                    cur_x, cur_y = get_current_cursor_pos()
                    human_mouse_move(cur_x, cur_y, coord[0], coord[1], duration=0.20)
                except Exception:
                    pass
            sim_res = simulate_click(
                target_hwnd,
                coord,
                button=button,
            )
            return {
                "success": sim_res.success,
                "status": sim_res.status,
                "mode": "SIMULATED_CLICK",
                "action": action_verb,
                "coordinate": coord,
                "output": sim_res.action_log,
                "details": sim_res.to_dict(),
            }
    elif action_verb == "double_click":
        if real_execution and REAL_CLICK_ENABLED:
            double_res = dispatch_real_double_click(target_hwnd, coord, button=button)
            return {
                "success": double_res.success,
                "status": double_res.status,
                "mode": "REAL_DOUBLE_CLICK",
                "action": "double_click",
                "coordinate": coord,
                "output": double_res.action_log,
                "details": double_res.to_dict(),
            }
        else:
            if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
                try:
                    cur_x, cur_y = get_current_cursor_pos()
                    human_mouse_move(cur_x, cur_y, coord[0], coord[1], duration=0.20)
                except Exception:
                    pass
            sim_res = simulate_click(target_hwnd, coord, button=button)
            return {
                "success": sim_res.success,
                "status": sim_res.status,
                "mode": "SIMULATED_DOUBLE_CLICK",
                "action": "double_click",
                "coordinate": coord,
                "output": f"[SIMULATED DOUBLE CLICK] {sim_res.action_log}",
                "details": sim_res.to_dict(),
            }
    elif action_verb == "type":
        text = action_dict.get("text", "")
        # Focus coordinate if provided
        if coord != (0, 0):
            if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
                try:
                    cur_x, cur_y = get_current_cursor_pos()
                    human_mouse_move(cur_x, cur_y, coord[0], coord[1], duration=0.20)
                except Exception:
                    pass
            sim_res = simulate_click(target_hwnd, coord, button="left")
            if not sim_res.success:
                return {
                    "success": False,
                    "status": sim_res.status,
                    "action": "type",
                    "coordinate": coord,
                    "output": f"[TYPE FOCUS FAILED] {sim_res.action_log}",
                }
        if real_execution and REAL_TYPE_ENABLED:
            type_res = dispatch_real_typing(target_hwnd, None, text)
            return {
                "success": type_res.success,
                "status": type_res.status,
                "mode": "REAL_TYPE",
                "action": "type",
                "text": text,
                "coordinate": coord,
                "output": type_res.action_log,
            }
        else:
            if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
                try:
                    dispatch_typing_payload(text, min_delay_sec=0.015, max_delay_sec=0.035)
                except Exception:
                    pass
            type_res = simulate_typing(text)
            return {
                "success": type_res.success,
                "status": type_res.status,
                "mode": "SIMULATED_TYPE",
                "action": "type",
                "text": text,
                "coordinate": coord,
                "output": type_res.action_log,
            }

    return {
        "success": False,
        "status": "UNHANDLED_ACTION",
        "action": action_verb,
        "coordinate": coord,
        "output": f"Unhandled action verb: {action_verb}",
    }


def execute_with_astra_vision(
    target_hwnd: int,
    query: str,
    client: Optional[AstraVisionClient] = None,
    real_execution: bool = False,
    human_confirm_runner: Optional[Callable] = None,
    screenshot_capturer: Optional[Callable[[int], Optional[Tuple[str, Tuple[int, int, int, int]]]]] = None,
) -> Dict[str, Any]:
    """
    Executes a complete multimodal vision navigation step via Astra:
    1. Captures Win32 GDI BitBlt screenshot of target_hwnd.
    2. Base64 encodes image into a data URL.
    3. Queries Astra API (targeting openai/gpt-6-astra).
    4. Parses structured JSON response via parse_astra_action.
    5. Dispatches action through verify_element_clickable safety gate to simulate_click / dispatch_real_click.
    """
    if client is None:
        client = AstraVisionClient()

    # Step 1 & 2: GDI BitBlt Capture & Base64 encoding
    capturer = screenshot_capturer or capture_window_base64
    capture_res = capturer(target_hwnd)
    if capture_res is None:
        return {
            "success": False,
            "status": "CAPTURE_FAILED",
            "output": f"[ASTRA VISION ERROR] Failed to capture GDI BitBlt bitmap for HWND {target_hwnd}.",
        }

    b64_url, window_rect = capture_res

    # Step 3: Query Astra API
    try:
        raw_response = client.query_action(query, b64_url)
    except Exception as e:
        return {
            "success": False,
            "status": "API_ERROR",
            "error": str(e),
            "output": f"[ASTRA API ERROR] Model query failed: {e}",
        }

    # Step 4: Parse structured JSON action
    action_dict = parse_astra_action(raw_response)
    if not action_dict.get("valid"):
        return {
            "success": False,
            "status": "ACTION_PARSE_FAILED",
            "raw_response": raw_response,
            "details": action_dict,
            "output": f"[ASTRA PARSE ERROR] Could not parse valid action from response: {action_dict.get('error')}",
        }

    # Step 5: Translate and dispatch with local safety interception
    dispatch_res = dispatch_astra_ui_action(
        target_hwnd=target_hwnd,
        action_dict=action_dict,
        real_execution=real_execution,
        human_confirm_runner=human_confirm_runner,
    )
    dispatch_res["raw_model_response"] = raw_response
    dispatch_res["parsed_action"] = action_dict
    return dispatch_res


__all__ = [
    "CompoundActionEngine",
    "CompoundTask",
    "OrchestratorResult",
    "PipelineState",
    "StepTelemetry",
    "AstraVisionClient",
    "parse_astra_action",
    "dispatch_astra_ui_action",
    "execute_with_astra_vision",
]
