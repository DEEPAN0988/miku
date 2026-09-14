"""
tools/computer_use_agent.py — Generalized Multi-Step Computer Use Agent (Phase 15 / gpt-6-astra)

Architectural Role:
  Provides fully generalized, multi-step computer use driven by visual perception
  and reasoned action execution via openai/gpt-6-astra. Miku continuously captures the
  desktop screen, reasons over the open-ended objective and action history, and dispatches
  Win32 keyboard and mouse primitives until the objective is accomplished or terminated.

Constitutional Safety & Defenses:
  1. MAX_STEPS = 15: Loop safeguard prevents runaway infinite iteration.
  2. Unconditional Spatial Safety Gate: Every click/double-click passes through
     verify_element_clickable (checking HWND validity, minimized state, foreground,
     window rect bounds, and point-in-time occlusion).
  3. Payload Sanitizer: All typed text is routed through sanitize_typing_payload.
  4. Triple Circuit Breaker: Honors MIKU_LIVE_EXECUTION and MIKU_AUTONOMOUS_MODE.
     If autonomous mode is off, prompts for user console approval before executing each step.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import win32gui

import tools.screen_inspector
import tools.typing_automation

from tools.screen_inspector import (
    _attach_thread_to_default_desktop,
    verify_element_clickable,
    simulate_click,
    dispatch_real_click,
    ClickVerificationResult,
    check_autonomous_authorization,
    REAL_CLICK_ENABLED,
    human_mouse_move,
    set_window_foreground_passive,
)
from tools.typing_automation import (
    sanitize_typing_payload,
    simulate_typing,
    dispatch_real_typing,
    dispatch_human_keystrokes,
    REAL_TYPE_ENABLED,
    INPUT,
    INPUT_KEYBOARD,
    KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE,
)
from tools.vision_grounder import capture_window_base64
from tools.orchestrator import AstraVisionClient, parse_astra_action

_attach_thread_to_default_desktop()

user32 = ctypes.windll.user32

# Loop safeguard constant
MAX_STEPS: int = 15

# ==============================================================================
# Keystroke Mapping: String Key -> Win32 Virtual-Key (VK) Code
# ==============================================================================

KEY_MAPPING: Dict[str, int] = {
    # Control & Navigation
    "enter": 0x0D,        # VK_RETURN
    "return": 0x0D,       # VK_RETURN
    "tab": 0x09,          # VK_TAB
    "esc": 0x1B,          # VK_ESCAPE
    "escape": 0x1B,       # VK_ESCAPE
    "backspace": 0x08,    # VK_BACK
    "delete": 0x2E,       # VK_DELETE
    "del": 0x2E,          # VK_DELETE
    "space": 0x20,        # VK_SPACE
    "spacebar": 0x20,     # VK_SPACE
    # Modifiers / Special
    "win": 0x5B,          # VK_LWIN
    "windows": 0x5B,      # VK_LWIN
    "super": 0x5B,        # VK_LWIN
    "rwin": 0x5C,         # VK_RWIN
    "shift": 0x10,        # VK_SHIFT
    "lshift": 0xA0,       # VK_LSHIFT
    "rshift": 0xA1,       # VK_RSHIFT
    "ctrl": 0x11,         # VK_CONTROL
    "control": 0x11,      # VK_CONTROL
    "lctrl": 0xA2,        # VK_LCONTROL
    "rctrl": 0xA3,        # VK_RCONTROL
    "alt": 0x12,          # VK_MENU
    "lalt": 0xA4,         # VK_LMENU
    "ralt": 0xA5,         # VK_RMENU
    "capslock": 0x14,     # VK_CAPITAL
    "caps_lock": 0x14,    # VK_CAPITAL
    # Directional Arrows
    "up": 0x26,           # VK_UP
    "down": 0x28,         # VK_DOWN
    "left": 0x25,         # VK_LEFT
    "right": 0x27,        # VK_RIGHT
    # Navigation Block
    "home": 0x24,         # VK_HOME
    "end": 0x23,          # VK_END
    "pageup": 0x21,       # VK_PRIOR
    "page_up": 0x21,      # VK_PRIOR
    "pagedown": 0x22,     # VK_NEXT
    "page_down": 0x22,    # VK_NEXT
    "insert": 0x2D,       # VK_INSERT
    "prtscr": 0x2C,       # VK_SNAPSHOT
    "printscreen": 0x2C,  # VK_SNAPSHOT
    # Function Keys
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


def resolve_virtual_key(key_name: str) -> Optional[int]:
    """
    Resolves a string key name to its Win32 Virtual-Key (VK) Code.
    Supports named keys (e.g. "enter", "win", "esc", "tab") and alphanumeric characters ('a'-'z', '0'-'9').
    """
    if not isinstance(key_name, str):
        return None
    clean = key_name.strip().lower()
    if clean in KEY_MAPPING:
        return KEY_MAPPING[clean]
    if len(clean) == 1:
        ch = clean.upper()
        if 'A' <= ch <= 'Z' or '0' <= ch <= '9':
            return ord(ch)
        vk = user32.VkKeyScanW(ord(clean[0]))
        if vk != -1:
            return vk & 0xFF
    return None


# ==============================================================================
# Circuit Breakers & User Step Approval
# ==============================================================================

def is_autonomous_authorized() -> bool:
    """
    Evaluates whether autonomous execution is fully authorized.
    Requires BOTH MIKU_LIVE_EXECUTION=true and MIKU_AUTONOMOUS_MODE=true.
    """
    return check_autonomous_authorization("COMPUTER_USE")


def request_step_approval(
    step_index: int,
    action: Dict[str, Any],
    prompt_runner: Optional[Callable[[str], bool]] = None,
) -> bool:
    """
    Prompts the user via console to approve the step when autonomous mode is off.
    If a custom prompt_runner is provided, delegates to it (essential for testing and custom UIs).
    Otherwise, prompts interactively on sys.stdin. Fails closed (returns False) in non-interactive environments.
    """
    prompt = (
        f"\n" + "=" * 70 + "\n"
        f"[COMPUTER USE STEP APPROVAL REQUIRED]\n"
        f"Step Index : {step_index}\n"
        f"Action     : {json.dumps(action, indent=2)}\n"
        f"Circuit Breakers: MIKU_AUTONOMOUS_MODE=false\n"
        f"Approve this step? (y/n / 'approve'): "
    )
    if prompt_runner is not None:
        return bool(prompt_runner(prompt))

    if not sys.stdin or not sys.stdin.isatty():
        print(f"[!] Non-interactive environment without prompt_runner; step {step_index} rejected.", flush=True)
        return False

    try:
        resp = input(prompt).strip().lower()
        return resp in ("y", "yes", "approve", "ok", "1")
    except Exception:
        return False


# ==============================================================================
# Spatial Target Window Resolution & Safety Gate
# ==============================================================================

def resolve_target_hwnd_at_point(x: int, y: int) -> int:
    """
    Resolves the target top-level root window handle at coordinate (x, y).
    If no specific top-level window is found, falls back to the current foreground window.
    """
    pt = wintypes.POINT(x, y)
    hwnd_at_pt = user32.WindowFromPoint(pt)
    if hwnd_at_pt and user32.IsWindow(hwnd_at_pt):
        root_hwnd = user32.GetAncestor(hwnd_at_pt, 2) or hwnd_at_pt
        if root_hwnd and user32.IsWindow(root_hwnd):
            return root_hwnd
    return user32.GetForegroundWindow() or 0


# ==============================================================================
# Telemetry & Result Structures
# ==============================================================================

@dataclass
class ComputerUseStepResult:
    step: int
    action: Dict[str, Any]
    success: bool
    status: str
    output: str
    details: Dict[str, Any] = field(default_factory=dict)
    verification: Optional[ClickVerificationResult] = None


@dataclass
class ComputerUseTaskResult:
    success: bool
    objective: str
    total_steps: int
    final_status: str
    steps: List[ComputerUseStepResult] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "objective": self.objective,
            "total_steps": self.total_steps,
            "final_status": self.final_status,
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "success": s.success,
                    "status": s.status,
                    "output": s.output,
                    "details": s.details,
                }
                for s in self.steps
            ],
            "action_history": self.action_history,
            "error": self.error,
            "output": self.output,
        }


# ==============================================================================
# Action Dispatcher
# ==============================================================================

def dispatch_computer_use_action(
    action_dict: Dict[str, Any],
    real_execution: bool = False,
    target_hwnd: Optional[int] = None,
) -> ComputerUseStepResult:
    action_verb = action_dict.get("action", "")
    if action_verb == "right_click":
        action_verb = "click"
        if "button" not in action_dict:
            action_dict["button"] = "right"

    # 1. Termination
    if action_verb == "terminate":
        reason = action_dict.get("reason", "Objective achieved")
        return ComputerUseStepResult(
            step=0,
            action=action_dict,
            success=True,
            status="TERMINATED",
            output=f"Task terminated: {reason}",
            details={"reason": reason},
        )

    # 2. Wait
    if action_verb == "wait":
        seconds = float(action_dict.get("seconds", 1.0))
        safe_wait = min(seconds, 10.0)
        time.sleep(safe_wait)
        return ComputerUseStepResult(
            step=0,
            action=action_dict,
            success=True,
            status="WAIT_COMPLETED",
            output=f"Waited {safe_wait:.2f} seconds",
            details={"requested_seconds": seconds, "waited_seconds": safe_wait},
        )

    # 3. Press Key
    if action_verb == "press_key":
        key = action_dict.get("key", "")

        # Support key chords/combinations (e.g., "ctrl+c", "ctrl+v", "ctrl+s")
        if "+" in key:
            parts = [p.strip().lower() for p in key.split("+") if p.strip()]
            vk_codes = [resolve_virtual_key(p) for p in parts]
            if any(v is None for v in vk_codes):
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=False,
                    status="ABORT_UNKNOWN_KEY",
                    output=f"[KEY REJECTION] Unknown or unsupported key in chord: '{key}'",
                    details={"key": key},
                )

            if real_execution:
                for vk in vk_codes:
                    user32.keybd_event(vk, 0, 0, 0)
                    time.sleep(0.02)
                time.sleep(0.05)
                for vk in reversed(vk_codes):
                    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
                    time.sleep(0.02)
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=True,
                    status="REAL_KEY_CHORD_DISPATCHED",
                    output=f"[REAL KEY] Pressed key chord '{key}'",
                    details={"key": key, "vk_codes": vk_codes, "real_input": True},
                )
            else:
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=True,
                    status="SIMULATED_KEY_SUCCESS",
                    output=f"[SIMULATED KEY] Would press key chord '{key}'",
                    details={"key": key, "vk_codes": vk_codes, "real_input": False},
                )

        vk = resolve_virtual_key(key)
        if vk is None:
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=False,
                status="ABORT_UNKNOWN_KEY",
                output=f"[KEY REJECTION] Unknown or unsupported virtual key: '{key}'",
                details={"key": key},
            )

        if real_execution:
            # Win32 keybd_event down & up
            user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.05)
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=True,
                status="REAL_KEY_DISPATCHED",
                output=f"[REAL KEY] Pressed key '{key}' (VK 0x{vk:02X})",
                details={"key": key, "vk_code": vk, "real_input": True},
            )
        else:
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=True,
                status="SIMULATED_KEY_SUCCESS",
                output=f"[SIMULATED KEY] Would press key '{key}' (VK 0x{vk:02X})",
                details={"key": key, "vk_code": vk, "real_input": False},
            )

    # 4. Typing
    if action_verb == "type":
        text = action_dict.get("text", "")
        is_safe, safe_text, text_reason = sanitize_typing_payload(text)
        if not is_safe:
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=False,
                status=f"ABORT_PAYLOAD_{text_reason}",
                output=f"[PAYLOAD REJECTION] Typing payload rejected: {text_reason}",
                details={"reason": text_reason},
            )

        allow_real_typing = real_execution and (
            REAL_TYPE_ENABLED or tools.typing_automation.REAL_TYPE_ENABLED or check_autonomous_authorization("TYPE")
        )
        if allow_real_typing:
            typed_count = dispatch_human_keystrokes(safe_text, min_delay_sec=0.015, max_delay_sec=0.035)
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=True,
                status="REAL_TYPE_SUCCESS",
                output=f"[REAL TYPING] Typed {typed_count} characters: '{safe_text[:30]}'",
                details={"characters_typed": typed_count, "real_input": True},
            )
        else:
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=True,
                status="SIMULATED_TYPE_SUCCESS",
                output=f"[SIMULATED TYPING] Would type '{safe_text[:30]}'",
                details={"text": safe_text, "real_input": False},
            )

    # 5. Spatial Click & Double Click (Grounded through Pre-Click Safety Verification)
    if action_verb in ("click", "double_click", "move", "hover"):
        x = int(action_dict.get("x", 0))
        y = int(action_dict.get("y", 0))
        coord = (x, y)
        button = action_dict.get("button", "left")

        if target_hwnd is not None:
            resolved_hwnd = target_hwnd
        else:
            resolved_hwnd = resolve_target_hwnd_at_point(x, y)
            if not resolved_hwnd or not user32.IsWindow(resolved_hwnd):
                resolved_hwnd = user32.GetForegroundWindow() or user32.GetDesktopWindow()

        # If real execution is active and target window is not currently foreground,
        # passively bring it to foreground so pre-click safety verification passes without race conditions.
        if real_execution and resolved_hwnd and user32.IsWindow(resolved_hwnd):
            fg_now = user32.GetForegroundWindow()
            root_fg = user32.GetAncestor(fg_now, 2) or fg_now
            root_resolved = user32.GetAncestor(resolved_hwnd, 2) or resolved_hwnd
            if fg_now != resolved_hwnd and root_fg != root_resolved:
                set_window_foreground_passive(resolved_hwnd, timeout_s=0.5)

        # Unconditional Pre-Click Verification
        verif: ClickVerificationResult = verify_element_clickable(resolved_hwnd, coord)
        if not verif.is_safe:
            return ComputerUseStepResult(
                step=0,
                action=action_dict,
                success=False,
                status=f"ABORT_{verif.reason}",
                output=(
                    f"[LOCAL SAFETY GATE REJECTION] {action_verb} at {coord} rejected: "
                    f"{verif.reason} ({verif.details})."
                ),
                details=verif.details,
                verification=verif,
            )

        allow_real_click = real_execution and (
            REAL_CLICK_ENABLED
            or tools.screen_inspector.REAL_CLICK_ENABLED
            or check_autonomous_authorization("CLICK")
            or check_autonomous_authorization("COMPUTER_USE")
        )

        if action_verb in ("move", "hover"):
            if allow_real_click:
                human_mouse_move(None, None, x, y, duration=0.20)
                time.sleep(0.05)
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=True,
                    status="REAL_HOVER_SUCCESS",
                    output=f"[REAL HOVER] Hovered over {coord}",
                    details={"coordinate": coord, "real_input": True},
                    verification=verif,
                )
            else:
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=True,
                    status="SIMULATED_HOVER_SUCCESS",
                    output=f"[SIMULATED HOVER] Would hover over {coord}",
                    details={"coordinate": coord, "real_input": False},
                    verification=verif,
                )

        elif action_verb == "click":
            if allow_real_click:
                human_mouse_move(None, None, x, y, duration=0.20)
                time.sleep(0.02)
                down_flag = 0x0008 if button.lower() == "right" else 0x0002
                up_flag = 0x0010 if button.lower() == "right" else 0x0004
                user32.mouse_event(down_flag, 0, 0, 0, 0)
                time.sleep(0.025)
                user32.mouse_event(up_flag, 0, 0, 0, 0)
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=True,
                    status="REAL_CLICK_SUCCESS",
                    output=f"[REAL CLICK] Clicked at {coord} with {button} button",
                    details={"coordinate": coord, "button": button, "real_input": True},
                    verification=verif,
                )
            else:
                sim_res = simulate_click(resolved_hwnd, coord, button=button)
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=sim_res.success,
                    status=sim_res.status,
                    output=f"[SIMULATED CLICK] {sim_res.action_log}",
                    details=sim_res.to_dict(),
                    verification=verif,
                )

        elif action_verb == "double_click":
            if allow_real_click:
                human_mouse_move(None, None, x, y, duration=0.20)
                time.sleep(0.02)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.02)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.05)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.02)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=True,
                    status="REAL_DOUBLE_CLICK_SUCCESS",
                    output=f"[REAL DOUBLE CLICK] Double clicked at {coord}",
                    details={"coordinate": coord, "real_input": True},
                    verification=verif,
                )
            else:
                sim_res = simulate_click(resolved_hwnd, coord, button=button)
                return ComputerUseStepResult(
                    step=0,
                    action=action_dict,
                    success=sim_res.success,
                    status="SIMULATED_DOUBLE_CLICK_SUCCESS",
                    output=f"[SIMULATED DOUBLE CLICK] Double clicked at {coord}: {sim_res.action_log}",
                    details=sim_res.to_dict(),
                    verification=verif,
                )

    return ComputerUseStepResult(
        step=0,
        action=action_dict,
        success=False,
        status="UNHANDLED_ACTION",
        output=f"Unhandled action: {action_verb}",
        details={"action": action_verb},
    )


# ==============================================================================
# Agentic Execution Loop
# ==============================================================================

def run_computer_use_task(
    objective: str,
    max_steps: int = MAX_STEPS,
    client: Optional[AstraVisionClient] = None,
    real_execution: Optional[bool] = None,
    screenshot_capturer: Optional[Callable[[int], Optional[Tuple[str, Tuple[int, int, int, int]]]]] = None,
    prompt_runner: Optional[Callable[[str], bool]] = None,
    delay_between_steps: float = 0.5,
    target_hwnd_override: Optional[int] = None,
) -> ComputerUseTaskResult:
    """
    Agentic Execution Loop for Generalized Multi-Step Computer Use:
    1. Loop safeguard: max_steps (default MAX_STEPS = 15).
    2. In each iteration:
       a. Capture full desktop screen (win32gui.GetDesktopWindow()).
       b. Send screenshot, objective, and action history to gpt-6-astra.
       c. Parse structured JSON action.
       d. Check circuit breakers & request console approval if autonomous mode is off.
       e. Dispatch action through Miku's native Win32 primitives & safety gates.
       f. Break loop on terminate or max_steps reached.
    """
    if client is None:
        client = AstraVisionClient(model="openai/gpt-6-astra")

    if real_execution is None:
        real_execution = os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes")

    autonomous = is_autonomous_authorized()
    capturer = screenshot_capturer or capture_window_base64

    step_results: List[ComputerUseStepResult] = []
    action_history: List[Dict[str, Any]] = []

    step = 0
    while True:
        if step >= max_steps:
            return ComputerUseTaskResult(
                success=False,
                objective=objective,
                total_steps=step,
                final_status="MAX_STEPS_EXCEEDED",
                steps=step_results,
                action_history=action_history,
                error=f"Exceeded maximum allowable steps ({max_steps}). Safe termination triggered.",
                output=f"[SAFEGUARD] Loop terminated after reaching MAX_STEPS={max_steps}.",
            )

        step += 1

        # ----------------------------------------------------------------------
        # Step a: Capture the full desktop screen
        # ----------------------------------------------------------------------
        _attach_thread_to_default_desktop()
        desktop_hwnd = win32gui.GetDesktopWindow()
        capture_res = capturer(desktop_hwnd)
        if capture_res is None:
            err_step = ComputerUseStepResult(
                step=step,
                action={"action": "capture"},
                success=False,
                status="CAPTURE_FAILED",
                output=f"Failed to capture desktop screen for HWND {desktop_hwnd}.",
            )
            step_results.append(err_step)
            return ComputerUseTaskResult(
                success=False,
                objective=objective,
                total_steps=step,
                final_status="CAPTURE_FAILED",
                steps=step_results,
                action_history=action_history,
                error="Desktop screen capture failed.",
                output="Screen capture failed.",
            )

        b64_url, window_rect = capture_res

        # ----------------------------------------------------------------------
        # Step b: Send screenshot, objective, and action history to gpt-6-astra
        # ----------------------------------------------------------------------
        try:
            raw_response = client.query_action(
                query=objective,
                base64_image_url=b64_url,
                history=action_history,
            )
        except Exception as e:
            err_step = ComputerUseStepResult(
                step=step,
                action={"action": "query_model"},
                success=False,
                status="API_ERROR",
                output=f"Model query failed: {e}",
                details={"error": str(e)},
            )
            step_results.append(err_step)
            return ComputerUseTaskResult(
                success=False,
                objective=objective,
                total_steps=step,
                final_status="API_ERROR",
                steps=step_results,
                action_history=action_history,
                error=str(e),
                output=f"Model query failed: {e}",
            )

        # ----------------------------------------------------------------------
        # Step c: Parse returned JSON action
        # ----------------------------------------------------------------------
        action_dict = parse_astra_action(raw_response)
        if not action_dict.get("valid"):
            err_step = ComputerUseStepResult(
                step=step,
                action={"action": "parse", "raw": raw_response},
                success=False,
                status="ACTION_PARSE_FAILED",
                output=f"Failed to parse valid action: {action_dict.get('error')}",
                details=action_dict,
            )
            step_results.append(err_step)
            return ComputerUseTaskResult(
                success=False,
                objective=objective,
                total_steps=step,
                final_status="ACTION_PARSE_FAILED",
                steps=step_results,
                action_history=action_history,
                error=action_dict.get("error"),
                output=f"Failed to parse action: {action_dict.get('error')}",
            )

        # ----------------------------------------------------------------------
        # Step d: Circuit Breaker & Approval Gate
        # ----------------------------------------------------------------------
        if not autonomous:
            approved = request_step_approval(step, action_dict, prompt_runner=prompt_runner)
            if not approved:
                rej_step = ComputerUseStepResult(
                    step=step,
                    action=action_dict,
                    success=False,
                    status="ABORT_USER_REJECTED",
                    output=f"Step {step} rejected by human approval gate.",
                    details={"action": action_dict},
                )
                step_results.append(rej_step)
                return ComputerUseTaskResult(
                    success=False,
                    objective=objective,
                    total_steps=step,
                    final_status="USER_REJECTED",
                    steps=step_results,
                    action_history=action_history,
                    error="Action was not approved by user.",
                    output="Execution stopped: user rejected step.",
                )

        # ----------------------------------------------------------------------
        # Step e: Dispatch action to Miku's native Win32 primitives
        # ----------------------------------------------------------------------
        step_exec = dispatch_computer_use_action(
            action_dict=action_dict,
            real_execution=real_execution,
            target_hwnd=target_hwnd_override,
        )
        step_exec.step = step
        step_results.append(step_exec)

        # Record into history
        action_history.append({
            "step": step,
            "action": action_dict,
            "success": step_exec.success,
            "status": step_exec.status,
            "output": step_exec.output,
        })

        # ----------------------------------------------------------------------
        # Step f: Check for termination
        # ----------------------------------------------------------------------
        if action_dict.get("action") == "terminate":
            reason = action_dict.get("reason", "Objective achieved")
            return ComputerUseTaskResult(
                success=True,
                objective=objective,
                total_steps=step,
                final_status="TERMINATED",
                steps=step_results,
                action_history=action_history,
                output=f"Objective achieved in {step} steps. Reason: {reason}",
            )

        # Small stabilization delay between actions
        if delay_between_steps > 0:
            time.sleep(delay_between_steps)


__all__ = [
    "MAX_STEPS",
    "KEY_MAPPING",
    "resolve_virtual_key",
    "is_autonomous_authorized",
    "request_step_approval",
    "resolve_target_hwnd_at_point",
    "dispatch_computer_use_action",
    "run_computer_use_task",
    "ComputerUseStepResult",
    "ComputerUseTaskResult",
]
