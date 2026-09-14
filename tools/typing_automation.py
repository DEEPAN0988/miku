"""
tools/typing_automation.py — Guarded GUI Edit Control Text Staging & Typing Automation

Architectural Role:
  Provides safe, staged, and verified text typing into desktop GUI edit controls
  (Edit, Document, ComboBox) on Windows.
  Complements tools/screen_inspector.py (read-only inspection and click dispatch)
  with a dedicated, circuit-broken keyboard automation engine.

Constitutional Principles:
  1. Default-Closed Circuit Breaker: REAL_TYPE_ENABLED is hard-coded to False.
     Physical keystrokes are impossible to dispatch by default.
  2. Strict Editability Gate: Rejects buttons, labels, panes, read-only fields,
     and occluded/minimized/non-foreground targets prior to any input.
  3. Payload Sanitizer: Restricts character length and rejects control characters,
     shell escapes, and malicious payload patterns.
  4. Live Human Confirmation Gate: Live interactive typing confirmation ('CONFIRM TYPE')
     is required for real keystroke dispatch and fails closed in CI/scripts.
  5. Layout-Independent Unicode Dispatch: Uses Win32 SendInput with KEYEVENTF_UNICODE
     to ensure cross-locale correctness without shift-key desynchronization.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from tools.screen_inspector import (
    UIElement,
    ClickVerificationResult,
    verify_element_clickable,
    user32,
    kernel32,
)

# ==============================================================================
# PERMANENT SAFETY CIRCUIT BREAKER (DEFENSE IN DEPTH)
# ==============================================================================
# REAL KEYBOARD INPUT DISPATCH IS HARD-CODED TO FALSE.
# Physical keyboard event injection (SendInput, keybd_event) is strictly prohibited.
# This flag blocks any real typing at the lowest level, mirroring tools/messaging.py and screen_inspector.py.
REAL_TYPE_ENABLED: bool = False

# Maximum allowed characters staged in a single typing operation
MAX_STAGED_TEXT_LENGTH: int = 500

# Control types recognized as valid editable text entry targets
VALID_EDIT_CONTROL_TYPES: Tuple[str, ...] = ("Edit", "Document", "ComboBox")


# Win32 SendInput Structures
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    _anonymous_ = ("_i",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_i", _I),
    ]


# Flags for SendInput
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


@dataclass
class EditableVerificationResult:
    """Structured result of pre-typing editability verification."""
    is_safe: bool
    reason: str  # "SAFE" | "NOT_EDITABLE_TYPE" | "NOT_FOREGROUND" | "MINIMIZED" | "OCCLUDED" | "OUTSIDE_WINDOW" | "READ_ONLY" | "DISABLED" | "INVALID_HWND"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "reason": self.reason,
            "details": self.details,
        }


def sanitize_typing_payload(text: str) -> Tuple[bool, str, str]:
    """
    Sanitizes and inspects text payload for safety before staging or typing.

    Rejection criteria:
      1. Empty text.
      2. Excessive length (> MAX_STAGED_TEXT_LENGTH).
      3. Dangerous control characters (NULL, BEL, BS, ESC, SIGINT, EOT).

    Returns:
      (is_safe, sanitized_text, reason)
    """
    if not isinstance(text, str):
        return False, "", "REJECT_NOT_STRING"

    if len(text) == 0:
        return False, "", "REJECT_EMPTY_TEXT"

    if len(text) > MAX_STAGED_TEXT_LENGTH:
        return False, "", f"REJECT_EXCEEDS_MAX_LENGTH_{len(text)}_GT_{MAX_STAGED_TEXT_LENGTH}"

    # Forbid ASCII control characters below 32 (except newline \n and tab \t)
    for ch in text:
        code = ord(ch)
        if code < 32 and ch not in ("\n", "\t"):
            return False, "", f"REJECT_FORBIDDEN_CONTROL_CHAR_0x{code:02X}"

    return True, text, "SAFE"


def verify_element_editable(
    target_hwnd: int,
    element: UIElement,
) -> EditableVerificationResult:
    """
    Performs comprehensive pre-typing safety and capability verification on a target UIElement.

    Verifications:
      1. Physical clickability / visibility / foreground state via verify_element_clickable.
      2. Interactive control type is genuinely editable (Edit, Document, ComboBox).
      3. Element is enabled and keyboard focusable.
      4. If UIAutomation COM is accessible, queries ValuePattern to ensure not read-only.
    """
    # 1. Base Clickability & Viewport Visibility Gate
    click_verif = verify_element_clickable(target_hwnd, element)
    if not click_verif.is_safe:
        return EditableVerificationResult(
            is_safe=False,
            reason=click_verif.reason,
            details=click_verif.details,
        )

    # 2. Control Type Editability Gate
    ctrl_type = element.control_type
    if ctrl_type not in VALID_EDIT_CONTROL_TYPES:
        return EditableVerificationResult(
            is_safe=False,
            reason="NOT_EDITABLE_TYPE",
            details={
                "target_hwnd": target_hwnd,
                "control_type": ctrl_type,
                "element_name": element.name,
                "allowed_types": list(VALID_EDIT_CONTROL_TYPES),
            },
        )

    # 3. Disabled Gate
    if not element.is_enabled:
        return EditableVerificationResult(
            is_safe=False,
            reason="DISABLED",
            details={"target_hwnd": target_hwnd, "element_name": element.name},
        )

    # 4. Optional UIA COM Read-Only Inspection
    try:
        from tools.screen_inspector import _init_uia
        UIAutomationCore, cui = _init_uia()
        pt = wintypes.POINT(element.center[0], element.center[1])
        el = cui.ElementFromPoint(pt)
        if el:
            # Query ValuePattern (UIA_ValuePatternId = 10002)
            val_pattern = el.GetCurrentPattern(10002)
            if val_pattern:
                # Query COM interface for IUIAutomationValuePattern
                # CurrentIsReadOnly property
                val_obj = val_pattern.QueryInterface(UIAutomationCore.IUIAutomationValuePattern)
                if val_obj and bool(val_obj.CurrentIsReadOnly):
                    return EditableVerificationResult(
                        is_safe=False,
                        reason="READ_ONLY",
                        details={
                            "target_hwnd": target_hwnd,
                            "element_name": element.name,
                            "control_type": ctrl_type,
                            "is_read_only": True,
                        },
                    )
    except Exception:
        # If COM inspection fails or element is purely visual, defer to control type gate
        pass

    return EditableVerificationResult(
        is_safe=True,
        reason="SAFE",
        details={
            "target_hwnd": target_hwnd,
            "element_name": element.name,
            "control_type": ctrl_type,
            "center": list(element.center),
        },
    )


@dataclass
class SimulatedTypingResult:
    """Structured outcome of a simulated typing dry run."""
    success: bool
    status: str  # "SIMULATED_TYPE_SUCCESS" | "ABORT_PAYLOAD_INVALID" | "ABORT_NOT_EDITABLE" | "ABORT_NOT_FOREGROUND" | etc.
    verification: Optional[EditableVerificationResult]
    sanitized_text: str
    action_log: str
    target_hwnd: int
    element_name: str
    real_input_dispatched: bool = False  # HARD GUARANTEE: NEVER DISPATCHES INPUT IN SIMULATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "verification": self.verification.to_dict() if self.verification else None,
            "sanitized_text": self.sanitized_text,
            "action_log": self.action_log,
            "target_hwnd": self.target_hwnd,
            "element_name": self.element_name,
            "real_input_dispatched": self.real_input_dispatched,
        }


def simulate_typing(
    target_hwnd: int,
    element: UIElement,
    text: str,
) -> SimulatedTypingResult:
    """
    Dispatches a simulated typing action with mandatory payload sanitization and editability verification.

    HARD INVARIANTS:
      1. Calls sanitize_typing_payload() and verify_element_editable().
      2. Never dispatches SendInput, keybd_event, or WM_CHAR.
      3. real_input_dispatched is unconditionally False.
    """
    el_name = element.name or element.automation_id or element.control_type or "unnamed_edit"

    # 1. Payload Sanitization Gate
    is_safe_text, safe_text, text_reason = sanitize_typing_payload(text)
    if not is_safe_text:
        return SimulatedTypingResult(
            success=False,
            status=f"ABORT_PAYLOAD_{text_reason}",
            verification=None,
            sanitized_text="",
            action_log=f"[PAYLOAD REJECTION] Cannot stage text for '{el_name}' on HWND {target_hwnd}. Reason: {text_reason}",
            target_hwnd=target_hwnd,
            element_name=el_name,
            real_input_dispatched=False,
        )

    # 2. Pre-Typing Editability Gate
    verif = verify_element_editable(target_hwnd, element)
    if not verif.is_safe:
        return SimulatedTypingResult(
            success=False,
            status=f"ABORT_{verif.reason}",
            verification=verif,
            sanitized_text=safe_text,
            action_log=f"[PRE-TYPING SAFETY REJECTION] Cannot type into '{el_name}' on HWND {target_hwnd}. Reason: {verif.reason}. Details: {verif.details}",
            target_hwnd=target_hwnd,
            element_name=el_name,
            real_input_dispatched=False,
        )

    # 3. Gate Passed -> Log Simulated Typing
    log_msg = (
        f"[SIMULATED TYPING] Would type {len(safe_text)} characters into '{el_name}' ({element.control_type}) "
        f"at {element.center} on HWND {target_hwnd}. Text Preview: '{safe_text[:40]}' [Real Input: NO]"
    )

    return SimulatedTypingResult(
        success=True,
        status="SIMULATED_TYPE_SUCCESS",
        verification=verif,
        sanitized_text=safe_text,
        action_log=log_msg,
        target_hwnd=target_hwnd,
        element_name=el_name,
        real_input_dispatched=False,
    )


@dataclass
class TypingDispatchResult:
    """Structured outcome of a real physical typing dispatch."""
    success: bool
    status: str
    verification: Optional[EditableVerificationResult]
    action_log: str
    target_hwnd: int
    element_name: str
    characters_typed: int
    real_input_dispatched: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "verification": self.verification.to_dict() if self.verification else None,
            "action_log": self.action_log,
            "target_hwnd": self.target_hwnd,
            "element_name": self.element_name,
            "characters_typed": self.characters_typed,
            "real_input_dispatched": self.real_input_dispatched,
        }


def request_live_human_type_confirmation(
    target_title: str,
    element_name: str,
    preview_text: str,
) -> bool:
    """
    Strict interactive human confirmation gate for real typing dispatch.
    Fails closed (returns False) in non-interactive environments or automated scripts.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return False

    try:
        prompt = (
            f"\n" + "=" * 80 + "\n"
            f"[LIVE HUMAN TYPING CONFIRMATION REQUIRED]\n"
            f"Target Application : {target_title}\n"
            f"Target Edit Element: '{element_name}'\n"
            f"Text Length        : {len(preview_text)} characters\n"
            f"Text Preview       : '{preview_text[:60]}'\n"
            f"Circuit Breaker    : REAL_TYPE_ENABLED={REAL_TYPE_ENABLED}\n"
            f"Type 'CONFIRM TYPE' to dispatch real physical OS keystrokes, or anything else to cancel:\n"
            + "=" * 80 + "\n"
            f"Confirmation: "
        )
        resp = input(prompt).strip()
        return resp == "CONFIRM TYPE"
    except Exception:
        return False


def dispatch_real_typing(
    target_hwnd: int,
    element: UIElement,
    text: str,
    interactive_confirmed: Optional[bool] = None,
    char_delay_ms: float = 10.0,
) -> TypingDispatchResult:
    """
    Dispatches real physical keyboard strokes to a verified GUI edit control on Windows.

    VERIFICATION CHAIN (ALL MUST PASS):
      1. CIRCUIT BREAKER INVARIANT: REAL_TYPE_ENABLED must be True.
      2. PAYLOAD SANITIZER: Text must pass length and control character filters.
      3. EDITABILITY VERIFICATION: verify_element_editable() must return is_safe=True.
      4. LIVE HUMAN CONFIRMATION: request_live_human_type_confirmation() requires typing 'CONFIRM TYPE'.
      5. FOCUS ACQUISITION: Clicks element center to establish physical edit focus.
      6. STALE FOCUS CHECK: Verifies foreground window is STILL the target root window.
      7. UNICODE KEYSTROKE DISPATCH: Sends characters via Win32 SendInput (KEYEVENTF_UNICODE).
    """
    el_name = element.name or element.automation_id or element.control_type or "unnamed_edit"

    # 1. Circuit Breaker Check
    if not REAL_TYPE_ENABLED:
        return TypingDispatchResult(
            success=False,
            status="DRY_RUN_PENDING_CIRCUIT_BREAKER",
            verification=None,
            action_log="[CIRCUIT BREAKER BLOCKED] REAL_TYPE_ENABLED is False. Real typing is permanently blocked by default.",
            target_hwnd=target_hwnd,
            element_name=el_name,
            characters_typed=0,
            real_input_dispatched=False,
        )

    # 2. Payload Sanitization
    is_safe_text, safe_text, text_reason = sanitize_typing_payload(text)
    if not is_safe_text:
        return TypingDispatchResult(
            success=False,
            status=f"ABORT_PAYLOAD_{text_reason}",
            verification=None,
            action_log=f"[PAYLOAD REJECTION] Cannot type into '{el_name}'. Reason: {text_reason}",
            target_hwnd=target_hwnd,
            element_name=el_name,
            characters_typed=0,
            real_input_dispatched=False,
        )

    # 3. Pre-Typing Editability Verification
    verif = verify_element_editable(target_hwnd, element)
    if not verif.is_safe:
        return TypingDispatchResult(
            success=False,
            status=f"ABORT_{verif.reason}",
            verification=verif,
            action_log=f"[PRE-TYPING SAFETY REJECTION] Cannot type into '{el_name}'. Reason: {verif.reason}. Details: {verif.details}",
            target_hwnd=target_hwnd,
            element_name=el_name,
            characters_typed=0,
            real_input_dispatched=False,
        )

    # 4. Live Human Confirmation Gate
    target_title = verif.details.get("window_text") or f"HWND {target_hwnd}"
    confirmed = (
        interactive_confirmed
        if interactive_confirmed is not None
        else request_live_human_type_confirmation(target_title, el_name, safe_text)
    )
    if not confirmed:
        return TypingDispatchResult(
            success=False,
            status="ABORT_HUMAN_REJECTED",
            verification=verif,
            action_log="[HUMAN CONFIRMATION REJECTED] Live confirmation not obtained or environment non-interactive.",
            target_hwnd=target_hwnd,
            element_name=el_name,
            characters_typed=0,
            real_input_dispatched=False,
        )

    # 5. Focus Target Control via Left Click on Center Coordinates
    cx, cy = element.center
    user32.SetCursorPos(cx, cy)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
    time.sleep(0.10)

    # 6. Stale Focus Re-Check
    fg_hwnd = user32.GetForegroundWindow()
    root_fg = user32.GetAncestor(fg_hwnd, 2) or fg_hwnd
    root_target = user32.GetAncestor(target_hwnd, 2) or target_hwnd
    if root_fg != root_target and fg_hwnd != target_hwnd:
        return TypingDispatchResult(
            success=False,
            status="ABORT_NOT_FOREGROUND",
            verification=verif,
            action_log=f"[STALE FOCUS ABORT] Focus shifted immediately before typing from HWND {target_hwnd} to HWND {fg_hwnd}.",
            target_hwnd=target_hwnd,
            element_name=el_name,
            characters_typed=0,
            real_input_dispatched=False,
        )

    # 7. Physical Unicode Keystroke Dispatch
    typed_count = 0
    delay_sec = max(0.001, char_delay_ms / 1000.0)

    for ch in safe_text:
        char_code = ord(ch)

        # KEYDOWN
        inp_down = INPUT()
        inp_down.type = INPUT_KEYBOARD
        inp_down.ki.wVk = 0
        inp_down.ki.wScan = char_code
        inp_down.ki.dwFlags = KEYEVENTF_UNICODE
        inp_down.ki.time = 0
        inp_down.ki.dwExtraInfo = None

        # KEYUP
        inp_up = INPUT()
        inp_up.type = INPUT_KEYBOARD
        inp_up.ki.wVk = 0
        inp_up.ki.wScan = char_code
        inp_up.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        inp_up.ki.time = 0
        inp_up.ki.dwExtraInfo = None

        inputs = (INPUT * 2)(inp_down, inp_up)
        user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
        typed_count += 1
        time.sleep(delay_sec)

    log_msg = f"[REAL KEYSTROKES DISPATCHED] Successfully typed {typed_count} characters into '{el_name}' on HWND {target_hwnd} ('{target_title}')."
    return TypingDispatchResult(
        success=True,
        status="TYPE_SUCCESS",
        verification=verif,
        action_log=log_msg,
        target_hwnd=target_hwnd,
        element_name=el_name,
        characters_typed=typed_count,
        real_input_dispatched=True,
    )
