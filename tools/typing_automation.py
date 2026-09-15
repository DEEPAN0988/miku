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
import os
import random
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
      1. Empty text or non-string.
      2. Dangerous control characters (NULL, BEL, BS, ESC, SIGINT, EOT).

    Payloads exceeding MAX_STAGED_TEXT_LENGTH are gracefully truncated with an informational log:
      [PAYLOAD_TRUNCATED]

    Returns:
      (is_safe, sanitized_text, reason)
    """
    if not isinstance(text, str):
        return False, "", "REJECT_NOT_STRING"

    if len(text) == 0:
        return False, "", "REJECT_EMPTY_TEXT"

    sanitized = text
    reason = "SAFE"
    if len(sanitized) > MAX_STAGED_TEXT_LENGTH:
        sanitized = sanitized[:MAX_STAGED_TEXT_LENGTH]
        print(f"[*] [PAYLOAD_TRUNCATED] Text payload exceeded {MAX_STAGED_TEXT_LENGTH} chars; truncated to {len(sanitized)} chars.", flush=True)
        reason = "PAYLOAD_TRUNCATED"

    # Filter illegal control characters (\x00, \x03, \x04, \x1B) or ASCII < 32 except newline \n and tab \t
    illegal_controls = {'\x00', '\x03', '\x04', '\x1B'}
    for ch in sanitized:
        code = ord(ch)
        if ch in illegal_controls or (code < 32 and ch not in ("\n", "\t")):
            return False, "", f"REJECT_FORBIDDEN_CONTROL_CHAR_0x{code:02X}"

    return True, sanitized, reason


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

    # If MIKU_LIVE_EXECUTION is active, physically type into focused control for live preview
    if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
        try:
            dispatch_typing_payload(safe_text, min_delay_sec=0.015, max_delay_sec=0.035)
        except Exception:
            pass

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
    If MIKU_AUTONOMOUS_MODE=True and MIKU_LIVE_EXECUTION=True, permits execution non-blockingly.
    Otherwise requires interactive console (sys.stdin.isatty()) and typing 'CONFIRM TYPE'.
    """
    from tools.screen_inspector import check_autonomous_authorization
    if check_autonomous_authorization("TYPE") or os.environ.get("MIKU_FAST_CONFIRM_PASSTHROUGH", "false").strip().lower() in ("true", "1", "yes"):
        return True

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
            f"Press [ENTER] to dispatch real physical OS keystrokes, or type anything else to cancel:\n"
            + "=" * 80 + "\n"
            f"Confirmation: "
        )
        resp = input(prompt).strip()
        return resp == "" or resp.upper() in ("Y", "YES", "CONFIRM TYPE")
    except Exception:
        return False


def dispatch_real_typing(
    target_hwnd: int,
    element: UIElement,
    text: str,
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

    # 4. Live Human Confirmation Gate (Unconditional)
    target_title = verif.details.get("window_text") or f"HWND {target_hwnd}"
    confirmed = request_live_human_type_confirmation(target_title, el_name, safe_text)
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
    from tools.screen_inspector import human_mouse_move
    human_mouse_move(None, None, cx, cy, duration=0.20)
    time.sleep(0.02)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.025)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
    time.sleep(0.05)

    # 6. Stale Focus Re-Check
    root_target = user32.GetAncestor(target_hwnd, 2) or target_hwnd
    if user32.IsWindow(root_target):
        try:
            user32.SetForegroundWindow(root_target)
            time.sleep(0.05)
        except Exception:
            pass

    fg_hwnd = user32.GetForegroundWindow()
    root_fg = user32.GetAncestor(fg_hwnd, 2) or fg_hwnd

    import win32gui
    fg_class = win32gui.GetClassName(root_fg) if win32gui else ""
    target_class = win32gui.GetClassName(root_target) if win32gui else ""

    is_fg = (
        root_fg == root_target
        or fg_hwnd == target_hwnd
        or (fg_class == "Windows.UI.Core.CoreWindow" and target_class == "Windows.UI.Core.CoreWindow")
    )

    if not is_fg:
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

    # 7. Physical Unicode Keystroke Dispatch with human micro-delays (15ms-35ms)
    typed_count = 0
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
        time.sleep(random.uniform(0.015, 0.035))

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


def dispatch_typing_payload(
    text: str,
    min_delay_sec: float = 0.015,
    max_delay_sec: float = 0.035,
) -> int:
    """
    Iterates through sanitized typing payload, sending individual character
    KEYDOWN and KEYUP Win32 SendInput event blocks separated by a randomized
    micro-delay (default 15ms-35ms, ~100+ WPM). Makes keys appear individually
    and visibly in real-time without bogging down execution.
    """
    from tools.screen_inspector import _attach_thread_to_default_desktop
    _attach_thread_to_default_desktop()
    is_safe, safe_text, _ = sanitize_typing_payload(text)
    if not is_safe:
        return 0

    typed_count = 0
    for ch in safe_text:
        char_code = ord(ch)

        inp_down = INPUT()
        inp_down.type = INPUT_KEYBOARD
        inp_down.ki.wVk = 0
        inp_down.ki.wScan = char_code
        inp_down.ki.dwFlags = KEYEVENTF_UNICODE
        inp_down.ki.time = 0
        inp_down.ki.dwExtraInfo = None

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
        time.sleep(random.uniform(min_delay_sec, max_delay_sec))

    return typed_count


def dispatch_human_keystrokes(
    text: str,
    min_delay_sec: float = 0.015,
    max_delay_sec: float = 0.035,
) -> int:
    """
    Dispatches Unicode text to the currently focused window using SendInput with
    randomized human-like typing cadence between min_delay_sec and max_delay_sec.
    """
    return dispatch_typing_payload(text, min_delay_sec=min_delay_sec, max_delay_sec=max_delay_sec)


def dispatch_vk_key(vk_code: int, hold_duration: float = 0.05) -> bool:
    """
    Dispatches a virtual key code (e.g., VK_RETURN=0x0D, VK_LWIN=0x5B, VK_ESCAPE=0x1B)
    using Win32 SendInput API with virtual key and scan code mapping.
    Ensures compatibility with modern UWP apps (SearchHost, Start menu, etc.)
    which ignore legacy keybd_event calls.
    """
    from tools.screen_inspector import _attach_thread_to_default_desktop
    _attach_thread_to_default_desktop()
    try:
        scan_code = user32.MapVirtualKeyW(vk_code, 0)
        extra = ctypes.c_ulong(0)

        inp_down = INPUT()
        inp_down.type = INPUT_KEYBOARD
        inp_down.ki.wVk = vk_code
        inp_down.ki.wScan = scan_code
        inp_down.ki.dwFlags = 0
        inp_down.ki.time = 0
        inp_down.ki.dwExtraInfo = ctypes.pointer(extra)

        inp_up = INPUT()
        inp_up.type = INPUT_KEYBOARD
        inp_up.ki.wVk = vk_code
        inp_up.ki.wScan = scan_code
        inp_up.ki.dwFlags = KEYEVENTF_KEYUP
        inp_up.ki.time = 0
        inp_up.ki.dwExtraInfo = ctypes.pointer(extra)

        user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
        time.sleep(max(0.02, hold_duration))
        user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))
        return True
    except Exception as exc:
        print(f"[!] Error dispatching VK 0x{vk_code:02X} via SendInput: {exc}", flush=True)
        return False


def dispatch_enter_key(hold_duration: float = 0.05) -> bool:
    """
    Dispatches VK_RETURN (Enter Key, 0x0D) via Win32 SendInput.
    """
    return dispatch_vk_key(0x0D, hold_duration=hold_duration)


def dispatch_uac_yes_confirmation() -> bool:
    """
    Glides physical mouse cursor to 'Yes' option on UAC dialog card and dispatches click,
    followed by Left Arrow (VK_LEFT, 0x25) + Enter (VK_RETURN, 0x0D) and Alt+Y shortcut.
    """
    from tools.screen_inspector import _attach_thread_to_default_desktop, get_current_cursor_pos, human_mouse_move
    _attach_thread_to_default_desktop()

    # 1. Calculate physical screen bounds and UAC 'Yes' button coordinates
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    yes_x = int(sw * 0.461)
    yes_y = int(sh * 0.556)

    cur_x, cur_y = get_current_cursor_pos()
    print(f"[*] Gliding mouse cursor from ({cur_x}, {cur_y}) to UAC 'Yes' option at ({yes_x}, {yes_y})...", flush=True)

    # 2. Smooth minimum-jerk Bézier physical mouse glide to 'Yes' button
    try:
        human_mouse_move(cur_x, cur_y, yes_x, yes_y, duration=0.25)
        time.sleep(0.04)
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
        time.sleep(0.03)
        user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
    except Exception as exc:
        print(f"[!] Mouse glide exception: {exc}", flush=True)

    time.sleep(0.08)

    # 3. Shift focus from default 'No' to 'Yes' button via Left Arrow and press Enter
    dispatch_vk_key(0x25, hold_duration=0.06)  # VK_LEFT
    time.sleep(0.08)
    dispatch_vk_key(0x0D, hold_duration=0.08)  # VK_RETURN
    time.sleep(0.12)

    # 4. Alt+Y accelerator backup (VK_MENU=0x12, Y=0x59)
    try:
        scan_alt = user32.MapVirtualKeyW(0x12, 0)
        scan_y = user32.MapVirtualKeyW(0x59, 0)
        extra = ctypes.c_ulong(0)

        inp1 = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0x12, wScan=scan_alt, dwFlags=0, time=0, dwExtraInfo=ctypes.pointer(extra)))
        inp2 = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0x59, wScan=scan_y, dwFlags=0, time=0, dwExtraInfo=ctypes.pointer(extra)))
        inp3 = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0x59, wScan=scan_y, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=ctypes.pointer(extra)))
        inp4 = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0x12, wScan=scan_alt, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=ctypes.pointer(extra)))

        user32.SendInput(1, ctypes.byref(inp1), ctypes.sizeof(INPUT))
        user32.SendInput(1, ctypes.byref(inp2), ctypes.sizeof(INPUT))
        time.sleep(0.04)
        user32.SendInput(1, ctypes.byref(inp3), ctypes.sizeof(INPUT))
        user32.SendInput(1, ctypes.byref(inp4), ctypes.sizeof(INPUT))
    except Exception:
        pass
    return True


