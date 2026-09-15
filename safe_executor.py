"""
safe_executor.py — Action Dispatcher with Token Authorization Gate

STRICT SAFETY GATE:
Every physical hardware action method (click, type) MUST demand a valid, approved,
and unused AuthToken issued by FastConfirm. Unauthorized or replayed actions are rejected.
"""

import ctypes
import time
from typing import Any, Dict, Optional, Tuple

from human_gate import AuthToken

try:
    from tools.screen_inspector import dispatch_real_click
    HAVE_REAL_CLICK = True
except ImportError:
    HAVE_REAL_CLICK = False

try:
    from tools.typing_automation import dispatch_real_typing
    HAVE_REAL_TYPING = True
except ImportError:
    HAVE_REAL_TYPING = False


class UnauthorizedActionError(Exception):
    """Raised when an attempt is made to execute a physical action without a valid AuthToken."""
    pass


class ActionDispatch:
    """
    Physical hardware action executor guarded by FastConfirm AuthToken validation.
    """

    def __init__(self, simulation_mode: bool = False):
        """
        Initialize ActionDispatch.
        
        :param simulation_mode: If True, physical OS hardware events are logged/simulated
                                without moving physical peripherals (ideal for unit testing).
        """
        self.simulation_mode = simulation_mode
        self.executed_actions: list = []

    def _validate_token(self, token: AuthToken, expected_action_type: str) -> None:
        """Validates token authenticity, approval state, and single-use constraint."""
        if not isinstance(token, AuthToken):
            raise UnauthorizedActionError(
                "Execution rejected: Parameter is not a valid AuthToken object."
            )
        if not token.approved:
            raise UnauthorizedActionError(
                f"Execution rejected: AuthToken '{token.token_id}' was DENIED or NOT approved."
            )
        if token.is_used:
            raise UnauthorizedActionError(
                f"Execution rejected: AuthToken '{token.token_id}' has already been USED (token replay attempt)."
            )
        if token.action_type != expected_action_type:
            raise UnauthorizedActionError(
                f"Execution rejected: AuthToken action type mismatch (Expected '{expected_action_type}', got '{token.action_type}')."
            )

    def click(self, token: AuthToken) -> Dict[str, Any]:
        """
        Executes a physical click at the coordinates specified in the AuthToken.
        
        MUST receive a valid, approved, unused AuthToken.
        """
        self._validate_token(token, expected_action_type="click")

        # Mark token as consumed to prevent replay attacks
        token.mark_used()

        x, y = token.coords
        target = token.target

        if self.simulation_mode:
            record = {
                "action": "click",
                "target": target,
                "coords": (x, y),
                "simulated": True,
                "token_id": token.token_id,
                "timestamp": time.time(),
            }
            self.executed_actions.append(record)
            print(f"[SAFE EXECUTOR] [SIMULATED] Click executed at X:{x}, Y:{y} for '{target}'.")
            return record

        # Real OS Hardware Execution
        if HAVE_REAL_CLICK:
            # Dispatch physical click via tools.screen_inspector
            success, msg = dispatch_real_click(x, y)
            record = {
                "action": "click",
                "target": target,
                "coords": (x, y),
                "success": success,
                "message": msg,
                "token_id": token.token_id,
                "timestamp": time.time(),
            }
            self.executed_actions.append(record)
            print(f"[SAFE EXECUTOR] Physical click dispatched at X:{x}, Y:{y} -> Success: {success}")
            return record
        else:
            # Win32 user32 SetCursorPos + mouse_event fallback
            ctypes.windll.user32.SetCursorPos(x, y)
            time.sleep(0.05)
            # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
            time.sleep(0.05)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            record = {
                "action": "click",
                "target": target,
                "coords": (x, y),
                "success": True,
                "token_id": token.token_id,
                "timestamp": time.time(),
            }
            self.executed_actions.append(record)
            print(f"[SAFE EXECUTOR] Physical Win32 click dispatched at X:{x}, Y:{y}.")
            return record

    def type(self, token: AuthToken) -> Dict[str, Any]:
        """
        Executes physical text typing at the specified coordinates or active edit control.
        
        MUST receive a valid, approved, unused AuthToken.
        """
        self._validate_token(token, expected_action_type="type")

        # Mark token as consumed
        token.mark_used()

        x, y = token.coords
        target = token.target
        payload = token.payload or ""

        if self.simulation_mode:
            record = {
                "action": "type",
                "target": target,
                "coords": (x, y),
                "payload": payload,
                "simulated": True,
                "token_id": token.token_id,
                "timestamp": time.time(),
            }
            self.executed_actions.append(record)
            print(f"[SAFE EXECUTOR] [SIMULATED] Type '{payload}' executed into '{target}' at X:{x}, Y:{y}.")
            return record

        # Real OS Hardware Typing Execution
        if HAVE_REAL_TYPING:
            success, msg = dispatch_real_typing(payload, delay_between_keys=0.02)
            record = {
                "action": "type",
                "target": target,
                "coords": (x, y),
                "payload": payload,
                "success": success,
                "message": msg,
                "token_id": token.token_id,
                "timestamp": time.time(),
            }
            self.executed_actions.append(record)
            print(f"[SAFE EXECUTOR] Physical text dispatched -> Success: {success}")
            return record
        else:
            # Fallback typing dispatch
            import pywintypes
            import win32api
            import win32con
            for char in payload:
                vk = win32api.VkKeyScan(char) & 0xFF
                win32api.keybd_event(vk, 0, 0, 0)
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.02)

            record = {
                "action": "type",
                "target": target,
                "coords": (x, y),
                "payload": payload,
                "success": True,
                "token_id": token.token_id,
                "timestamp": time.time(),
            }
            self.executed_actions.append(record)
            print(f"[SAFE EXECUTOR] Win32 text dispatched for '{target}'.")
            return record
