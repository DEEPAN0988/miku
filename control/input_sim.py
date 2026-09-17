"""
Input Simulation Layer for Windows (Keyboard & Mouse).
Controls mouse cursor, clicks (LMB/RMB/MMB), and text typing.
Uses PyAutoGUI / ctypes Win32 mouse_event and keybd_event.
"""

import time
from typing import Dict, Any, Tuple, Optional
import pyautogui

# Set safety pauses
pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True


class InputSimulator:
    @classmethod
    def move_mouse(cls, x: int, y: int, duration: float = 0.2) -> Dict[str, Any]:
        """Move cursor to screen coordinates (x, y)."""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return {"success": True, "position": (x, y)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @classmethod
    def click_mouse(cls, button: str = "left", clicks: int = 1, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        """
        Click mouse button: 'left' (LMB), 'right' (RMB), or 'middle' (MMB).
        """
        try:
            btn = button.lower()
            if btn not in ["left", "right", "middle"]:
                btn = "left"
            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, clicks=clicks, button=btn)
            else:
                pyautogui.click(clicks=clicks, button=btn)
            return {"success": True, "button": btn, "clicks": clicks}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @classmethod
    def type_text(cls, text: str, interval: float = 0.02) -> Dict[str, Any]:
        """Type characters sequentially into the currently active window."""
        try:
            pyautogui.write(text, interval=interval)
            return {"success": True, "length": len(text)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @classmethod
    def press_key(cls, key: str) -> Dict[str, Any]:
        """Press a special key (e.g. 'enter', 'tab', 'esc', 'backspace')."""
        try:
            pyautogui.press(key)
            return {"success": True, "key": key}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @classmethod
    def hotkey(cls, *keys) -> Dict[str, Any]:
        """Send keyboard combination (e.g. 'ctrl', 's' or 'alt', 'f4')."""
        try:
            pyautogui.hotkey(*keys)
            return {"success": True, "keys": keys}
        except Exception as e:
            return {"success": False, "error": str(e)}
