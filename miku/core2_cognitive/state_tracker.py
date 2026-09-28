"""
OS Accessibility Tree & Window State Tracker.
Core 2: Cognitive Router (Brain & Memory)
Maintains a live snapshot of active windows and UI controls.
"""
import time
import sys
from typing import Dict, Any, List, Optional

class OSStateTracker:
    def __init__(self):
        self.last_snapshot_time = 0.0
        self.cached_state: Dict[str, Any] = {
            "active_title": "",
            "active_process": "",
            "window_rect": (0, 0, 1920, 1080),
            "visible_controls": [],
            "timestamp": time.time()
        }

    def capture_active_state(self) -> Dict[str, Any]:
        """
        Captures current foreground window info and UI elements on Windows.
        Returns state dictionary.
        """
        now = time.time()
        # Rate limit to 10Hz to save CPU
        if now - self.last_snapshot_time < 0.1:
            return self.cached_state

        title = "Desktop"
        process = "explorer.exe"
        rect = (0, 0, 1920, 1080)
        controls: List[Dict[str, Any]] = []

        if sys.platform == "win32":
            try:
                import win32gui
                import win32process
                import psutil

                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    title = win32gui.GetWindowText(hwnd)
                    rect = win32gui.GetWindowRect(hwnd)
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        p = psutil.Process(pid)
                        process = p.name()
                    except Exception:
                        process = "unknown.exe"

                # Check top controls via UIAutomation if available
                try:
                    import uiautomation as auto
                    fg_control = auto.GetFocusedControl()
                    if fg_control:
                        controls.append({
                            "name": fg_control.Name,
                            "type": fg_control.ControlTypeName,
                            "rect": (fg_control.BoundingRectangle.left,
                                     fg_control.BoundingRectangle.top,
                                     fg_control.BoundingRectangle.width(),
                                     fg_control.BoundingRectangle.height())
                        })
                except Exception:
                    pass
            except Exception:
                pass

        self.cached_state = {
            "active_title": title,
            "active_process": process,
            "window_rect": rect,
            "visible_controls": controls,
            "timestamp": now
        }
        self.last_snapshot_time = now
        return self.cached_state

    def score_state_relevance(self, action_type: str, state: Dict[str, Any]) -> float:
        """
        Heuristic affinity score S_state between current OS window and action.
        """
        active_title = state.get("active_title", "").lower()
        active_proc = state.get("active_process", "").lower()

        if action_type in ("open_app", "launch"):
            return 0.5  # Neutral affinity
        if action_type in ("close_app", "window_state") and active_title:
            return 0.9  # High affinity to current window
        if "notepad" in active_proc and "note" in action_type:
            return 1.0
        if "chrome" in active_proc and "browser" in action_type:
            return 1.0

        return 0.4
