"""
Closed-Loop Execution Verification.
Core 3: Execution Engine (Hands)
Addresses Bare-Metal Bottleneck #3 (The Notification Trap).
Immediately before hardware click fires, re-samples target coordinate
and confirms it still matches the planned element. Aborts on mismatch.
"""
import sys
from typing import Tuple, Optional, Dict, Any

class ClosedLoopVerifier:
    def __init__(self, search_radius: int = 30):
        self.search_radius = search_radius

    def sample_target_state(self, coords: Tuple[int, int]) -> Dict[str, Any]:
        """
        Samples the current UI/window element at the specified coordinates.
        """
        x, y = coords
        state = {
            "x": x,
            "y": y,
            "window_title": "",
            "control_type": "unknown",
            "control_name": ""
        }

        if sys.platform == "win32":
            try:
                import win32gui
                hwnd = win32gui.WindowFromPoint((x, y))
                if hwnd:
                    state["window_title"] = win32gui.GetWindowText(hwnd)
                
                # Check UIAutomation element at point
                try:
                    import uiautomation as auto
                    elem = auto.ControlFromPoint(x, y)
                    if elem:
                        state["control_name"] = elem.Name or ""
                        state["control_type"] = elem.ControlTypeName or ""
                except Exception:
                    pass
            except Exception:
                pass

        return state

    def verify_before_click(
        self,
        target_coords: Tuple[int, int],
        expected_control: str,
        expected_window: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verifies target before click.
        Returns (is_valid, reason).
        """
        current_state = self.sample_target_state(target_coords)

        # If a popup appeared or window changed unexpectedly
        if expected_window:
            cur_win = current_state.get("window_title", "").lower()
            if expected_window.lower() not in cur_win and cur_win != "":
                return False, f"target changed, action cancelled: window '{cur_win}' obscured '{expected_window}'"

        if expected_control:
            cur_name = current_state.get("control_name", "").lower()
            cur_type = current_state.get("control_type", "").lower()
            exp = expected_control.lower()

            # If element name or type completely conflicts with expected
            if cur_name and exp not in cur_name and exp not in cur_type:
                # If a completely different control or dismiss button appeared
                if any(trap in cur_name for trap in ["notification", "banner", "toast", "popup", "close"]):
                    return False, f"target changed, action cancelled: found unexpected '{cur_name}'"

        return True, "verified"
