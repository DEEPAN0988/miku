"""
Screen Inspection Module for Miku.
Captures desktop screenshots and queries Windows UI Automation tree
to locate active windows and UI elements reliably.
"""

import os
from typing import Dict, Any, Optional, List
from PIL import ImageGrab


class ScreenModule:
    @staticmethod
    def capture_screenshot(output_path: Optional[str] = None) -> Dict[str, Any]:
        """Grab current desktop image and optionally save to file."""
        try:
            image = ImageGrab.grab()
            width, height = image.size
            saved = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
                saved = output_path
            return {
                "success": True,
                "resolution": (width, height),
                "saved_path": saved
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def inspect_active_window() -> Dict[str, Any]:
        """Query currently active foreground window title and coordinates via UI Automation / Win32."""
        try:
            import uiautomation as auto
            focused = auto.GetFocusedControl()
            active_window = auto.GetForegroundControl()
            
            win_name = active_window.Name if active_window else "Unknown"
            win_class = active_window.ClassName if active_window else "Unknown"
            rect = active_window.BoundingRectangle if active_window else None
            
            focused_name = focused.Name if focused else "None"
            focused_type = focused.ControlTypeName if focused else "None"

            return {
                "success": True,
                "window_title": win_name,
                "window_class": win_class,
                "bounding_rect": [rect.left, rect.top, rect.right, rect.bottom] if rect else [],
                "focused_element": focused_name,
                "focused_type": focused_type,
                "summary": f"Active window: '{win_name}' ({win_class}), Focused control: '{focused_name}' [{focused_type}]"
            }
        except Exception:
            # Fallback if uiautomation fails
            try:
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return {
                    "success": True,
                    "window_title": title,
                    "summary": f"Active window title: '{title}'"
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
