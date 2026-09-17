"""
Unified Vision Engine for Miku (See capability).
Coordinates camera scene observation and screen UI inspection.
"""

from typing import Dict, Any, Optional
from .camera import CameraModule
from .screen import ScreenModule


class VisionEngine:
    def __init__(self, camera_index: int = 0):
        self.camera = CameraModule(camera_index)
        self.screen = ScreenModule()

    def see_room(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Capture room status via camera."""
        return self.camera.capture_frame(output_path=save_path)

    def see_screen(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Capture desktop screen and inspect foreground window."""
        scr = self.screen.capture_screenshot(output_path=save_path)
        win = self.screen.inspect_active_window()
        return {
            "success": scr.get("success", False),
            "screenshot": scr,
            "active_window": win,
            "summary": f"{win.get('summary', 'Screen captured')}."
        }
