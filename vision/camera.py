"""
Local Camera Capture & Scene Analysis for Miku.
Captures frames using OpenCV, analyzes scene brightness, motion, and presence.
Zero cloud API, zero external black-box models.
"""

import os
from typing import Dict, Any, Optional
import cv2
import numpy as np


class CameraModule:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

    def capture_frame(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Capture a single frame from the local webcam.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Unable to open camera index {self.camera_index} (no camera connected or in use)."
            }

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {
                "success": False,
                "error": "Camera frame capture failed."
            }

        height, width, channels = frame.shape
        avg_brightness = float(np.mean(frame))

        saved_path = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            saved_path = output_path

        # Basic light/scene classification
        lighting = "well lit" if avg_brightness > 80 else ("dim" if avg_brightness > 30 else "dark")

        return {
            "success": True,
            "width": width,
            "height": height,
            "avg_brightness": avg_brightness,
            "lighting": lighting,
            "saved_path": saved_path,
            "summary": f"Camera captured {width}x{height} frame. Scene appears {lighting} (mean brightness {avg_brightness:.1f})."
        }
