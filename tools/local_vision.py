"""
tools/local_vision.py — Miku Visual Grounding (From Scratch)

STATUS: PARTIAL — Vision encoder NOT YET TRAINED.

Current capabilities:
  - Pure coordinate math utilities (scale_and_clamp_point, parse output)
  - Desktop capture via Win32 GDI
  - Text-based screen state description (via tools.miku_inference)

NOT YET IMPLEMENTED:
  - Pixel-level visual element grounding (requires training a vision encoder
    from scratch and attaching it to MikuLM)

Previous state: This module loaded vikhyatk/moondream2 (a third-party pre-trained
HuggingFace model). That has been REMOVED to comply with the project constitution
(§3 No Fake AI, §6 From-Scratch, §25 No External Models).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

user32 = ctypes.windll.user32

# ---------------------------------------------------------------------------
# HAS_VISION_DEPS — kept for backward compatibility with miku.py imports
# Now refers to basic torch/PIL availability (for desktop capture only),
# NOT any pre-trained model.
# ---------------------------------------------------------------------------

HAS_VISION_DEPS: bool = True
VISION_DEPS_ERROR: Optional[str] = None

try:
    import torch
    from PIL import Image
except (ImportError, ModuleNotFoundError) as _err:
    HAS_VISION_DEPS = False
    VISION_DEPS_ERROR = str(_err)
    torch = None
    Image = None


def verify_vision_dependencies() -> bool:
    """Checks if torch and PIL are available for desktop capture."""
    if not HAS_VISION_DEPS:
        print("\n" + "=" * 70, flush=True)
        print("[!] Missing torch/PIL for desktop capture.", flush=True)
        if VISION_DEPS_ERROR:
            print(f"    Error detail: {VISION_DEPS_ERROR}", flush=True)
        print("[!] Please run: pip install torch pillow", flush=True)
        print("=" * 70 + "\n", flush=True)
        return False
    return True


def get_acceleration_device_and_dtype() -> Tuple[str, Any]:
    """
    Dynamically checks for hardware acceleration.
    Returns (device_str, dtype) for Miku's own model loading.
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32 if torch is not None else None


# ---------------------------------------------------------------------------
# Pure coordinate math (no model dependency)
# ---------------------------------------------------------------------------

def scale_and_clamp_point(
    norm_x: float,
    norm_y: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """
    Converts normalized coordinates [0.0, 1.0] to integer pixel coordinates.
    Handles both normalized floats and absolute pixel inputs.
    """
    if 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0:
        pixel_x = int(round(norm_x * image_width))
        pixel_y = int(round(norm_y * image_height))
    else:
        pixel_x = int(round(norm_x))
        pixel_y = int(round(norm_y))

    clamped_x = max(0, min(pixel_x, max(0, image_width - 1)))
    clamped_y = max(0, min(pixel_y, max(0, image_height - 1)))
    return clamped_x, clamped_y


def parse_moondream_point_output(
    output: Any,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int]]:
    """
    Parses coordinate output from various formats.
    Kept for backward compatibility with tests; the name is historical.
    """
    if not output:
        return None

    pt_candidate: Optional[Dict[str, Any]] = None

    if isinstance(output, dict):
        if "points" in output and isinstance(output["points"], list) and output["points"]:
            first = output["points"][0]
            if isinstance(first, dict):
                pt_candidate = first
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                return scale_and_clamp_point(float(first[0]), float(first[1]), image_width, image_height)
        elif "point" in output and isinstance(output["point"], dict):
            pt_candidate = output["point"]
        elif "x" in output and "y" in output:
            pt_candidate = output
    elif isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            pt_candidate = first
        elif isinstance(first, (list, tuple)) and len(first) >= 2:
            return scale_and_clamp_point(float(first[0]), float(first[1]), image_width, image_height)
    elif isinstance(output, tuple) and len(output) >= 2:
        return scale_and_clamp_point(float(output[0]), float(output[1]), image_width, image_height)

    if pt_candidate and "x" in pt_candidate and "y" in pt_candidate:
        try:
            nx = float(pt_candidate["x"])
            ny = float(pt_candidate["y"])
            return scale_and_clamp_point(nx, ny, image_width, image_height)
        except (ValueError, TypeError):
            return None

    return None


# ---------------------------------------------------------------------------
# Desktop capture (no model dependency — pure Win32 GDI)
# ---------------------------------------------------------------------------

def capture_desktop_pil() -> Optional[Any]:
    """
    Captures the physical desktop frame as a PIL.Image.Image.
    Uses Win32 GDI BitBlt via tools.vision_grounder.
    """
    if not verify_vision_dependencies():
        return None

    try:
        from tools.vision_grounder import capture_window_bitmap
        import cv2

        desktop_hwnd = user32.GetDesktopWindow()
        captured = capture_window_bitmap(desktop_hwnd)
        if captured is not None:
            img_bgr, _ = captured
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
    except Exception:
        pass

    try:
        from PIL import ImageGrab
        return ImageGrab.grab()
    except Exception as exc:
        print(f"[!] Desktop capture failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Visual grounding — NOT YET IMPLEMENTED
# ---------------------------------------------------------------------------

def get_action_coordinates(
    image: Any,
    target_element: str,
    model: Optional[Any] = None,
) -> Optional[Tuple[int, int]]:
    """
    [NOT YET IMPLEMENTED] Pixel-level visual element grounding.

    This previously used vikhyatk/moondream2 (a third-party pre-trained model).
    That has been removed to comply with the project constitution.

    To implement this from scratch, a vision encoder must be trained as part
    of MikuLM. Until then, use get_screen_state_text() from tools.miku_inference
    for text-based screen reasoning.
    """
    raise NotImplementedError(
        "[NOT YET IMPLEMENTED] Miku vision encoder not yet trained. "
        "Pixel-level visual grounding requires training a vision encoder from scratch. "
        "Use tools.miku_inference.get_screen_state_text() for text-based screen reasoning."
    )


def ground_and_click_target(
    target_element: str,
    duration: float = 0.20,
    real_execution: bool = True,
    model: Optional[Any] = None,
    image_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    [NOT YET IMPLEMENTED] Autonomous visual grounding + click.

    This previously used vikhyatk/moondream2 (a third-party pre-trained model).
    That has been removed. Pixel-level click grounding requires training Miku's
    own vision encoder from scratch.
    """
    return {
        "success": False,
        "error": "NOT_IMPLEMENTED",
        "message": (
            "[NOT YET IMPLEMENTED] Visual grounding requires training Miku's own "
            "vision encoder from scratch. Use Win32 window-based actions instead "
            "(open by name, press key, type text)."
        ),
        "target": target_element,
    }


__all__ = [
    "HAS_VISION_DEPS",
    "verify_vision_dependencies",
    "get_acceleration_device_and_dtype",
    "scale_and_clamp_point",
    "parse_moondream_point_output",
    "get_action_coordinates",
    "capture_desktop_pil",
    "ground_and_click_target",
]
