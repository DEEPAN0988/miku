"""
tools/local_vision.py — 100% Native In-Process Vision-Language Model Bridge (Moondream2)

Architectural Role:
  Provides in-process visual grounding for desktop elements using a local
  vision-language model (vikhyatk/moondream2) powered by PyTorch and Hugging Face transformers.
  Runs entirely inside the main Python process with zero external server daemons or cloud APIs.

Key Capabilities:
  1. Dynamic hardware acceleration: Inspects torch.cuda.is_available(); automatically uses
     device_map="cuda" and dtype=torch.bfloat16 when an NVIDIA GPU is detected, falling back
     to CPU otherwise.
  2. Spatial Grounding: Calls model.point(image, target_element) to extract normalized coordinates
     and scales them to exact physical desktop pixel coordinates (X, Y).
  3. Direct Win32 Humanizer Dispatch: Connects grounded coordinates directly into human_mouse_move
     (cubic Bézier glide) and dispatch_real_click / dispatch_real_double_click.
  4. Resilient Dependency Handling: If PyTorch, PIL, or transformers are not installed, fails
     gracefully with a terminal prompt instructing the user to run:
       pip install torch transformers pillow
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Safe imports for vision libraries
HAS_VISION_DEPS: bool = True
VISION_DEPS_ERROR: Optional[str] = None

try:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM
except (ImportError, ModuleNotFoundError) as _err:
    HAS_VISION_DEPS = False
    VISION_DEPS_ERROR = str(_err)
    torch = None
    Image = None
    AutoModelForCausalLM = None

user32 = ctypes.windll.user32

# In-process singleton cache for the loaded Moondream2 model
_CACHED_MOONDREAM_MODEL: Optional[Any] = None


def verify_vision_dependencies() -> bool:
    """
    Verifies whether torch, PIL, and transformers are installed and importable.
    Prints a clear terminal instruction if any dependency is missing.
    """
    if not HAS_VISION_DEPS:
        print("\n" + "=" * 70, flush=True)
        print("[!] Missing required vision libraries for in-process local VLM.", flush=True)
        if VISION_DEPS_ERROR:
            print(f"    Error detail: {VISION_DEPS_ERROR}", flush=True)
        print("[!] Please run: pip install torch transformers pillow", flush=True)
        print("=" * 70 + "\n", flush=True)
        return False
    return True


def get_acceleration_device_and_dtype() -> Tuple[str, Any]:
    """
    Dynamically checks for hardware acceleration:
    If torch.cuda.is_available(), returns ('cuda', torch.bfloat16).
    Otherwise returns ('cpu', torch.float32).
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32 if torch is not None else None


def get_or_load_moondream_model(
    model_id: str = "vikhyatk/moondream2",
    revision: Optional[str] = "2025-01-09",
    trust_remote_code: bool = True,
    force_reload: bool = False,
) -> Any:
    """
    Loads or retrieves the in-process Moondream2 model.
    Dynamically applies CUDA hardware acceleration with bfloat16 precision.
    """
    global _CACHED_MOONDREAM_MODEL
    if _CACHED_MOONDREAM_MODEL is not None and not force_reload:
        return _CACHED_MOONDREAM_MODEL

    if not verify_vision_dependencies():
        return None

    device, dtype = get_acceleration_device_and_dtype()
    print(
        f"[*] [LOCAL VISION] Initializing native in-process VLM '{model_id}' "
        f"(device={device}, dtype={dtype})...",
        flush=True,
    )

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        load_kwargs["revision"] = revision

    if device == "cuda":
        load_kwargs["device_map"] = "cuda"
        load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if device == "cpu":
            model = model.to("cpu")
        _CACHED_MOONDREAM_MODEL = model
        print(f"[+] [LOCAL VISION] Successfully loaded '{model_id}' in-process.", flush=True)
        return _CACHED_MOONDREAM_MODEL
    except Exception as exc:
        print(f"[!] [LOCAL VISION ERROR] Failed to load '{model_id}': {exc}", flush=True)
        return None


def scale_and_clamp_point(
    norm_x: float,
    norm_y: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """
    Converts normalized coordinates [0.0, 1.0] to integer pixel coordinates [0, width-1], [0, height-1].
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
    Parses the spatial grounding output from model.point(image, target_element).
    Expected moondream formats:
      - {"points": [{"x": float, "y": float}, ...]}
      - {"point": {"x": float, "y": float}}
      - [{"x": float, "y": float}, ...]
      - {"x": float, "y": float}
      - (float, float) or [float, float]
    Returns (x, y) pixel coordinates scaled to image resolution, or None if not located.
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


def get_action_coordinates(
    image: Any,
    target_element: str,
    model: Optional[Any] = None,
) -> Optional[Tuple[int, int]]:
    """
    Accepts a PIL.Image of the desktop and a target_element query (e.g. 'Calculator icon').
    Uses the model's native spatial grounding method model.point(image, target_element).
    Parses and returns the exact (X, Y) pixel coordinates scaled to the desktop image resolution.
    """
    if not verify_vision_dependencies():
        return None

    active_model = model or get_or_load_moondream_model()
    if active_model is None:
        print("[!] [LOCAL VISION] Cannot run get_action_coordinates: model is not loaded.")
        return None

    if not hasattr(active_model, "point"):
        print(f"[!] [LOCAL VISION] Model {type(active_model)} has no 'point' spatial grounding method.")
        return None

    try:
        raw_output = active_model.point(image, target_element)
    except Exception as exc:
        print(f"[!] [LOCAL VISION] model.point() failed for query '{target_element}': {exc}", flush=True)
        return None

    w, h = getattr(image, "size", (1920, 1080))
    coords = parse_moondream_point_output(raw_output, image_width=w, image_height=h)
    return coords


def capture_desktop_pil() -> Optional[Any]:
    """
    Captures the physical desktop frame as a PIL.Image.Image.
    Uses Win32 GDI BitBlt via tools.vision_grounder for robust display session capture.
    """
    if not verify_vision_dependencies():
        return None

    # 1. Primary: Use Miku's GDI BitBlt with default desktop thread attachment
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

    # 2. Secondary fallback: PIL ImageGrab
    try:
        from PIL import ImageGrab
        return ImageGrab.grab()
    except Exception as exc:
        print(f"[!] [LOCAL VISION] Desktop capture failed: {exc}", flush=True)
        return None


def ground_and_click_target(
    target_element: str,
    duration: float = 0.20,
    real_execution: bool = True,
    model: Optional[Any] = None,
    image_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Autonomous In-Process Grounding & Execution Dispatch:
    1. Captures the physical desktop frame as a PIL Image.
    2. Passes the frame and target_element to get_action_coordinates.
    3. Feeds the resulting coordinates directly into human_mouse_move (Bézier glide)
       and dispatch_real_click from tools.screen_inspector.
    """
    from tools.screen_inspector import (
        human_mouse_move,
        dispatch_real_click,
        simulate_click,
        get_current_cursor_pos,
    )
    from tools.computer_use_agent import resolve_target_hwnd_at_point

    # Step 1: Capture desktop frame
    frame = image_override if image_override is not None else capture_desktop_pil()
    if frame is None:
        return {
            "success": False,
            "error": "CAPTURE_FAILED",
            "message": "Failed to capture desktop screenshot.",
            "target": target_element,
        }

    # Step 2: Ground target element coordinates
    coords = get_action_coordinates(frame, target_element, model=model)
    if coords is None:
        return {
            "success": False,
            "error": "GROUNDING_FAILED",
            "message": f"Could not visually ground '{target_element}' on screen.",
            "target": target_element,
        }

    target_x, target_y = coords

    # Step 3: Glide cursor smoothly with Bézier curve
    cur_x, cur_y = get_current_cursor_pos()
    human_mouse_move(cur_x, cur_y, target_x, target_y, duration=duration)

    # Step 4: Dispatch click
    desktop_hwnd = user32.GetDesktopWindow()
    target_hwnd = resolve_target_hwnd_at_point(target_x, target_y) or desktop_hwnd

    if real_execution:
        click_res = dispatch_real_click(target_hwnd, (target_x, target_y), button="left")
        return {
            "success": click_res.success,
            "coordinates": (target_x, target_y),
            "target": target_element,
            "verification": click_res.verification.status if click_res.verification else "NONE",
            "action_log": click_res.action_log,
        }
    else:
        sim_res = simulate_click(target_hwnd, (target_x, target_y), button="left")
        return {
            "success": sim_res.success,
            "coordinates": (target_x, target_y),
            "target": target_element,
            "verification": "SIMULATED",
            "action_log": sim_res.action_log,
        }


__all__ = [
    "HAS_VISION_DEPS",
    "verify_vision_dependencies",
    "get_acceleration_device_and_dtype",
    "get_or_load_moondream_model",
    "scale_and_clamp_point",
    "parse_moondream_point_output",
    "get_action_coordinates",
    "capture_desktop_pil",
    "ground_and_click_target",
]
