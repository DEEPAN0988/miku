"""
tools/vision_grounder.py — Visual Fallback Screen Grounder for Non-UIA / Canvas Bitmaps

Architectural Role:
  Provides visual fallback grounding when the Windows UIAutomation (UIA) accessibility
  tree is empty, sparse, or unable to see owner-drawn elements (e.g., custom GDI apps,
  DirectX canvases, web <canvas>, Electron bitmap panels, or unlabeled icon buttons).
  
Constitutional Compliance:
  - Evidence-based only: Reports exact bounding boxes derived from pixel edge analysis.
  - Zero GPU requirement: Runs on CPU via native Win32 GDI BitBlt, OpenCV (cv2), and Pillow.
  - Read-only: Dispatches zero OS mouse or keyboard events.
  - Hybrid-ready schema: Outputs UIElement objects with source="vision" and is_unlabeled=True.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np

from tools.screen_inspector import UIElement, ScreenSnapshot, INTERACTIVE_CONTROL_TYPES

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def _attach_thread_to_default_desktop():
    """Ensures calling thread accesses the interactive desktop."""
    try:
        hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


def capture_window_bitmap(hwnd: int) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Captures the visual bitmap of a target window via Win32 GDI BitBlt.
    Returns (image_bgr, (left, top, right, bottom)) or None if capture fails.
    """
    if not user32.IsWindow(hwnd):
        return None

    _attach_thread_to_default_desktop()

    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None

    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    w = max(0, right - left)
    h = max(0, bottom - top)

    if w < 10 or h < 10:
        return None

    hdc_src = user32.GetWindowDC(hwnd)
    if not hdc_src:
        return None

    hdc_mem = gdi32.CreateCompatibleDC(hdc_src)
    hbm = gdi32.CreateCompatibleBitmap(hdc_src, w, h)
    gdi32.SelectObject(hdc_mem, hbm)

    # SRCCOPY = 0x00CC0020
    success = gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_src, 0, 0, 0x00CC0020)

    arr: Optional[np.ndarray] = None
    if success:
        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # Top-down DIB
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        buf = ctypes.create_string_buffer(w * h * 4)
        lines = gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)
        if lines > 0:
            raw = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            # Convert BGRA to BGR
            arr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    # Cleanup GDI handles
    gdi32.DeleteObject(hbm)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc_src)

    if arr is None:
        return None

    return arr, (left, top, right, bottom)


def detect_visual_interactive_regions(
    image: np.ndarray,
    window_origin: Tuple[int, int] = (0, 0),
    min_area: int = 150,
    max_area_ratio: float = 0.85,
) -> List[UIElement]:
    """
    Analyzes an RGB/BGR image to detect interactive UI candidate regions
    (buttons, edit fields, clickable icons, containers) using multi-scale edge and contour analysis.

    Args:
        image: BGR numpy array of the window bitmap.
        window_origin: (screen_x, screen_y) offset of the window's top-left corner.
        min_area: minimum pixel area for an element.
        max_area_ratio: maximum fraction of the total window area.

    Returns:
        List of UIElement objects with source="vision" and calibrated bounding boxes.
    """
    if image is None or len(image.shape) < 2:
        return []

    h, w = image.shape[:2]
    total_area = h * w
    if total_area <= 0:
        return []

    # 1. Grayscale conversion
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Multi-strategy edge & contour detection
    # Strategy A: Adaptive Thresholding for sharp UI boundaries (light/dark themes)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Strategy B: Canny Edge Detection for subtle contrast boundaries
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Combine edge maps
    combined = cv2.bitwise_or(adapt, edges)

    # Morphological rectangular closing to connect button/box strokes
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_rect)

    # Find external and nested contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    elements: List[UIElement] = []
    seen_boxes: List[Tuple[int, int, int, int]] = []
    ox, oy = window_origin

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > (total_area * max_area_ratio):
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Discard extreme aspect ratios or full-span borders
        aspect = cw / max(1, ch)
        if aspect < 0.08 or aspect > 20.0:
            continue
        if cw < 12 or ch < 12:
            continue
        if cw > w * 0.98 and ch > h * 0.98:
            continue

        # Screen coordinates
        screen_left = ox + x
        screen_top = oy + y
        screen_right = screen_left + cw
        screen_bottom = screen_top + ch
        center_x = screen_left + (cw // 2)
        center_y = screen_top + (ch // 2)
        box = (screen_left, screen_top, screen_right, screen_bottom)

        # Deduplicate overlapping bounding boxes (IoU > 0.6)
        is_dup = False
        for sbox in seen_boxes:
            iou = compute_box_iou(box, sbox)
            if iou > 0.6:
                is_dup = True
                break
        if is_dup:
            continue
        seen_boxes.append(box)

        # Control Type Heuristic:
        # - Long horizontal bars (h in [18..60], aspect > 3.0) -> Edit / Input field
        # - Compact rectangles (aspect in [0.8..4.0], ch in [18..80]) -> Button
        # - Small square icons (ch in [12..45], aspect in [0.7..1.4]) -> Button / Icon
        # - Larger rectangles -> Pane
        if 18 <= ch <= 60 and aspect >= 3.0:
            ctrl_type = "Edit"
        elif (18 <= ch <= 80 and 0.8 <= aspect <= 4.0) or (12 <= ch <= 45 and 0.7 <= aspect <= 1.4):
            ctrl_type = "Button"
        else:
            ctrl_type = "Pane"

        idx = len(elements) + 1
        el = UIElement(
            control_type=ctrl_type,
            name=f"visual_{ctrl_type.lower()}_{idx}",
            automation_id=f"vision_id_{idx}",
            rect=box,
            center=(center_x, center_y),
            is_enabled=True,
            is_focusable=True if ctrl_type in ("Edit", "Button") else False,
            is_unlabeled=True,
            class_name="VisionContour",
            source="vision",
        )
        elements.append(el)

    return elements


def compute_box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (l, t, r, b)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(areaA + areaB - inter_area)

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def merge_uia_and_vision_elements(
    uia_elements: List[UIElement],
    vision_elements: List[UIElement],
    iou_threshold: float = 0.35,
) -> List[UIElement]:
    """
    Fuses UIA elements with visual contour elements.
    UIA elements are authoritative (source='uia').
    Vision elements are retained if they do not significantly overlap with any existing UIA element.
    If UIA elements is empty, all vision elements are returned.
    """
    if not uia_elements:
        return list(vision_elements)
    if not vision_elements:
        return list(uia_elements)

    merged = list(uia_elements)

    for vel in vision_elements:
        overlaps = False
        for uel in uia_elements:
            iou = compute_box_iou(vel.rect, uel.rect)
            if iou >= iou_threshold:
                overlaps = True
                break
        if not overlaps:
            merged.append(vel)

    return merged


def inspect_screen_with_vision_fallback(
    target_hwnd: int,
    interactive_only: bool = True,
    fallback_threshold: int = 1,
) -> ScreenSnapshot:
    """
    Inspects target window using primary Windows UIA COM tree.
    If UIA produces fewer elements than fallback_threshold (e.g. empty canvas or DirectX window),
    automatically executes visual bitmap capture and contour detection as fallback.
    """
    from tools.screen_inspector import inspect_screen

    t0 = time.perf_counter()
    snap = inspect_screen(target_hwnd, interactive_only=interactive_only)

    # If UIA found sufficient elements, return snapshot directly
    if snap.interactive_count >= fallback_threshold:
        return snap

    # Trigger Vision Grounder Fallback
    capture = capture_window_bitmap(target_hwnd)
    if capture is None:
        return snap

    image_bgr, win_rect = capture
    origin = (win_rect[0], win_rect[1])

    vision_els = detect_visual_interactive_regions(image_bgr, window_origin=origin)

    merged = merge_uia_and_vision_elements(snap.elements, vision_els)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    return ScreenSnapshot(
        hwnd=target_hwnd,
        title=snap.title,
        class_name=snap.class_name,
        process_name=snap.process_name,
        window_rect=win_rect,
        elements=merged,
        interactive_count=len(merged),
        total_scanned=snap.total_scanned + len(vision_els),
        latency_ms=latency,
        error=None,
    )
