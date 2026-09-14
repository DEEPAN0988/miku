"""
eval/test_vision_grounder.py — Comprehensive Test Suite for Visual Fallback Screen Grounding

Tests:
  1. Box IoU computation (exact geometric assertions).
  2. Synthetic GUI canvas generation & contour detection:
     - Detects standard buttons, text inputs, icon badges, and containers.
  3. Hybrid Fusion:
     - UIA priority over overlapping visual contours.
     - Fallback inclusion of novel visual contours.
     - Full fallback activation when UIA yields 0 elements.
  4. Real Win32 GDI Bitmap Capture Smoke Test.
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))

from tools.screen_inspector import UIElement, ScreenSnapshot
from tools.vision_grounder import (
    compute_box_iou,
    detect_visual_interactive_regions,
    merge_uia_and_vision_elements,
    capture_window_bitmap,
    inspect_screen_with_vision_fallback,
)


class TestVisionGrounder(unittest.TestCase):

    def test_box_iou_computation(self):
        """Verify IoU calculation on identical, disjoint, and overlapping boxes."""
        boxA = (10, 10, 50, 50)  # 40x40 = 1600
        boxB = (10, 10, 50, 50)  # Identical
        self.assertAlmostEqual(compute_box_iou(boxA, boxB), 1.0)

        box_disjoint = (100, 100, 150, 150)
        self.assertEqual(compute_box_iou(boxA, box_disjoint), 0.0)

        # 50% width overlap
        box_half = (10, 10, 30, 50)  # 20x40 = 800
        # intersection = 20x40 = 800, union = 1600 + 800 - 800 = 1600 -> iou = 0.5
        self.assertAlmostEqual(compute_box_iou(boxA, box_half), 0.5)

    def test_synthetic_gui_detection(self):
        """
        Creates a synthetic 800x600 GUI bitmap with known UI elements:
        - 1 Edit box (wide horizontal)
        - 2 Action buttons (compact rectangles)
        - 1 Icon button (small square)
        Verifies all are detected with proper coordinates and source='vision'.
        """
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Light gray background

        # Draw Edit box at (50, 50, 450, 90) -> w=400, h=40, aspect=10.0
        cv2.rectangle(canvas, (50, 50), (450, 90), (255, 255, 255), -1)
        cv2.rectangle(canvas, (50, 50), (450, 90), (120, 120, 120), 2)
        cv2.putText(canvas, "Search...", (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

        # Draw Button 1 at (50, 120, 170, 160) -> w=120, h=40, aspect=3.0
        cv2.rectangle(canvas, (50, 120), (170, 160), (70, 130, 240), -1)
        cv2.rectangle(canvas, (50, 120), (170, 160), (30, 80, 180), 2)
        cv2.putText(canvas, "Submit", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw Button 2 at (200, 120, 320, 160) -> w=120, h=40
        cv2.rectangle(canvas, (200, 120), (320, 160), (220, 220, 220), -1)
        cv2.rectangle(canvas, (200, 120), (320, 160), (100, 100, 100), 2)
        cv2.putText(canvas, "Cancel", (220, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

        # Draw Icon Button at (500, 50, 535, 85) -> w=35, h=35
        cv2.rectangle(canvas, (500, 50), (535, 85), (200, 200, 200), -1)
        cv2.rectangle(canvas, (500, 50), (535, 85), (80, 80, 80), 2)

        elements = detect_visual_interactive_regions(canvas, window_origin=(100, 200))

        self.assertGreaterEqual(len(elements), 3)
        # All detected elements must have source="vision" and is_unlabeled=True
        for el in elements:
            self.assertEqual(el.source, "vision")
            self.assertTrue(el.is_unlabeled)
            # Ensure coordinates were offset by window_origin
            self.assertGreaterEqual(el.rect[0], 100)
            self.assertGreaterEqual(el.rect[1], 200)

        # Check that Edit box was classified as Edit
        edit_candidates = [e for e in elements if e.control_type == "Edit"]
        self.assertGreaterEqual(len(edit_candidates), 1)
        self.assertTrue(any(abs((e.rect[2] - e.rect[0]) - 400) < 15 for e in edit_candidates))

    def test_merge_uia_and_vision_elements(self):
        """Tests hybrid fusion logic: UIA dominance, deduplication, and fallback inclusion."""
        # 1. Existing UIA button at (100, 100, 200, 140)
        uia_btn = UIElement(
            control_type="Button",
            name="Save",
            automation_id="btn_save",
            rect=(100, 100, 200, 140),
            center=(150, 120),
            source="uia",
        )
        uia_list = [uia_btn]

        # 2. Vision elements:
        # One overlapping with uia_btn (IoU high)
        vis_dup = UIElement(
            control_type="Button",
            name="visual_button_1",
            automation_id="vis_1",
            rect=(102, 101, 198, 139),
            center=(150, 120),
            source="vision",
        )
        # One novel visual button (e.g. owner-drawn canvas button)
        vis_novel = UIElement(
            control_type="Button",
            name="visual_button_2",
            automation_id="vis_2",
            rect=(300, 100, 400, 140),
            center=(350, 120),
            source="vision",
        )

        merged = merge_uia_and_vision_elements(uia_list, [vis_dup, vis_novel], iou_threshold=0.35)

        # Result should contain uia_btn (authoritative) and vis_novel, suppressing vis_dup
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].source, "uia")
        self.assertEqual(merged[0].name, "Save")
        self.assertEqual(merged[1].source, "vision")
        self.assertEqual(merged[1].name, "visual_button_2")

    def test_empty_uia_fallback(self):
        """When UIA returns 0 elements, fusion returns all vision elements."""
        vis_el = UIElement(
            control_type="Button",
            name="visual_canvas_btn",
            automation_id="vis_canvas_1",
            rect=(50, 50, 150, 90),
            center=(100, 70),
            source="vision",
        )
        merged = merge_uia_and_vision_elements([], [vis_el])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].name, "visual_canvas_btn")
        self.assertEqual(merged[0].source, "vision")

    def test_real_desktop_capture_smoke(self):
        """Smoke test Win32 GDI capture on Desktop window."""
        import ctypes
        user32 = ctypes.windll.user32
        desk_hwnd = user32.GetDesktopWindow()

        capture = capture_window_bitmap(desk_hwnd)
        self.assertIsNotNone(capture)
        img, rect = capture
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)
        self.assertGreater(rect[2] - rect[0], 100)
        self.assertGreater(rect[3] - rect[1], 100)


if __name__ == "__main__":
    unittest.main()
