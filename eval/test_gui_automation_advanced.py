"""
eval/test_gui_automation_advanced.py — Advanced GUI Automation Integration Suite

Tests the unified combination of:
  - Option 4: Vision Grounder Fallback (Bitmap capture, contour detection, hybrid fusion)
  - Option 2: GUI Edit Control Text Staging & Typing Automation (Sanitizer, editability, simulation)
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import unittest
import time
import ctypes
from ctypes import wintypes
import win32gui
import win32con

from tools.screen_inspector import (
    inspect_screen,
    find_element,
    verify_element_clickable,
    simulate_click,
    UIElement,
)
from tools.vision_grounder import (
    capture_window_bitmap,
    detect_visual_interactive_regions,
    merge_uia_and_vision_elements,
    inspect_screen_with_vision_fallback,
)
from tools.typing_automation import (
    REAL_TYPE_ENABLED,
    sanitize_typing_payload,
    verify_element_editable,
    simulate_typing,
    dispatch_real_typing,
)
from eval.test_screen_inspector import get_or_spawn_calculator, set_window_foreground_passive


class TestGUIAutomationAdvanced(unittest.TestCase):

    def test_end_to_end_simulated_workflow(self):
        """
        End-to-End Simulation of:
          1. Vision grounder fallback on a custom/synthetic UI canvas.
          2. Discovery of visual edit box and button.
          3. Simulated typing into edit box.
          4. Simulated click on button.
        """
        import numpy as np
        import cv2

        # Create a synthetic canvas representing a custom non-UIA desktop application
        h, w = 500, 700
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 235

        # Draw a custom edit box at (50, 60, 450, 100) -> w=400, h=40
        cv2.rectangle(canvas, (50, 60), (450, 100), (255, 255, 255), -1)
        cv2.rectangle(canvas, (50, 60), (450, 100), (100, 100, 100), 2)
        cv2.putText(canvas, "Username", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # Draw a custom button at (50, 140, 170, 180) -> w=120, h=40
        cv2.rectangle(canvas, (50, 140), (170, 180), (60, 120, 240), -1)
        cv2.rectangle(canvas, (50, 140), (170, 180), (20, 60, 180), 2)
        cv2.putText(canvas, "Login", (75, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Step 1: Detect visual regions
        vis_elements = detect_visual_interactive_regions(canvas, window_origin=(0, 0))
        self.assertGreaterEqual(len(vis_elements), 2)

        # Find visual Edit box
        edit_els = [e for e in vis_elements if e.control_type == "Edit"]
        self.assertGreaterEqual(len(edit_els), 1)
        target_edit = edit_els[0]

        # Find visual Button
        btn_els = [e for e in vis_elements if e.control_type == "Button"]
        self.assertGreaterEqual(len(btn_els), 1)
        target_btn = btn_els[0]

        # Step 2: Test payload sanitization for typing
        payload = "admin_user_01"
        is_safe, clean_text, reason = sanitize_typing_payload(payload)
        self.assertTrue(is_safe)
        self.assertEqual(clean_text, payload)

        # Step 3: Verify simulated typing structure
        sim_type = simulate_typing(0, target_edit, clean_text)
        self.assertFalse(sim_type.success)
        self.assertEqual(sim_type.status, "ABORT_INVALID_HWND")
        self.assertFalse(sim_type.real_input_dispatched)

        # Step 4: Verify simulated click on button catches invalid HWND safely
        sim_click = simulate_click(0, target_btn)
        self.assertFalse(sim_click.success)
        self.assertEqual(sim_click.status, "ABORT_INVALID_HWND")
        self.assertFalse(sim_click.real_input_dispatched)

    def test_live_window_hybrid_inspection(self):
        """
        Tests live hybrid UIA + Vision fallback inspection and pre-typing control
        gating on an active desktop window (Calculator).
        """
        calc_hwnds = []
        def _find_calc(h, _):
            if win32gui.IsWindowVisible(h):
                try:
                    t = win32gui.GetWindowText(h)
                    c = win32gui.GetClassName(h)
                    if ("calculator" in t.lower() or "calculator" in c.lower()) and "frame" in c.lower():
                        calc_hwnds.append(h)
                except Exception:
                    pass
            return True
        win32gui.EnumWindows(_find_calc, None)
        if not calc_hwnds:
            self.skipTest("Calculator is not already open; skipping live test to avoid unprompted process spawning")
        calc_hwnd = calc_hwnds[0]
        set_window_foreground_passive(calc_hwnd)
        time.sleep(0.3)

        # Inspect screen using vision fallback engine
        snap = inspect_screen_with_vision_fallback(calc_hwnd)
        self.assertGreater(snap.interactive_count, 10)
        self.assertEqual(snap.hwnd, calc_hwnd)

        # Locate "Clear" button
        clear_btn = find_element(snap, "clear") or snap.elements[0]
        self.assertIsNotNone(clear_btn)
        self.assertEqual(clear_btn.control_type, "Button")

        # Verify editability rejection on Button (Buttons cannot receive text input)
        verif = verify_element_editable(calc_hwnd, clear_btn)
        self.assertFalse(verif.is_safe)
        self.assertEqual(verif.reason, "NOT_EDITABLE_TYPE")

        # Verify simulated typing into Button is safely aborted
        sim = simulate_typing(calc_hwnd, clear_btn, "12345")
        self.assertFalse(sim.success)
        self.assertEqual(sim.status, "ABORT_NOT_EDITABLE_TYPE")
        self.assertFalse(sim.real_input_dispatched)

    def test_native_edit_control_staging_and_circuit_breaker(self):
        """
        Creates an authentic native Win32 Edit control in-process, tests payload staging,
        and verifies circuit breaker blocks physical keystroke dispatch.
        """
        hwnd = win32gui.CreateWindow(
            "EDIT",
            "MikuTestEditArea",
            win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE,
            100,
            100,
            350,
            200,
            0,
            0,
            0,
            None,
        )
        self.assertNotEqual(hwnd, 0, "Native edit window must be created")

        try:
            time.sleep(0.1)
            target_el = UIElement(
                control_type="Edit",
                name="MikuTestEditArea",
                automation_id="txtMiku",
                rect=(110, 110, 300, 160),
                center=(205, 135),
                is_enabled=True,
            )

            # Test payload sanitizer
            test_payload = "Staged Text for Input\nLine 2"
            is_safe, clean_text, reason = sanitize_typing_payload(test_payload)
            self.assertTrue(is_safe)

            # Test dispatch_real_typing circuit breaker
            real_res = dispatch_real_typing(hwnd, target_el, clean_text)
            self.assertFalse(real_res.success)
            self.assertEqual(real_res.status, "DRY_RUN_PENDING_CIRCUIT_BREAKER")
            self.assertFalse(real_res.real_input_dispatched)
            self.assertEqual(real_res.characters_typed, 0)

        finally:
            win32gui.DestroyWindow(hwnd)


if __name__ == "__main__":
    unittest.main()
