"""
eval/test_screen_inspector.py — Comprehensive Unit & Integration Tests for Screen Inspector

Covers:
1. UIElement and ScreenSnapshot data contract & serialization.
2. Element query and matching (exact automation ID, exact name, substring, control-type filtering).
3. Viewport clipping and off-screen coordinate rejection logic.
4. Error handling and fail-closed behavior on invalid HWNDs.
5. Live Windows application inspection against Calculator (read-only, with clean lifecycle cleanup).
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import unittest
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import win32gui
import win32con
import psutil
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

from tools.screen_inspector import (
    ClickVerificationResult,
    REAL_CLICK_ENABLED,
    ScreenSnapshot,
    SimulatedClickResult,
    UIElement,
    find_all_elements,
    find_element,
    inspect_active_window,
    inspect_screen_elements,
    simulate_click,
    verify_element_clickable,
    _attach_thread_to_default_desktop,
)


class TestScreenInspectorUnit(unittest.TestCase):
    """Deterministic, mock-based unit tests for screen inspector logic."""

    def setUp(self):
        self.mock_elements = [
            UIElement(
                control_type="Button",
                name="Clear entry",
                automation_id="clearEntryButton",
                rect=(266, 310, 438, 389),
                center=(352, 349),
                is_enabled=True,
                is_focusable=True,
                source="uia",
            ),
            UIElement(
                control_type="Button",
                name="Memory add",
                automation_id="MemPlus",
                rect=(251, 258, 329, 308),
                center=(290, 283),
                is_enabled=True,
                is_focusable=True,
                source="uia",
            ),
            UIElement(
                control_type="Text",
                name="Display is 0",
                automation_id="CalculatorResults",
                rect=(87, 142, 791, 257),
                center=(439, 199),
                is_enabled=True,
                is_focusable=True,
                source="uia",
            ),
            UIElement(
                control_type="Edit",
                name="Type a message",
                automation_id="ChatInputBox",
                rect=(100, 600, 500, 650),
                center=(300, 625),
                is_enabled=True,
                is_focusable=True,
                source="uia",
            ),
            UIElement(
                control_type="MenuItem",
                name="File",
                automation_id="FileMenu",
                rect=(10, 10, 50, 30),
                center=(30, 20),
                is_enabled=True,
                is_focusable=False,
                source="uia",
            ),
        ]
        self.snapshot = ScreenSnapshot(
            hwnd=12345,
            title="Calculator",
            class_name="ApplicationFrameWindow",
            process_name="calculatorapp.exe",
            window_rect=(0, 0, 1000, 800),
            elements=self.mock_elements,
            interactive_count=len(self.mock_elements),
            total_scanned=10,
            latency_ms=12.5,
            error=None,
        )

    def test_element_dimensions_and_center(self):
        el = self.mock_elements[0]
        self.assertEqual(el.width, 438 - 266)
        self.assertEqual(el.height, 389 - 310)
        self.assertEqual(el.center, (352, 349))
        self.assertEqual(el.source, "uia")

    def test_serialization(self):
        d = self.snapshot.to_dict()
        self.assertEqual(d["hwnd"], 12345)
        self.assertEqual(d["title"], "Calculator")
        self.assertEqual(d["interactive_count"], 5)
        self.assertEqual(len(d["elements"]), 5)
        self.assertEqual(d["elements"][0]["automation_id"], "clearEntryButton")

    def test_find_by_exact_automation_id(self):
        el = find_element(self.snapshot, "MemPlus")
        self.assertIsNotNone(el)
        self.assertEqual(el.name, "Memory add")
        self.assertEqual(el.automation_id, "MemPlus")

    def test_find_by_exact_name(self):
        el = find_element(self.snapshot, "Clear entry")
        self.assertIsNotNone(el)
        self.assertEqual(el.automation_id, "clearEntryButton")

    def test_find_by_case_insensitive_substring(self):
        el = find_element(self.snapshot, "results")
        self.assertIsNotNone(el)
        self.assertEqual(el.automation_id, "CalculatorResults")

    def test_find_with_control_type_filter(self):
        el_btn = find_element(self.snapshot, "File", control_type="Button")
        self.assertIsNone(el_btn)  # 'File' is a MenuItem, not Button

        el_menu = find_element(self.snapshot, "File", control_type="MenuItem")
        self.assertIsNotNone(el_menu)
        self.assertEqual(el_menu.name, "File")

    def test_find_all_elements(self):
        buttons = find_all_elements(self.snapshot, control_type="Button")
        self.assertEqual(len(buttons), 2)
        self.assertTrue(all(b.control_type == "Button" for b in buttons))

        all_els = find_all_elements(self.snapshot)
        self.assertEqual(len(all_els), 5)

    def test_unlabeled_element_detection(self):
        unlabeled_el = UIElement(
            control_type="Button",
            name="",
            automation_id="",
            rect=(50, 50, 80, 80),
            center=(65, 65),
            is_unlabeled=True,
            source="uia",
        )
        self.assertTrue(unlabeled_el.is_unlabeled)
        self.assertEqual(unlabeled_el.to_dict()["is_unlabeled"], True)

    def test_inspect_screen_elements_signature_and_alias(self):
        # Verify function name and alias contract
        self.assertIs(inspect_active_window, inspect_screen_elements)

    def test_invalid_hwnd_fails_closed(self):
        snap = inspect_screen_elements(hwnd=99999999)
        self.assertIsNotNone(snap.error)
        self.assertEqual(len(snap.elements), 0)

    def test_click_verification_result_contract(self):
        res = ClickVerificationResult(
            is_safe=True,
            reason="SAFE",
            details={"target_hwnd": 12345, "coordinate": (100, 200)},
        )
        self.assertTrue(res.is_safe)
        self.assertEqual(res.reason, "SAFE")
        self.assertEqual(res.details["target_hwnd"], 12345)

        # Tuple unpacking support
        safe, reason, details = res
        self.assertTrue(safe)
        self.assertEqual(reason, "SAFE")
        self.assertEqual(details["coordinate"], (100, 200))

        # Dict serialization
        d = res.to_dict()
        self.assertEqual(d["is_safe"], True)
        self.assertEqual(d["reason"], "SAFE")

    def test_verify_element_invalid_inputs(self):
        # Invalid HWND
        safe, reason, _ = verify_element_clickable(0, (100, 100))
        self.assertFalse(safe)
        self.assertEqual(reason, "INVALID_HWND")

        safe2, reason2, _ = verify_element_clickable(99999999, (100, 100))
        self.assertFalse(safe2)
        self.assertEqual(reason2, "INVALID_HWND")

        # Invalid Element type against dummy desktop window
        dt_hwnd = win32gui.GetDesktopWindow()
        safe3, reason3, _ = verify_element_clickable(dt_hwnd, "not_an_element")
        self.assertFalse(safe3)
        self.assertIn(reason3, ["INVALID_ELEMENT", "MINIMIZED", "NOT_FOREGROUND"])


def set_window_foreground_passive(hwnd: int, timeout_s: float = 2.0) -> bool:
    """
    Passively sets target window as foreground using standard Win32 thread attachment.
    ZERO keystrokes, ZERO simulated clicks, ZERO SendInput.
    """
    _attach_thread_to_default_desktop()
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.AllowSetForegroundWindow(-1)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        fg_hwnd = user32.GetForegroundWindow()
        root_fg = user32.GetAncestor(fg_hwnd, 2) or fg_hwnd
        root_target = user32.GetAncestor(hwnd, 2) or hwnd
        if fg_hwnd == hwnd or root_fg == hwnd or root_fg == root_target:
            return True

        fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None)
        target_tid = user32.GetWindowThreadProcessId(hwnd, None)
        cur_tid = kernel32.GetCurrentThreadId()

        user32.AttachThreadInput(cur_tid, fg_tid, True)
        user32.AttachThreadInput(cur_tid, target_tid, True)
        user32.ShowWindow(hwnd, win32con.SW_RESTORE)
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(cur_tid, target_tid, False)
        user32.AttachThreadInput(cur_tid, fg_tid, False)

        time.sleep(0.1)

    fg_final = user32.GetForegroundWindow()
    root_final = user32.GetAncestor(fg_final, 2) or fg_final
    root_target = user32.GetAncestor(hwnd, 2) or hwnd
    return (fg_final == hwnd or root_final == hwnd or root_final == root_target)


CALC_HWND: int = 0


def get_or_spawn_calculator() -> int:
    """Provides a guaranteed healthy, single open Calculator window handle."""
    global CALC_HWND
    _attach_thread_to_default_desktop()
    user32 = ctypes.windll.user32

    def _is_healthy(h: int) -> bool:
        if not (h and user32.IsWindow(h)):
            return False
        if user32.IsIconic(h):
            user32.ShowWindow(h, win32con.SW_RESTORE)
            user32.ShowWindow(h, win32con.SW_SHOWNORMAL)
            time.sleep(0.1)
        snap = inspect_screen_elements(h)
        return len(snap.elements) > 10

    if _is_healthy(CALC_HWND):
        user32.ShowWindow(CALC_HWND, win32con.SW_RESTORE)
        user32.ShowWindow(CALC_HWND, win32con.SW_SHOWNORMAL)
        set_window_foreground_passive(CALC_HWND)
        return CALC_HWND

    # Find existing matching windows
    matching_hwnds = []
    def _find(h, _):
        if win32gui.IsWindowVisible(h):
            try:
                t = win32gui.GetWindowText(h)
                c = win32gui.GetClassName(h)
            except Exception:
                return True
            if ("calculator" in t.lower() or "calculator" in c.lower()) and "frame" in c.lower():
                matching_hwnds.append(h)
        return True

    win32gui.EnumWindows(_find, None)
    for h in matching_hwnds:
        if _is_healthy(h):
            CALC_HWND = h
            # Close any duplicate/stale windows
            for extra in matching_hwnds:
                if extra != h:
                    try:
                        win32gui.PostMessage(extra, win32con.WM_CLOSE, 0, 0)
                    except Exception:
                        pass
            user32.ShowWindow(CALC_HWND, win32con.SW_RESTORE)
            user32.ShowWindow(CALC_HWND, win32con.SW_SHOWNORMAL)
            set_window_foreground_passive(CALC_HWND)
            time.sleep(0.2)
            return CALC_HWND

    # Terminate any hung/invisible leftover calc processes
    for p in psutil.process_iter(["name"]):
        if p.info["name"] and any(x in p.info["name"].lower() for x in ["calculator", "calc"]):
            try:
                p.terminate()
            except Exception:
                pass
    time.sleep(0.5)

    subprocess.Popen(["calc.exe"])
    for _ in range(15):
        time.sleep(0.4)
        matching_hwnds.clear()
        win32gui.EnumWindows(_find, None)
        for h in matching_hwnds:
            if _is_healthy(h):
                CALC_HWND = h
                break
        if CALC_HWND:
            break

    if not _is_healthy(CALC_HWND):
        raise RuntimeError("Failed to obtain healthy Calculator window")

    user32.ShowWindow(CALC_HWND, win32con.SW_RESTORE)
    user32.ShowWindow(CALC_HWND, win32con.SW_SHOWNORMAL)
    set_window_foreground_passive(CALC_HWND)
    time.sleep(0.3)
    return CALC_HWND


def find_other_desktop_window(exclude_hwnd: int) -> int:
    """Finds another top-level visible desktop window (e.g. IDE, terminal, browser) to switch focus to."""
    _attach_thread_to_default_desktop()
    user32 = ctypes.windll.user32
    other_hwnd = 0
    def _find(h, _):
        nonlocal other_hwnd
        if other_hwnd:
            return True
        root_h = user32.GetAncestor(h, 2) or h
        if h != exclude_hwnd and root_h != exclude_hwnd and win32gui.IsWindowVisible(h):
            try:
                t = win32gui.GetWindowText(h).strip()
                r = win32gui.GetWindowRect(h)
            except Exception:
                return True
            if t and (r[2] - r[0] > 200) and any(k in t.lower() for k in ["antigravity", "claude", "visual studio", "powershell", "cmd", "terminal", "code"]):
                other_hwnd = h
        return True

    win32gui.EnumWindows(_find, None)
    if not other_hwnd:
        other_hwnd = win32gui.FindWindow("Progman", None)
    return other_hwnd


class TestScreenInspectorLiveIntegration(unittest.TestCase):
    """Live integration tests executing read-only UIA inspection on real apps."""

    def test_live_calculator_inspection(self):
        _attach_thread_to_default_desktop()
        calc_hwnd = get_or_spawn_calculator()
        self.assertGreater(calc_hwnd, 0, "Calculator window HWND should be found.")
        set_window_foreground_passive(calc_hwnd)
        time.sleep(0.3)

        # 2. Inspect active Calculator window
        snap = inspect_active_window(target_hwnd=calc_hwnd, max_elements=200)

        # 3. Assertions on real discovered elements
        self.assertIsNone(snap.error)
        self.assertGreater(snap.interactive_count, 15, "Should discover at least 15 interactive elements.")
        self.assertLess(snap.latency_ms, 1500, "Inspection latency should be well under 1.5 seconds.")

        # 4. Search for key calculator controls
        clear_btn = find_element(snap, "clear")
        self.assertIsNotNone(clear_btn, "Should find a 'clear' button in Calculator.")
        self.assertEqual(clear_btn.control_type, "Button")
        self.assertGreater(clear_btn.width, 10)
        self.assertGreater(clear_btn.height, 10)

        # Calculator numeric/entry controls
        mem_btn = find_element(snap, "MemPlus")
        self.assertIsNotNone(mem_btn, "Should find button with id 'MemPlus'.")

        # 5. Passive Coordinate Verification via WindowFromPoint
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        calc_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(calc_hwnd, ctypes.byref(calc_pid))

        verified_coords = 0
        for el in snap.elements:
            cx, cy = el.center
            pt = wintypes.POINT(cx, cy)
            hit_hwnd = user32.WindowFromPoint(pt)
            root = user32.GetAncestor(hit_hwnd, 2)
            hit_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hit_hwnd, ctypes.byref(hit_pid))

            # Point must land inside Calculator or its child island
            if root == calc_hwnd or hit_hwnd == calc_hwnd or hit_pid.value == calc_pid.value:
                verified_coords += 1

        self.assertEqual(
            verified_coords,
            len(snap.elements),
            f"All {len(snap.elements)} coordinates must resolve to Calculator window region"
        )


class TestPreClickSafetyGateLive(unittest.TestCase):
    """
    Empirically verifies all 4 distinct pre-click safety scenarios against live Windows desktop:
      1. SAFE (Foreground, unminimized, direct hit)
      2. OCCLUDED_AT_POINT (Topmost overlay covering target coordinate)
      3. NOT_FOREGROUND (Target visible but another window has focus)
      4. MINIMIZED (Target window iconic)
      + BOUNDS check (Target coordinate outside window rect)
    """

    calc_hwnd: int = 0

    @classmethod
    def setUpClass(cls):
        cls.calc_hwnd = get_or_spawn_calculator()
        set_window_foreground_passive(cls.calc_hwnd)
        time.sleep(0.3)

    def setUp(self):
        self.calc_hwnd = get_or_spawn_calculator()
        user32 = ctypes.windll.user32
        user32.ShowWindow(self.calc_hwnd, win32con.SW_RESTORE)
        user32.ShowWindow(self.calc_hwnd, win32con.SW_SHOWNORMAL)
        set_window_foreground_passive(self.calc_hwnd)
        time.sleep(0.3)

    def test_gate_scenario_1_safe(self):
        """Scenario 1: Calculator is foreground, unminimized, and coordinate is unoccluded."""
        snap = inspect_screen_elements(self.calc_hwnd)
        self.assertGreater(len(snap.elements), 0)
        el = find_element(snap, "clear") or snap.elements[0]

        is_safe, reason, details = verify_element_clickable(self.calc_hwnd, el)
        self.assertTrue(is_safe)
        self.assertEqual(reason, "SAFE")
        self.assertEqual(details["target_hwnd"], self.calc_hwnd)
        self.assertTrue(
            details["hit_root_hwnd"] == self.calc_hwnd or details["target_pid"] == details["hit_pid"]
        )

    def test_gate_scenario_2_occluded_at_point(self):
        """Scenario 2: Real topmost window occludes the element's coordinate."""
        user32 = ctypes.windll.user32
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]
        cx, cy = el.center

        # Verify safe before occlusion
        safe_before, reason_before, details_before = verify_element_clickable(self.calc_hwnd, el)
        self.assertTrue(safe_before, f"Expected SAFE before occlusion, got {reason_before}: {details_before}")

        # Place a topmost non-activating window covering the element coordinates
        overlay_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE,
            "STATIC",
            "OcclusionCover",
            win32con.WS_POPUP | win32con.WS_VISIBLE | 0x0100,  # SS_NOTIFY
            cx - 30, cy - 30, 60, 60,
            0, 0, 0, None
        )
        time.sleep(0.2)

        try:
            is_safe, reason, details = verify_element_clickable(self.calc_hwnd, el)
            self.assertFalse(is_safe)
            self.assertEqual(reason, "OCCLUDED_AT_POINT")
            self.assertEqual(details["hit_hwnd"], overlay_hwnd)
            self.assertEqual(details["hit_class_name"].lower(), "static")
        finally:
            win32gui.DestroyWindow(overlay_hwnd)
            time.sleep(0.2)

        # Confirm immediate recovery (zero false positives) once occluder is removed
        safe_after, reason_after, _ = verify_element_clickable(self.calc_hwnd, el)
        self.assertTrue(safe_after)
        self.assertEqual(reason_after, "SAFE")

    def test_gate_scenario_3_not_foreground(self):
        """Scenario 3: Target window is open/visible, but another application has focus."""
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]

        other_hwnd = find_other_desktop_window(self.calc_hwnd)
        self.assertGreater(other_hwnd, 0, "Should locate another desktop window to switch focus to")

        set_window_foreground_passive(other_hwnd)
        time.sleep(0.4)

        try:
            is_safe, reason, details = verify_element_clickable(self.calc_hwnd, el)
            self.assertFalse(is_safe)
            self.assertEqual(reason, "NOT_FOREGROUND")
            self.assertNotEqual(details.get("foreground_hwnd"), self.calc_hwnd)
        finally:
            set_window_foreground_passive(self.calc_hwnd)
            time.sleep(0.3)

    def test_gate_scenario_4_minimized(self):
        """Scenario 4: Target window is iconic / minimized."""
        user32 = ctypes.windll.user32
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]

        user32.ShowWindow(self.calc_hwnd, win32con.SW_MINIMIZE)
        time.sleep(0.3)

        try:
            is_safe, reason, details = verify_element_clickable(self.calc_hwnd, el)
            self.assertFalse(is_safe)
            self.assertEqual(reason, "MINIMIZED")
            self.assertTrue(details.get("is_iconic"))
        finally:
            user32.ShowWindow(self.calc_hwnd, win32con.SW_RESTORE)
            user32.ShowWindow(self.calc_hwnd, win32con.SW_SHOWNORMAL)
            set_window_foreground_passive(self.calc_hwnd)
            time.sleep(0.4)

    def test_gate_coordinate_outside_window(self):
        """Bounds Guard: Coordinate falling outside the target window rect is caught immediately."""
        is_safe, reason, details = verify_element_clickable(self.calc_hwnd, (-500, -500))
        self.assertFalse(is_safe)
        self.assertEqual(reason, "COORDINATE_OUTSIDE_WINDOW")


class TestSimulatedClickDispatcherLive(unittest.TestCase):
    """
    End-to-end integration test suite for the simulated click dispatcher (Phase B).
    Enforces verify-before-act pattern and guarantees zero real input events.
    """

    calc_hwnd: int = 0

    @classmethod
    def setUpClass(cls):
        cls.calc_hwnd = get_or_spawn_calculator()
        set_window_foreground_passive(cls.calc_hwnd)
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        global CALC_HWND
        CALC_HWND = 0
        for p in psutil.process_iter(["name"]):
            if p.info["name"] and any(x in p.info["name"].lower() for x in ["calculator", "calc"]):
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(0.5)

    def setUp(self):
        self.calc_hwnd = get_or_spawn_calculator()
        user32 = ctypes.windll.user32
        user32.ShowWindow(self.calc_hwnd, win32con.SW_RESTORE)
        user32.ShowWindow(self.calc_hwnd, win32con.SW_SHOWNORMAL)
        set_window_foreground_passive(self.calc_hwnd)
        time.sleep(0.4)

    def test_circuit_breaker_disabled_by_default(self):
        """Permanent safety circuit breaker must be hardcoded to False."""
        self.assertFalse(REAL_CLICK_ENABLED)

    def test_dispatch_scenario_1_safe(self):
        """Task 2 Scenario 1: Unconditional gate pass -> simulated click logged, zero real input."""
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]

        res = simulate_click(self.calc_hwnd, el)
        print(f"\n[DISPATCH TELEMETRY 1 - SAFE]\n  Status: {res.status}\n  Log: {res.action_log}\n  Real Input: {res.real_input_dispatched}")

        self.assertTrue(res.success)
        self.assertEqual(res.status, "SIMULATED_CLICK_SUCCESS")
        self.assertFalse(res.real_input_dispatched)
        self.assertTrue(res.verification.is_safe)
        self.assertEqual(res.verification.reason, "SAFE")
        self.assertIn("[SIMULATED CLICK] Would click", res.action_log)
        self.assertIn("Real Input Dispatched: NO", res.action_log)

    def test_dispatch_scenario_2_occluded_at_point(self):
        """Task 2 Scenario 2: Occluded coordinate -> immediate abort with ABORT_OCCLUDED."""
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]
        cx, cy = el.center

        overlay_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE,
            "STATIC",
            "OcclusionCover",
            win32con.WS_POPUP | win32con.WS_VISIBLE | 0x0100,
            cx - 30, cy - 30, 60, 60,
            0, 0, 0, None
        )
        time.sleep(0.2)

        try:
            res = simulate_click(self.calc_hwnd, el)
            print(f"\n[DISPATCH TELEMETRY 2 - OCCLUDED]\n  Status: {res.status}\n  Log: {res.action_log}\n  Real Input: {res.real_input_dispatched}")

            self.assertFalse(res.success)
            self.assertEqual(res.status, "ABORT_OCCLUDED")
            self.assertFalse(res.real_input_dispatched)
            self.assertEqual(res.verification.reason, "OCCLUDED_AT_POINT")
            self.assertIn("[PRE-CLICK SAFETY REJECTION]", res.action_log)
            self.assertEqual(res.verification.details.get("hit_hwnd"), overlay_hwnd)
        finally:
            win32gui.DestroyWindow(overlay_hwnd)
            time.sleep(0.2)

        # Immediate recovery
        res_recovered = simulate_click(self.calc_hwnd, el)
        self.assertTrue(res_recovered.success)
        self.assertEqual(res_recovered.status, "SIMULATED_CLICK_SUCCESS")
        self.assertFalse(res_recovered.real_input_dispatched)

    def test_dispatch_scenario_3_not_foreground(self):
        """Task 2 Scenario 3: Target not foreground -> immediate abort with ABORT_NOT_FOREGROUND."""
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]

        other_hwnd = find_other_desktop_window(self.calc_hwnd)
        self.assertGreater(other_hwnd, 0, "Must locate another desktop window to switch focus")
        set_window_foreground_passive(other_hwnd)
        time.sleep(0.4)

        try:
            res = simulate_click(self.calc_hwnd, el)
            print(f"\n[DISPATCH TELEMETRY 3 - NOT_FOREGROUND]\n  Status: {res.status}\n  Log: {res.action_log}\n  Real Input: {res.real_input_dispatched}")

            self.assertFalse(res.success)
            self.assertEqual(res.status, "ABORT_NOT_FOREGROUND")
            self.assertFalse(res.real_input_dispatched)
            self.assertEqual(res.verification.reason, "NOT_FOREGROUND")
            self.assertIn("[PRE-CLICK SAFETY REJECTION]", res.action_log)
        finally:
            set_window_foreground_passive(self.calc_hwnd)
            time.sleep(0.3)

    def test_dispatch_scenario_4_minimized(self):
        """Task 2 Scenario 4: Target minimized -> immediate abort with ABORT_MINIMIZED."""
        user32 = ctypes.windll.user32
        snap = inspect_screen_elements(self.calc_hwnd)
        el = find_element(snap, "clear") or snap.elements[0]

        user32.ShowWindow(self.calc_hwnd, win32con.SW_MINIMIZE)
        time.sleep(0.3)

        try:
            res = simulate_click(self.calc_hwnd, el)
            print(f"\n[DISPATCH TELEMETRY 4 - MINIMIZED]\n  Status: {res.status}\n  Log: {res.action_log}\n  Real Input: {res.real_input_dispatched}")

            self.assertFalse(res.success)
            self.assertEqual(res.status, "ABORT_MINIMIZED")
            self.assertFalse(res.real_input_dispatched)
            self.assertEqual(res.verification.reason, "MINIMIZED")
            self.assertIn("[PRE-CLICK SAFETY REJECTION]", res.action_log)
        finally:
            user32.ShowWindow(self.calc_hwnd, win32con.SW_RESTORE)
            user32.ShowWindow(self.calc_hwnd, win32con.SW_SHOWNORMAL)
            set_window_foreground_passive(self.calc_hwnd)
            time.sleep(0.4)

    def test_dispatch_scenario_5_outside_window(self):
        """Task 2 Scenario 5: Out of bounds coordinate -> immediate abort with ABORT_OUTSIDE_WINDOW."""
        res = simulate_click(self.calc_hwnd, (-500, -500))
        print(f"\n[DISPATCH TELEMETRY 5 - OUTSIDE_WINDOW]\n  Status: {res.status}\n  Log: {res.action_log}\n  Real Input: {res.real_input_dispatched}")

        self.assertFalse(res.success)
        self.assertEqual(res.status, "ABORT_OUTSIDE_WINDOW")
        self.assertFalse(res.real_input_dispatched)
        self.assertEqual(res.verification.reason, "COORDINATE_OUTSIDE_WINDOW")
        self.assertIn("[PRE-CLICK SAFETY REJECTION]", res.action_log)

    def test_dispatch_scenario_6_invalid_hwnd(self):
        """Task 2 Scenario 6: Invalid/closed HWND -> immediate abort with ABORT_INVALID_HWND."""
        res = simulate_click(0xDEADBEEF, (100, 100))
        print(f"\n[DISPATCH TELEMETRY 6 - INVALID_HWND]\n  Status: {res.status}\n  Log: {res.action_log}\n  Real Input: {res.real_input_dispatched}")

        self.assertFalse(res.success)
        self.assertEqual(res.status, "ABORT_INVALID_HWND")
        self.assertFalse(res.real_input_dispatched)
        self.assertEqual(res.verification.reason, "INVALID_HWND")
        self.assertIn("[PRE-CLICK SAFETY REJECTION]", res.action_log)

    def test_multistep_task_nominal_trace_and_stale_inspection_obstacle(self):
        """
        Task 3: Realistic Multi-Step Simulated Task
        Part A: Nominal workflow: Inspect -> Locate 'Clear' -> Verify -> Log Simulated Click.
        Part B: Stale Inspection Obstacle: Inspect while FG -> Focus shift mid-task ->
                Attempt click with stale coordinates -> Caught & aborted at click-time.
        """
        # --- PART A: Nominal Multi-Step Task ---
        step_trace = []

        # Step 1: Inspect elements
        snap = inspect_screen_elements(self.calc_hwnd)
        step_trace.append(f"Step 1: Inspected screen. Discovered {len(snap.elements)} interactive elements (latency={snap.latency_ms}ms).")

        # Step 2: Locate target element ("clear" button)
        clear_btn = find_element(snap, "clear")
        self.assertIsNotNone(clear_btn, "Clear button must be found")
        step_trace.append(f"Step 2: Located target element '{clear_btn.name}' (id={clear_btn.automation_id}) at center {clear_btn.center}.")

        # Step 3: Dispatch simulated click
        click_res = simulate_click(self.calc_hwnd, clear_btn)
        step_trace.append(f"Step 3: Dispatched simulate_click() -> status={click_res.status}, real_input={click_res.real_input_dispatched}.")

        self.assertTrue(click_res.success)
        self.assertEqual(click_res.status, "SIMULATED_CLICK_SUCCESS")
        self.assertFalse(click_res.real_input_dispatched)
        print("\n[TASK 3 PART A: NOMINAL MULTI-STEP TRACE]")
        for st in step_trace:
            print(f"  {st}")

        # --- PART B: Stale Inspection / Mid-Task Obstacle ---
        stale_trace = []

        # Step 1: Caller inspects elements while Calculator is foreground
        stale_snap = inspect_screen_elements(self.calc_hwnd)
        stale_btn = find_element(stale_snap, "clear")
        stale_trace.append(f"Step 1: Stale caller inspected Calculator and cached button at {stale_btn.center}.")

        # Step 2: Deliberate obstacle introduced mid-task (foreground switched to another window)
        other_hwnd = find_other_desktop_window(self.calc_hwnd)
        self.assertGreater(other_hwnd, 0, "Must locate another desktop window for mid-task obstacle")
        set_window_foreground_passive(other_hwnd)
        time.sleep(0.4)
        stale_trace.append(f"Step 2: Mid-task obstacle introduced! Foreground shifted to HWND {other_hwnd} ('{win32gui.GetWindowText(other_hwnd)}').")

        # Step 3: Stale caller attempts to click using cached element/coordinates
        try:
            stale_click_res = simulate_click(self.calc_hwnd, stale_btn)
            stale_trace.append(f"Step 3: simulate_click() intercepted stale action -> status={stale_click_res.status}, reason={stale_click_res.verification.reason}.")

            self.assertFalse(stale_click_res.success)
            self.assertEqual(stale_click_res.status, "ABORT_NOT_FOREGROUND")
            self.assertFalse(stale_click_res.real_input_dispatched)
            self.assertIn("[PRE-CLICK SAFETY REJECTION]", stale_click_res.action_log)
        finally:
            set_window_foreground_passive(self.calc_hwnd)
            time.sleep(0.3)

        print("\n[TASK 3 PART B: STALE INSPECTION MID-TASK OBSTACLE TRACE]")
        for st in stale_trace:
            print(f"  {st}")


if __name__ == "__main__":
    unittest.main()
