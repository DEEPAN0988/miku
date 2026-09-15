"""
eval/test_desktop_operator.py

Automated regression suite for Desktop Operator Subsystem (tools/desktop_operator.py).

Verifies:
1. Baseline Safety Invariants: REAL_CLICK_ENABLED and REAL_TYPE_ENABLED fail closed (False).
2. Capability (A) Simulation: Opening app by icon/shortcut in dry-run mode (zero physical input).
3. Capability (B) Simulation: Start Menu search and click flow in dry-run mode (zero physical input).
"""

import unittest
from tools.screen_inspector import REAL_CLICK_ENABLED, inspect_screen_elements, UIElement, ScreenSnapshot
from tools.typing_automation import REAL_TYPE_ENABLED
from tools.desktop_operator import (
    find_search_box_element,
    find_app_icon_or_shortcut,
    simulate_open_app_by_icon,
    simulate_start_menu_search_and_click,
)


class TestDesktopOperator(unittest.TestCase):

    def test_01_circuit_breaker_baselines(self):
        """Constitutional Safety Invariant: Circuit breakers must be False by default."""
        self.assertFalse(REAL_CLICK_ENABLED, "CRITICAL FAULT: REAL_CLICK_ENABLED is not False!")
        self.assertFalse(REAL_TYPE_ENABLED, "CRITICAL FAULT: REAL_TYPE_ENABLED is not False!")

    def test_02_find_search_box_element(self):
        """Verify search box element finder logic on synthetic ScreenSnapshot."""
        edit_el = UIElement(
            control_type="Edit",
            name="Search box",
            automation_id="SearchTextBox",
            rect=(100, 100, 300, 130),
            center=(200, 115),
        )
        snap = ScreenSnapshot(
            hwnd=12345,
            title="Search",
            class_name="CoreWindow",
            process_name="searchhost.exe",
            window_rect=(0, 0, 800, 600),
            elements=[edit_el],
        )
        found = find_search_box_element(snap)
        self.assertIsNotNone(found)
        self.assertEqual(found.control_type, "Edit")
        self.assertEqual(found.automation_id, "SearchTextBox")

    def test_03_find_app_icon_or_shortcut(self):
        """Verify app icon finder logic matching target app names."""
        app_el = UIElement(
            control_type="Button",
            name="Notepad",
            automation_id="AppNotepad",
            rect=(50, 50, 100, 100),
            center=(75, 75),
            is_enabled=True,
        )
        snap = ScreenSnapshot(
            hwnd=12345,
            title="Start",
            class_name="CoreWindow",
            process_name="explorer.exe",
            window_rect=(0, 0, 800, 600),
            elements=[app_el],
        )
        found = find_app_icon_or_shortcut(snap, "notepad")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Notepad")

    def test_04_simulate_start_menu_search_trace(self):
        """Task 2: Full simulated search trace executes safely with zero real input."""
        res = simulate_start_menu_search_and_click(query="notepad", target_app_name="Notepad")
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "SIMULATED_SEARCH_TRACE_SUCCESS")
        self.assertFalse(res["real_input_dispatched"])

        # Check trace structure
        trace = res["trace"]
        self.assertGreaterEqual(len(trace), 4)
        for step in trace:
            if "click_result" in step:
                self.assertFalse(step["click_result"]["real_input_dispatched"])
            if "type_result" in step:
                self.assertFalse(step["type_result"]["real_input_dispatched"])


if __name__ == "__main__":
    unittest.main()
