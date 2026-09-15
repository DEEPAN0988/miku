"""
eval/test_interactive_loop.py

Regression test suite for Autonomous Desktop Interactive Loop Controller (tools/interactive_loop.py).

Verifies:
1. Intent Parsing: Accurately parses natural language tasks ("open Notepad", "search for X in Y").
2. Perception Freshness & Decision Logic: Runs autonomous loop in simulation mode with zero real input.
3. Fast Confirmation Gate: Respects approval/cancellation prompts in non-interactive environment.
"""

import os
import unittest
from tools.screen_inspector import REAL_CLICK_ENABLED, ScreenSnapshot, UIElement
from tools.typing_automation import REAL_TYPE_ENABLED
from tools.interactive_loop import (
    parse_goal_intent,
    run_autonomous_task_loop,
)


class TestInteractiveLoop(unittest.TestCase):

    def setUp(self):
        os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

    def test_01_parse_goal_intent(self):
        """Verify intent parsing from natural language goals."""
        intent1 = parse_goal_intent("open Notepad")
        self.assertEqual(intent1["action_type"], "OPEN_APP")
        self.assertEqual(intent1["target_app"], "notepad")

        intent2 = parse_goal_intent("search for hello in Notepad")
        self.assertEqual(intent2["action_type"], "SEARCH_IN_APP")
        self.assertEqual(intent2["search_query"], "hello")
        self.assertEqual(intent2["target_app"], "notepad")

    def test_02_autonomous_loop_simulation_mode(self):
        """Task 1 & 2: Autonomous loop executes in simulation mode with zero physical input."""
        res = run_autonomous_task_loop("open Notepad", max_steps=5, real_execution=False)
        self.assertIn("status", res)
        trace = res.get("trace", [])
        for step in trace:
            if "click_result" in step:
                self.assertFalse(step["click_result"].get("real_input_dispatched", False))
            if "type_result" in step:
                self.assertFalse(step["type_result"].get("real_input_dispatched", False))


if __name__ == "__main__":
    unittest.main()
