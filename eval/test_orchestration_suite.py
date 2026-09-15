"""
eval/test_orchestration_suite.py — Unit & Integration Test Suite for Desktop Automation Orchestration Layer

Tests:
  1. config.py strict safety defaults.
  2. human_gate.py FastConfirm token issuance, denial, timeout, manual edit mode, visual highlighter, and rationale formatting.
  3. visual_highlighter.py TargetHighlighter daemon thread and overlay lifecycle.
  4. ui_inspector.py TreeInspector node lookup, read_element_text, and strict ElementNotFoundError logic.
  5. safe_executor.py ActionDispatch token authorization, consumption, and replay rejection.
  6. agent_loop.py end-to-end task runner with read_screen_text FastConfirm bypass and manual recovery.
  7. llm_bridge.py MikuOrchestrationBridge tool execution and error handling.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("."))

from config import OrchestratorConfig, default_config
from human_gate import (
    ActionDeniedError,
    AuthToken,
    ConfirmationTimeoutError,
    FastConfirm,
)
from safe_executor import ActionDispatch, UnauthorizedActionError
from ui_inspector import ElementNotFoundError, TreeInspector
from agent_loop import AgentLoop, PipelineHaltedError, TaskSpec
from visual_highlighter import TargetHighlighter
from cli_macro_engine import RegexCommandParser


class TestOrchestratorConfig(unittest.TestCase):
    """Test safety constraints in config.py."""

    def test_strict_safety_defaults(self):
        cfg = OrchestratorConfig()
        self.assertTrue(cfg.CONFIRMATION_GATE_ENABLED, "FastConfirm gate MUST default to ON.")
        self.assertFalse(cfg.ENABLE_AUTO_RETRIES, "Auto-retries MUST default to False.")
        self.assertGreater(cfg.CONFIRMATION_TIMEOUT_SECONDS, 0)
        self.assertGreater(cfg.MAX_EXECUTION_STEPS_PER_RUN, 0)


class TestTargetHighlighter(unittest.TestCase):
    """Test TargetHighlighter non-blocking thread lifecycle."""

    def setUp(self):
        self.highlighter = TargetHighlighter(color="red", line_width=4, enabled=True)

    def tearDown(self):
        self.highlighter.close()

    def test_highlighter_start_stop(self):
        self.highlighter.start_highlight(x=100, y=200, bounds=(50, 150, 150, 250), label="TestNode")
        self.assertTrue(self.highlighter._running)
        self.highlighter.stop_highlight()


class TestFastConfirm(unittest.TestCase):
    """Test FastConfirm gate token authorization, rejection, and rationale intent rendering."""

    def test_confirmation_approval(self):
        async def run_test():
            def mock_input(prompt):
                return "y"

            hl = TargetHighlighter(enabled=False)
            gate = FastConfirm(input_handler=mock_input, highlighter=hl)
            token = await gate.request_confirmation(
                action_type="click",
                target="Submit",
                coords=(450, 800),
                bounds=(400, 780, 500, 820),
                rationale="Save form before exiting.",
            )
            self.assertTrue(token.approved)
            self.assertEqual(token.action_type, "click")
            self.assertEqual(token.coords, (450, 800))
            self.assertEqual(token.rationale, "Save form before exiting.")
            self.assertFalse(token.is_used)

        asyncio.run(run_test())

    def test_rationale_terminal_prompt_formatting(self):
        async def run_test():
            captured_prompts = []

            def mock_input(prompt):
                captured_prompts.append(prompt)
                return "y"

            hl = TargetHighlighter(enabled=False)
            gate = FastConfirm(input_handler=mock_input, highlighter=hl)
            await gate.request_confirmation(
                action_type="click",
                target="SaveBtn",
                coords=(450, 800),
                rationale="User requested saving configuration.",
            )
            self.assertEqual(len(captured_prompts), 1)
            prompt_text = captured_prompts[0]
            self.assertIn("[MIKU'S INTENT]: User requested saving configuration.", prompt_text)
            self.assertIn("[ACTION REQUIRED] Agent wants to click 'SaveBtn'", prompt_text)

        asyncio.run(run_test())

    def test_confirmation_denied(self):
        async def run_test():
            def mock_input(prompt):
                return "n"

            hl = TargetHighlighter(enabled=False)
            gate = FastConfirm(input_handler=mock_input, highlighter=hl)
            with self.assertRaises(ActionDeniedError):
                await gate.request_confirmation(
                    action_type="click",
                    target="Submit",
                    coords=(450, 800),
                )

        asyncio.run(run_test())

    def test_confirmation_timeout(self):
        async def run_test():
            cfg = OrchestratorConfig(CONFIRMATION_TIMEOUT_SECONDS=0.1)

            async def slow_read(prompt):
                await asyncio.sleep(0.5)
                return "y"

            hl = TargetHighlighter(enabled=False)
            gate = FastConfirm(config=cfg, highlighter=hl)
            gate._read_terminal_input = slow_read  # type: ignore

            with self.assertRaises(ConfirmationTimeoutError):
                await gate.request_confirmation(
                    action_type="click",
                    target="Submit",
                    coords=(450, 800),
                )

        asyncio.run(run_test())

    def test_confirmation_manual_edit(self):
        async def run_test():
            inputs = ["e", "500", "850"]
            input_idx = 0

            def mock_input(prompt):
                nonlocal input_idx
                val = inputs[input_idx]
                input_idx += 1
                return val

            hl = TargetHighlighter(enabled=False)
            gate = FastConfirm(input_handler=mock_input, highlighter=hl)
            token = await gate.request_confirmation(
                action_type="click",
                target="Submit",
                coords=(450, 800),
            )
            self.assertTrue(token.approved)
            self.assertTrue(token.is_edited)
            self.assertEqual(token.coords, (500, 850))

        asyncio.run(run_test())


class TestTreeInspector(unittest.TestCase):
    """Test TreeInspector UIA tree resolution and read_element_text."""

    def setUp(self):
        self.mock_nodes = [
            {"name": "Submit", "automation_id": "btn_submit", "control_type": "Button", "bounds": (100, 200, 180, 240), "center": (140, 220), "text": "Submit Form"},
            {"name": "DocPane", "automation_id": "pane_doc", "control_type": "Pane", "bounds": (0, 0, 500, 500), "center": (250, 250), "text": "Document Header"},
            {"name": "Username", "automation_id": "txt_user", "control_type": "Edit", "bounds": (50, 50, 250, 80), "center": (150, 65), "text": "john_doe"},
        ]
        self.inspector = TreeInspector(mock_elements=self.mock_nodes)

    def test_find_element_success(self):
        res = self.inspector.find_element_by_name("Submit")
        self.assertEqual(res["name"], "Submit")
        self.assertEqual(res["center"], (140, 220))

    def test_partial_and_automation_id_substring_match(self):
        # Match by partial substring in name
        res_sub = self.inspector.find_element_by_name("subm")
        self.assertEqual(res_sub["name"], "Submit")

        # Match by partial substring in automation_id
        res_id = self.inspector.find_element_by_name("txt_u")
        self.assertEqual(res_id["name"], "Username")

    def test_read_element_text_single(self):
        res = self.inspector.read_element_text("Submit")
        self.assertEqual(res["target"], "Submit")
        self.assertEqual(res["primary_text"], "Submit Form")

    def test_read_element_text_container(self):
        res = self.inspector.read_element_text("DocPane")
        self.assertEqual(res["target"], "DocPane")
        self.assertIn("Username", res["children_text"])
        self.assertIn("Submit", res["children_text"])

    def test_missing_element_throws(self):
        with self.assertRaises(ElementNotFoundError):
            self.inspector.find_element_by_name("NonExistentButton", timeout=0.05, poll_interval=0.01)

    def test_implicit_auto_wait_polling(self):
        # Verify custom timeout parameter accepted without error
        res = self.inspector.find_element_by_name("Submit", timeout=1.0, poll_interval=0.01)
        self.assertEqual(res["name"], "Submit")



class TestSafeExecutor(unittest.TestCase):
    """Test ActionDispatch token validation and execution safety."""

    def setUp(self):
        self.executor = ActionDispatch(simulation_mode=True)

    def test_valid_click_execution(self):
        token = AuthToken(
            token_id="tok-123",
            action_type="click",
            target="Submit",
            coords=(140, 220),
            approved=True,
        )
        res = self.executor.click(token)
        self.assertEqual(res["action"], "click")
        self.assertTrue(token.is_used)

    def test_unapproved_token_rejected(self):
        token = AuthToken(
            token_id="tok-unapproved",
            action_type="click",
            target="Submit",
            coords=(140, 220),
            approved=False,
        )
        with self.assertRaises(UnauthorizedActionError):
            self.executor.click(token)

    def test_token_replay_rejected(self):
        token = AuthToken(
            token_id="tok-replay",
            action_type="click",
            target="Submit",
            coords=(140, 220),
            approved=True,
        )
        self.executor.click(token)
        self.assertTrue(token.is_used)

        with self.assertRaises(UnauthorizedActionError):
            self.executor.click(token)

    def test_action_type_mismatch_rejected(self):
        token = AuthToken(
            token_id="tok-mismatch",
            action_type="click",
            target="Submit",
            coords=(140, 220),
            approved=True,
        )
        with self.assertRaises(UnauthorizedActionError):
            self.executor.type(token)


class TestAgentLoop(unittest.TestCase):
    """Test AgentLoop async pipeline execution including read_screen_text bypass."""

    def test_read_screen_text_bypasses_fast_confirm(self):
        async def run_test():
            mock_nodes = [
                {"name": "DocPane", "automation_id": "pane_doc", "control_type": "Pane", "bounds": (0, 0, 500, 500), "center": (250, 250), "text": "Secret Document Text"},
            ]

            input_called = False

            def mock_input(prompt):
                nonlocal input_called
                input_called = True
                return "y"

            hl = TargetHighlighter(enabled=False)
            inspector = TreeInspector(mock_elements=mock_nodes)
            gate = FastConfirm(input_handler=mock_input, highlighter=hl)
            executor = ActionDispatch(simulation_mode=True)
            runner = AgentLoop(inspector=inspector, gate=gate, executor=executor)

            tasks = [
                {
                    "action": "read_screen_text",
                    "target": "DocPane",
                    "rationale": "Read document text passively.",
                }
            ]

            results = await runner.run(tasks)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "success")
            self.assertIn("Secret Document Text", results[0]["data"]["text"])
            self.assertFalse(input_called, "FastConfirm gate MUST be bypassed for read-only actions!")

        asyncio.run(run_test())

    def test_open_action_bypasses_tree_inspector(self):
        async def run_test():
            def mock_input(prompt):
                return "y"

            hl = TargetHighlighter(enabled=False)
            inspector = TreeInspector(mock_elements=[])  # Empty tree (closed app)
            gate = FastConfirm(input_handler=mock_input, highlighter=hl)
            executor = ActionDispatch(simulation_mode=True)
            runner = AgentLoop(inspector=inspector, gate=gate, executor=executor)

            tasks = [
                TaskSpec(
                    action_type="open",
                    target_name="notepad",
                    rationale="Launch notepad directly via OS.",
                )
            ]

            results = await runner.run(tasks)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "success")
            self.assertEqual(results[0]["exec_result"]["action"], "open")
            self.assertEqual(results[0]["exec_result"]["target"], "notepad")

        asyncio.run(run_test())



class TestRegexCommandParser(unittest.TestCase):
    """Test RegexCommandParser deterministic CLI parsing."""

    def setUp(self):
        self.parser = RegexCommandParser()

    def test_click_command_parsing(self):
        task = self.parser.parse("click Start")
        self.assertIsNotNone(task)
        self.assertEqual(task.action_type, "click")
        self.assertEqual(task.target_name, "Start")

    def test_type_command_parsing(self):
        task = self.parser.parse('type "Notepad" into "Search"')
        self.assertIsNotNone(task)
        self.assertEqual(task.action_type, "type")
        self.assertEqual(task.target_name, "Search")
        self.assertEqual(task.payload, "Notepad")

    def test_read_command_parsing(self):
        task = self.parser.parse("read Screen")
        self.assertIsNotNone(task)
        self.assertEqual(task.action_type, "read_screen_text")
        self.assertEqual(task.target_name, "Screen")

    def test_open_command_parsing(self):
        task = self.parser.parse('open "code"')
        self.assertIsNotNone(task)
        self.assertEqual(task.action_type, "open")
        self.assertEqual(task.target_name, "code")

    def test_invalid_syntax_returns_none(self):
        self.assertIsNone(self.parser.parse("open Notepad and type hello"))




if __name__ == "__main__":
    unittest.main()
