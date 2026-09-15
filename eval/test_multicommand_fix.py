"""
eval/test_multicommand_fix.py
Tests the quoted multi-command parsing fix in miku.py and _parse_intent.
"""
import re
import sys
import unittest

sys.path.insert(0, r"c:\miku")
from tools.miku_inference import _parse_intent


class TestMulticommandFix(unittest.TestCase):
    def test_multicommand_split(self):
        prompt = '"open chrome", "click start", "type hello world", "close"'
        quoted = re.findall(r'"([^"]+)"', prompt)
        expected_split = ["open chrome", "click start", "type hello world", "close"]
        self.assertEqual(quoted, expected_split)

        sub_commands = quoted if len(quoted) >= 2 else [prompt]
        self.assertEqual(sub_commands, expected_split)

    def test_parse_intent_subcommands(self):
        cases = [
            ("open chrome",       "action", "launch"),
            ("click start",       "action", "click"),
            ("click start",       "target", "start"),
            ("type hello world",  "action", "type"),
            ("type hello world",  "text",   "hello world"),
            ("close",             "action", "press_key"),
            ("close",             "key",    "alt+f4"),
        ]
        for cmd, key, expected_val in cases:
            res = _parse_intent(cmd)
            self.assertEqual(res.get(key), expected_val)

        # Check launch action maps to valid path or app name
        res_chrome = _parse_intent("open chrome")
        self.assertEqual(res_chrome.get("action"), "launch")
        self.assertTrue("chrome" in res_chrome.get("target", "").lower())

    def test_single_command_not_split(self):
        single = "open notepad"
        q2 = re.findall(r'"([^"]+)"', single)
        sub2 = q2 if len(q2) >= 2 else [single]
        self.assertEqual(sub2, ["open notepad"])

    def test_no_task_fallback(self):
        for cmd in ["open chrome", "click start", "type hello world", "close"]:
            result = _parse_intent(cmd)
            self.assertNotEqual(result.get("action"), "task")


if __name__ == "__main__":
    unittest.main()
