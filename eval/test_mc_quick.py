"""
eval/test_mc_quick.py
Quick smoke test — no model loading required (calls _parse_intent directly).
"""
import re
import sys
import unittest

sys.path.insert(0, r"c:\miku")
from tools.miku_inference import _parse_intent


class TestMCQuick(unittest.TestCase):
    def test_quick_parse_intent(self):
        prompt = '"open chrome", "click start", "type hello world", "close"'
        quoted = re.findall(r'"([^"]+)"', prompt)
        expected = ["open chrome", "click start", "type hello world", "close"]
        self.assertEqual(quoted, expected)

        sub_commands = quoted if len(quoted) >= 2 else [prompt]
        self.assertEqual(sub_commands, expected)

        res_chrome = _parse_intent("open chrome")
        self.assertEqual(res_chrome.get("action"), "launch")
        self.assertTrue("chrome" in res_chrome.get("target", "").lower())

        res_start = _parse_intent("click start")
        self.assertEqual(res_start.get("action"), "click")
        self.assertEqual(res_start.get("target"), "start")

        res_type = _parse_intent("type hello world")
        self.assertEqual(res_type.get("action"), "type")
        self.assertEqual(res_type.get("text"), "hello world")

        res_close = _parse_intent("close")
        self.assertEqual(res_close.get("action"), "press_key")
        self.assertEqual(res_close.get("key"), "alt+f4")


if __name__ == "__main__":
    unittest.main()
