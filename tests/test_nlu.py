import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlu.rules import RuleMatcher
from nlu.engine import NLUEngine


class TestNLU(unittest.TestCase):
    def test_rule_matching_compositional(self):
        res = RuleMatcher.match("open notepad and write an essay about machine learning")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "compositional_app_write")
        self.assertEqual(res["entities"]["app"], "notepad")
        self.assertEqual(res["entities"]["document_type"], "essay")
        self.assertIn("machine learning", res["entities"]["topic"])

    def test_rule_matching_time_and_date(self):
        t_res = RuleMatcher.match("what is the time")
        self.assertIsNotNone(t_res)
        self.assertEqual(t_res["intent"], "get_time")

        d_res = RuleMatcher.match("what day is it today")
        self.assertIsNotNone(d_res)
        self.assertEqual(d_res["intent"], "get_date")

    def test_rule_matching_tasks(self):
        res = RuleMatcher.match("create task buy groceries with priority high")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "create_task")
        self.assertEqual(res["entities"]["title"], "buy groceries")
        self.assertEqual(res["entities"]["priority"], "high")

    def test_rule_matching_destructive_gated(self):
        res = RuleMatcher.match("delete file c:\\temp\\old.log")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "delete_file")
        self.assertIn("old.log", res["entities"]["target"])

    def test_intent_classifier_and_unified_engine(self):
        nlu = NLUEngine()
        parsed = nlu.parse("open calculator")
        self.assertEqual(parsed["intent"], "launch_app")
        self.assertGreater(parsed["confidence"], 0.4)


if __name__ == "__main__":
    unittest.main()
