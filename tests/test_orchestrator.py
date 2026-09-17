import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.router import Orchestrator
from nlu.engine import NLUEngine


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = Orchestrator(tts_enabled=False)
        self.nlu = NLUEngine()

    def test_orchestrator_dialogue_and_time(self):
        # Time command
        parsed_time = self.nlu.parse("what is the time")
        res_time = self.orch.handle_intent(parsed_time)
        self.assertTrue(res_time["success"])
        self.assertIn("current time", res_time["response"].lower())

    def test_orchestrator_confirmation_flow(self):
        # Trigger destructive command
        del_cmd = self.nlu.parse("delete file dummy_report.txt")
        res_del = self.orch.handle_intent(del_cmd)
        self.assertTrue(res_del.get("requires_confirmation", False))
        self.assertTrue(self.orch.dialogue.has_pending_confirmation())

        # Reject confirmation
        no_cmd = self.nlu.parse("no")
        res_no = self.orch.handle_intent(no_cmd)
        self.assertEqual(res_no["action"], "cancelled")
        self.assertFalse(self.orch.dialogue.has_pending_confirmation())


if __name__ == "__main__":
    unittest.main()
