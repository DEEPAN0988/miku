"""
eval/test_close_app.py — Unit Tests for Safe Application Closing & Process Guardrails
"""

import unittest
from unittest.mock import patch, MagicMock
from tools.app_launcher import close_app, PROTECTED_PROCESSES
from tools.dispatcher import dispatch_tool, ToolCall, RiskLevel


class TestCloseApp(unittest.TestCase):
    def test_protected_apps_guardrail(self):
        for app in ("antigravity", "python", "explorer", "dwm.exe"):
            res = close_app(app)
            self.assertEqual(res["status"], "PROTECTED")
            self.assertFalse(res["executed"])
            self.assertIn("protected", res["output"].lower())

    def test_empty_app_name_and_no_pid(self):
        res = close_app("")
        self.assertEqual(res["status"], "ERROR")
        self.assertFalse(res["executed"])

    def test_dispatcher_integration_hitl_blocked(self):
        tc = ToolCall(action="Close App", argument="notepad", risk=RiskLevel.HIGH, is_valid=True)
        res = dispatch_tool(tc, hitl_confirmed=False)
        self.assertEqual(res.status, "DRY_RUN_PENDING_HITL")
        self.assertFalse(res.executed)

    @patch("tools.app_launcher.close_app")
    def test_dispatcher_integration_hitl_confirmed(self, mock_close):
        mock_close.return_value = {
            "status": "SUCCESS",
            "app_name": "notepad",
            "executed": True,
            "output": "Closed notepad successfully.",
        }
        tc = ToolCall(action="Close App", argument="notepad", risk=RiskLevel.HIGH, is_valid=True)
        res = dispatch_tool(tc, hitl_confirmed=True)
        self.assertEqual(res.status, "SUCCESS")
        self.assertTrue(res.executed)
        mock_close.assert_called_once_with("notepad")


if __name__ == "__main__":
    unittest.main()
