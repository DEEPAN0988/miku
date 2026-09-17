import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.gate import RiskGate, RiskLevel
from control.powershell import PowerShellEngine


class TestControl(unittest.TestCase):
    def test_risk_gate_evaluations(self):
        # Blocked
        risk, reason = RiskGate.evaluate("general", "solve captcha now")
        self.assertEqual(risk, RiskLevel.BLOCKED)

        # Requires confirmation
        risk, reason = RiskGate.evaluate("general", "delete file project.txt")
        self.assertEqual(risk, RiskLevel.REQUIRES_CONFIRMATION)

        # Safe
        risk, reason = RiskGate.evaluate("general", "get-date")
        self.assertEqual(risk, RiskLevel.SAFE)

    def test_powershell_execution_and_gating(self):
        ps = PowerShellEngine()

        # Safe execution
        res = ps.execute("Write-Output 'Miku Online'")
        self.assertTrue(res["success"])
        self.assertIn("Miku Online", res["output"])

        # Blocked execution
        blocked_res = ps.execute("solve captcha token")
        self.assertTrue(blocked_res.get("blocked", False))

        # Confirmation required execution without confirmed=True
        risky_res = ps.execute("Remove-Item -Path test.txt", confirmed=False)
        self.assertTrue(risky_res.get("requires_confirmation", False))


if __name__ == "__main__":
    unittest.main()
