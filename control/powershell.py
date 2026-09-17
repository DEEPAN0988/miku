"""
Safe PowerShell & CLI execution engine for Windows.
Gated by the RiskGate security layer.
"""

import subprocess
import os
from typing import Dict, Any, Tuple
from .gate import RiskGate, RiskLevel


class PowerShellEngine:
    def __init__(self, default_timeout_sec: int = 15):
        self.default_timeout_sec = default_timeout_sec

    def execute(self, script: str, confirmed: bool = False, timeout: int = None) -> Dict[str, Any]:
        """
        Execute a PowerShell command with risk check and timeout.
        """
        timeout = timeout or self.default_timeout_sec
        risk, reason = RiskGate.evaluate("powershell", script)

        if risk == RiskLevel.BLOCKED:
            return {
                "success": False,
                "risk": risk.value,
                "output": "",
                "error": reason,
                "blocked": True
            }

        if risk == RiskLevel.REQUIRES_CONFIRMATION and not confirmed:
            return {
                "success": False,
                "risk": risk.value,
                "output": "",
                "error": f"Confirmation required: {reason}",
                "requires_confirmation": True,
                "payload": script
            }

        try:
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": proc.returncode == 0,
                "risk": risk.value,
                "return_code": proc.returncode,
                "output": proc.stdout.strip(),
                "error": proc.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "risk": risk.value,
                "output": "",
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "risk": risk.value,
                "output": "",
                "error": str(e)
            }
