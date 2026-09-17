"""
Risk Gate & Confirmation Policy for Miku Control Layer.
Implements Section 3 (Non-Goals) & Section 7 (Security & Confirmation Gates) from PRD.
"""

from enum import Enum
from typing import Dict, Any, Tuple


class RiskLevel(str, Enum):
    SAFE = "safe"                          # Automatically allowed (e.g. read time, open notepad, check battery)
    REQUIRES_CONFIRMATION = "requires_confirmation"  # Must obtain spoken/typed user consent (e.g. delete file, kill process)
    BLOCKED = "blocked"                    # Permanently forbidden (e.g. CAPTCHA solving, silent pairing, payment bypass)


class RiskGate:
    # Explicitly blocked keywords per PRD non-goals
    BLOCKED_PATTERNS = [
        "captcha", "recaptcha", "hcaptcha",
        "silent_pair", "bypass_auth", "steal_token",
        "rmdir /s /q c:\\", "remove-item -recurse c:\\windows"
    ]

    # Risky patterns requiring confirmation
    RISKY_PATTERNS = [
        "delete", "remove-item", "del ", "erase", "format",
        "stop-process", "kill", "taskkill", "shutdown", "restart",
        "net user", "reg delete", "drop database", "purchase",
        "buy ", "checkout", "transfer funds", "login", "password"
    ]

    # Safe allowlist operations that execute directly without nagging
    SAFE_COMMANDS = [
        "notepad", "calc", "calculator", "mspaint", "explorer",
        "get-date", "get-time", "get-process", "ipconfig",
        "tasklist", "dir", "ls", "echo", "volume"
    ]

    @classmethod
    def evaluate(cls, action_type: str, command_payload: str) -> Tuple[RiskLevel, str]:
        """
        Evaluate risk level of an action.
        Returns (RiskLevel, explanation).
        """
        lower_payload = command_payload.lower().strip()

        # 1. Check blocked
        for pattern in cls.BLOCKED_PATTERNS:
            if pattern in lower_payload:
                return (
                    RiskLevel.BLOCKED,
                    f"Action permanently blocked per safety constraints: '{pattern}' is prohibited."
                )

        # 2. Check risky / confirmation required
        for pattern in cls.RISKY_PATTERNS:
            if pattern in lower_payload:
                return (
                    RiskLevel.REQUIRES_CONFIRMATION,
                    f"Action '{command_payload}' involves sensitive or destructive operations ('{pattern}'). Explicit confirmation required."
                )

        if action_type in ["file_delete", "process_kill", "system_admin", "device_pair"]:
            return (
                RiskLevel.REQUIRES_CONFIRMATION,
                f"Action category '{action_type}' requires explicit confirmation."
            )

        # 3. Otherwise safe
        return (RiskLevel.SAFE, "Action is within safe operation parameters.")
