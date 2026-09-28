"""
OS-Level Automation and Action Execution.
Core 3: Execution Engine (Hands)
PRD §3.2 & §7: Win32 SendInput, App Control, Confirmation Gating.
"""
import os
import sys
import subprocess
import time
from typing import Dict, Any, Tuple, Optional

class OSActionsExecutor:
    def __init__(self):
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}

    def launch_app(self, app_name: str) -> Tuple[bool, str]:
        app_map = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "explorer": "explorer.exe",
            "paint": "mspaint.exe"
        }
        target = app_map.get(app_name.lower(), app_name)
        try:
            subprocess.Popen(target, shell=True)
            return True, f"Launched {app_name}"
        except Exception as e:
            return False, f"Failed to launch {app_name}: {str(e)}"

    def close_app(self, app_name: str) -> Tuple[bool, str]:
        try:
            if sys.platform == "win32":
                subprocess.run(f"taskkill /IM {app_name}* /F", shell=True, capture_output=True)
                return True, f"Closed {app_name}"
            return True, f"Simulated close of {app_name}"
        except Exception as e:
            return False, f"Failed to close {app_name}: {str(e)}"

    def adjust_volume(self, direction: str) -> Tuple[bool, str]:
        # Using PowerShell audio toggle / volume commands
        try:
            if direction == "up":
                subprocess.run("powershell -Command \"(New-Object -ComObject WScript.Shell).SendKeys([char]175)\"", shell=True)
            elif direction == "down":
                subprocess.run("powershell -Command \"(New-Object -ComObject WScript.Shell).SendKeys([char]174)\"", shell=True)
            elif direction in ("mute", "unmute"):
                subprocess.run("powershell -Command \"(New-Object -ComObject WScript.Shell).SendKeys([char]173)\"", shell=True)
            return True, f"Adjusted volume: {direction}"
        except Exception as e:
            return False, f"Volume error: {str(e)}"

    def execute_destructive_action(self, action_id: str, action_type: str, path: str, confirmed: bool) -> Tuple[bool, str]:
        """
        PRD §7: Explicit confirmation gate for any file deletion or administrative task.
        """
        if not confirmed:
            self.pending_confirmations[action_id] = {
                "action": action_type,
                "path": path,
                "time": time.time()
            }
            return False, f"CONFIRMATION_REQUIRED: Are you sure you want to delete '{path}'? Spoken or typed confirmation required."

        if action_id in self.pending_confirmations:
            del self.pending_confirmations[action_id]

        if action_type == "delete_file":
            try:
                if os.path.exists(path):
                    os.remove(path)
                    return True, f"Deleted file '{path}' after user confirmation."
                return False, f"File not found: '{path}'"
            except Exception as e:
                return False, f"Delete failed: {str(e)}"

        return True, "Executed with confirmation"
