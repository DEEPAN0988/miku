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

import shutil

class OSActionsExecutor:
    APP_MAP = {
        "notepad": "notepad.exe",
        "notes": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "cmd": "cmd.exe",
        "command prompt": "cmd.exe",
        "powershell": "powershell.exe",
        "pwsh": "powershell.exe",
        "terminal": "wt.exe",
        "windows terminal": "wt.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "files": "explorer.exe",
        "paint": "mspaint.exe",
        "mspaint": "mspaint.exe",
        "edge": "microsoft-edge:",
        "microsoft edge": "microsoft-edge:",
        "msedge": "microsoft-edge:",
        "browser": "microsoft-edge:",
        "web browser": "microsoft-edge:",
        "internet": "microsoft-edge:",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "code": "code",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "cursor": "cursor",
        "taskmgr": "taskmgr.exe",
        "task manager": "taskmgr.exe",
        "settings": "ms-settings:",
        "windows settings": "ms-settings:",
        "control panel": "control.exe",
    }

    PROCESS_MAP = {
        "calculator": ["CalculatorApp.exe", "calc.exe"],
        "calc": ["CalculatorApp.exe", "calc.exe"],
        "notepad": ["notepad.exe"],
        "notes": ["notepad.exe"],
        "edge": ["msedge.exe"],
        "microsoft edge": ["msedge.exe"],
        "msedge": ["msedge.exe"],
        "browser": ["msedge.exe", "chrome.exe"],
        "chrome": ["chrome.exe"],
        "google chrome": ["chrome.exe"],
        "paint": ["mspaint.exe"],
        "mspaint": ["mspaint.exe"],
        "cmd": ["cmd.exe"],
        "command prompt": ["cmd.exe"],
        "powershell": ["powershell.exe", "pwsh.exe"],
        "terminal": ["WindowsTerminal.exe", "wt.exe"],
        "windows terminal": ["WindowsTerminal.exe", "wt.exe"],
        "vscode": ["Code.exe"],
        "code": ["Code.exe"],
        "vs code": ["Code.exe"],
        "visual studio code": ["Code.exe"],
        "cursor": ["cursor.exe"],
        "task manager": ["taskmgr.exe"],
        "taskmgr": ["taskmgr.exe"],
    }

    def __init__(self):
        self.pending_confirmations: Dict[str, Dict[str, Any]] = {}

    def launch_app(self, app_name: str) -> Tuple[bool, str]:
        from miku.core2_cognitive.app_catalog import predict_app
        canonical, info = predict_app(app_name)
        display_name = info.get("display_name", canonical.capitalize())

        # 1. Native protocol scheme (e.g. instagram:, whatsapp:, spotify:, calc:, ms-settings:)
        if "protocol" in info:
            proto = info["protocol"]
            try:
                if sys.platform == "win32" and hasattr(os, "startfile"):
                    os.startfile(proto)
                    return True, f"Launched {display_name}"
            except Exception:
                pass  # Fall back to URL or executable

        # 1b. Known installed application paths (e.g. Wuthering Waves, Steam games)
        if "paths" in info:
            for p in info["paths"]:
                if os.path.exists(p):
                    try:
                        if sys.platform == "win32" and hasattr(os, "startfile"):
                            os.startfile(p)
                            return True, f"Launched {display_name}"
                        else:
                            subprocess.Popen([p], cwd=os.path.dirname(p))
                            return True, f"Launched {display_name}"
                    except Exception:
                        pass

        # 2. Local executable (e.g. notepad.exe, calc.exe, code)
        if "executable" in info:
            exe = info["executable"]
            if sys.platform == "win32" and hasattr(os, "startfile"):
                try:
                    os.startfile(exe)
                    return True, f"Launched {display_name}"
                except Exception:
                    pass
            bin_path = shutil.which(exe) or shutil.which(f"{exe}.exe")
            if bin_path:
                try:
                    subprocess.Popen([bin_path], shell=False)
                    return True, f"Launched {display_name}"
                except Exception:
                    pass

        # 3. Web Service URL (e.g. https://www.instagram.com, https://www.youtube.com)
        if "url" in info:
            url = info["url"]
            try:
                import webbrowser
                webbrowser.open(url)
                return True, f"Opened {display_name} ({url})"
            except Exception:
                if sys.platform == "win32" and hasattr(os, "startfile"):
                    os.startfile(url)
                    return True, f"Opened {display_name} ({url})"

        # 4. Fallback to legacy APP_MAP or general executable discovery
        target = self.APP_MAP.get(canonical, self.APP_MAP.get(app_name.lower().strip(), canonical))

        if ":" in target and not target.endswith(".exe"):
            try:
                if sys.platform == "win32" and hasattr(os, "startfile"):
                    os.startfile(target)
                else:
                    subprocess.Popen(f'start "" "{target}"', shell=True)
                return True, f"Launched {display_name}"
            except Exception as e:
                return False, f"Failed to launch {display_name}: {str(e)}"

        if sys.platform == "win32" and hasattr(os, "startfile"):
            try:
                os.startfile(target)
                return True, f"Launched {display_name}"
            except Exception:
                if not target.endswith(".exe"):
                    try:
                        os.startfile(f"{target}.exe")
                        return True, f"Launched {display_name}"
                    except Exception:
                        pass

        try:
            bin_path = shutil.which(target) or shutil.which(f"{target}.exe")
            if bin_path:
                subprocess.Popen([bin_path], shell=False)
                return True, f"Launched {display_name}"

            if sys.platform == "win32":
                proc = subprocess.run(f'start "" "{target}"', shell=True, capture_output=True)
                if proc.returncode == 0:
                    return True, f"Launched {display_name}"
                return False, f"Failed to launch {display_name}: application '{target}' not found."
            else:
                subprocess.Popen(target, shell=True)
                return True, f"Launched {display_name}"
        except Exception as e:
            return False, f"Failed to launch {display_name}: {str(e)}"

    def close_app(self, app_name: str) -> Tuple[bool, str]:
        clean_name = app_name.strip().lower()
        if clean_name.startswith("the "):
            clean_name = clean_name[4:].strip()
        if clean_name.endswith(" application"):
            clean_name = clean_name[:-12].strip()
        elif clean_name.endswith(" app"):
            clean_name = clean_name[:-4].strip()

        # Generic window close via Alt+F4
        if clean_name in ("window", "active window", "current window", "app"):
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        "powershell -Command \"(New-Object -ComObject WScript.Shell).SendKeys('%{F4}')\"",
                        shell=True,
                        capture_output=True
                    )
                    return True, "Closed active window"
                return True, "Simulated close of active window"
            except Exception as e:
                return False, f"Failed to close window: {str(e)}"

        from miku.core2_cognitive.app_catalog import predict_app
        canonical, info = predict_app(clean_name)
        display_name = info.get("display_name", canonical.capitalize())

        targets = info.get("close_process") or self.PROCESS_MAP.get(canonical) or self.PROCESS_MAP.get(clean_name) or [f"{canonical}.exe", f"{clean_name}.exe", f"{clean_name}*"]
        try:
            if sys.platform == "win32":
                for tgt in targets:
                    subprocess.run(f"taskkill /IM {tgt} /F", shell=True, capture_output=True)
                return True, f"Closed {display_name}"
            return True, f"Simulated close of {display_name}"
        except Exception as e:
            return False, f"Failed to close {display_name}: {str(e)}"

    def open_file(self, path: str) -> Tuple[bool, str]:
        try:
            clean_path = path.strip().strip('"').strip("'")
            if os.path.exists(clean_path):
                if sys.platform == "win32" and hasattr(os, "startfile"):
                    os.startfile(clean_path)
                else:
                    subprocess.Popen(f'start "" "{clean_path}"', shell=True)
                return True, f"Opened '{clean_path}'"
            return False, f"File not found: '{clean_path}'"
        except Exception as e:
            return False, f"Failed to open '{path}': {str(e)}"

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
