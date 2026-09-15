"""
os_controller.py — Direct OS Controller for Windows Kernel & Process Management

Provides direct hardware and OS execution calls bypassing UI accessibility trees.
Uses Python's native ctypes and subprocess to directly command the Windows kernel and process manager.
"""

import ctypes
import subprocess
from typing import Any, Dict


class OSController:
    """Direct hardware and OS execution layer for Windows."""

    def __init__(self):
        # Access Windows User32 API for hardware manipulation
        self.user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None

    def kill_process(self, process_name: str) -> Dict[str, Any]:
        """Force kills an unresponsive background process."""
        process_name = process_name.strip()
        if not process_name:
            return {"status": "error", "msg": "[ERROR] No process name specified for termination."}
        if process_name.endswith(".exe"):
            process_name = process_name[:-4]
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", f"{process_name}.exe"],
                capture_output=True,
                text=True,
                check=True,
            )
            return {"status": "success", "msg": f"[INFO] Terminated process: {process_name}.exe"}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "msg": f"[ERROR] Failed to terminate {process_name}: {e.stderr.strip() if e.stderr else str(e)}"}
        except Exception as e:
            return {"status": "error", "msg": f"[ERROR] Failed to terminate {process_name}: {e}"}

    def lock_workstation(self) -> Dict[str, Any]:
        """Instantly locks the Windows machine for security."""
        try:
            if self.user32 and hasattr(self.user32, "LockWorkStation"):
                self.user32.LockWorkStation()
                return {"status": "success", "msg": "[INFO] Workstation secured."}
            else:
                return {"status": "error", "msg": "[ERROR] LockWorkStation unavailable on non-Windows environment."}
        except Exception as e:
            return {"status": "error", "msg": f"[ERROR] Failed to lock workstation: {e}"}

    def boot_workspace(self) -> Dict[str, Any]:
        """Hardcoded macro to initialize the system-life-os MERN/Vite stack."""
        try:
            # Launch IDE and local server non-blocking
            subprocess.Popen("code .", shell=True)
            subprocess.Popen("start http://localhost:5173", shell=True)
            return {"status": "success", "msg": "[INFO] Workspace environment initialized successfully."}
        except Exception as e:
            return {"status": "error", "msg": f"[ERROR] Workspace boot failed: {e}"}
