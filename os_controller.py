"""
os_controller.py — Direct OS Controller for Windows Kernel & Process Management

Provides direct hardware and OS execution calls bypassing UI accessibility trees.
Uses Python's native ctypes and subprocess to directly command the Windows kernel and process manager.
"""

import ctypes
import os
import subprocess
import time
from typing import Any, Dict


class OSController:
    """Direct hardware and OS execution layer for Windows."""

    def __init__(self):
        # Access Windows User32 API for hardware manipulation
        self.user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
        self.kernel32 = ctypes.windll.kernel32 if hasattr(ctypes, "windll") else None

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

    def list_processes(self, filter_name: str = "") -> Dict[str, Any]:
        """Queries running Windows background processes using tasklist."""
        filter_name = filter_name.strip().lower()
        try:
            cmd = ["tasklist", "/FO", "CSV", "/NH"]
            if filter_name:
                cmd = ["tasklist", "/FI", f"IMAGENAME eq {filter_name}*", "/FO", "CSV", "/NH"]

            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

            processes = []
            for line in lines[:20]:  # Cap at top 20 processes
                parts = [p.strip('"') for p in line.split('","')]
                if len(parts) >= 5:
                    processes.append(f"{parts[0]} (PID: {parts[1]}, Mem: {parts[4]})")

            summary = "\n".join(processes) if processes else "No matching processes found."
            return {"status": "success", "msg": f"[INFO] Running Processes:\n{summary}", "processes": processes}
        except Exception as e:
            return {"status": "error", "msg": f"[ERROR] Failed to list processes: {e}"}

    def get_system_info(self) -> Dict[str, Any]:
        """Retrieves system hardware diagnostics using native Win32 APIs."""
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            info_str = []
            if self.kernel32:
                mem = MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if self.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                    total_gb = mem.ullTotalPhys / (1024**3)
                    avail_gb = mem.ullAvailPhys / (1024**3)
                    info_str.append(f"RAM: {avail_gb:.1f} GB available / {total_gb:.1f} GB total ({mem.dwMemoryLoad}% used)")

            if self.user32 and hasattr(self.user32, "GetForegroundWindow"):
                fg_hwnd = self.user32.GetForegroundWindow()
                buf = ctypes.create_unicode_buffer(256)
                self.user32.GetWindowTextW(fg_hwnd, buf, 256)
                title = buf.value or "Desktop"
                info_str.append(f"Active Window HWND: {fg_hwnd} | Title: '{title}'")

            info_str.append(f"CPU Threads: {os.cpu_count()}")

            msg = "[INFO] OS Diagnostics:\n  " + "\n  ".join(info_str)
            return {"status": "success", "msg": msg}
        except Exception as e:
            return {"status": "error", "msg": f"[ERROR] Failed to get system info: {e}"}

    def take_screenshot(self, filename: str = "screenshot.png") -> Dict[str, Any]:
        """Captures primary desktop screenshot via native GDI / PIL."""
        filename = filename.strip() or "screenshot.png"
        if not filename.endswith(".png") and not filename.endswith(".jpg"):
            filename += ".png"
        try:
            from tools.local_vision import capture_desktop_pil
            img = capture_desktop_pil()
            if img is not None:
                img.save(filename)
                return {"status": "success", "msg": f"[INFO] Screenshot saved to '{os.path.abspath(filename)}'."}
            else:
                return {"status": "error", "msg": "[ERROR] Desktop GDI capture returned null."}
        except Exception as e:
            return {"status": "error", "msg": f"[ERROR] Screenshot capture failed: {e}"}

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
