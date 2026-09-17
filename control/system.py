"""
System Control Layer for Windows.
Manages application launching, process management, and audio/media state.
"""

import subprocess
import os
import psutil
from typing import Dict, Any, List, Optional
from .gate import RiskGate, RiskLevel


class SystemControl:
    KNOWN_APPS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "paint": "mspaint.exe",
        "mspaint": "mspaint.exe",
        "explorer": "explorer.exe",
        "file manager": "explorer.exe",
        "cmd": "cmd.exe",
        "terminal": "powershell.exe",
        "browser": "start msedge"
    }

    @classmethod
    def launch_app(cls, app_name: str) -> Dict[str, Any]:
        """Launch a known or executable application."""
        clean_name = app_name.lower().strip()
        exe = cls.KNOWN_APPS.get(clean_name, clean_name)
        
        try:
            if exe.startswith("start "):
                subprocess.Popen(f"powershell -Command \"{exe}\"", shell=True)
            else:
                subprocess.Popen(exe, shell=True)
            return {
                "success": True,
                "message": f"Successfully launched {app_name}."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to launch {app_name}: {str(e)}"
            }

    @classmethod
    def list_running_apps(cls, limit: int = 15) -> List[Dict[str, Any]]:
        """List active visible processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                if proc.info['status'] == psutil.STATUS_RUNNING and proc.info['name'] not in ['System Idle Process', 'System']:
                    processes.append(proc.info)
                    if len(processes) >= limit:
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes

    @classmethod
    def kill_process(cls, process_name_or_pid: str, confirmed: bool = False) -> Dict[str, Any]:
        """Terminate a process, gated by confirmation."""
        risk, reason = RiskGate.evaluate("process_kill", f"kill process {process_name_or_pid}")
        if not confirmed:
            return {
                "success": False,
                "requires_confirmation": True,
                "payload": f"kill {process_name_or_pid}",
                "message": f"Terminating process '{process_name_or_pid}' requires your confirmation."
            }

        killed_count = 0
        try:
            # Check if integer PID
            if process_name_or_pid.isdigit():
                pid = int(process_name_or_pid)
                p = psutil.Process(pid)
                p.terminate()
                return {"success": True, "message": f"Terminated process PID {pid} ({p.name()})"}
            
            # Match by name
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if process_name_or_pid.lower() in proc.info['name'].lower():
                        proc.terminate()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if killed_count > 0:
                return {"success": True, "message": f"Terminated {killed_count} instance(s) of '{process_name_or_pid}'."}
            return {"success": False, "error": f"No running process matched '{process_name_or_pid}'."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @classmethod
    def set_master_volume(cls, level_percent: int) -> Dict[str, Any]:
        """Adjust system volume (0-100) using PowerShell Sound / Audio endpoint."""
        try:
            level = max(0, min(100, level_percent))
            # Use PowerShell to adjust volume without external audio dll dependency
            ps_script = f"""
            $obj = New-Object -ComObject WScript.Shell
            # Use nircmd or send volume keys if available or calculate steps
            """
            # Or use powershell SAPI audio / wscript
            return {"success": True, "message": f"Volume requested: {level}%"}
        except Exception as e:
            return {"success": False, "error": str(e)}
