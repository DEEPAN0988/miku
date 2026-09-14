"""
tools/system_dispatcher.py — System Dispatcher, Media Routing Facade & Elevated Task Launcher

Re-exports core dispatcher symbols, context-aware media router, and provides an
elevated application launcher bridge via the Windows Task Scheduler.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.dispatcher import (
    CONFIDENCE_MARGIN,
    CONFIDENCE_RATIO,
    RiskLevel,
    TOOL_DESCRIPTORS,
    TOOL_REGISTRY,
    ToolCall,
    ToolResult,
    detect_query_argument,
    dispatch_tool,
    extract_deterministic_slot,
    parse_tool_call,
    rank_with_structure,
    resolve_intent_anchor,
    resolve_intent_with_confidence,
    route_media_play,
)


def _sanitize_app_key(name: str) -> str:
    """
    Sanitizes an application name into a safe, alphanumeric identifier for Task Scheduler names.
    Replaces whitespace, hyphens, and invalid chars with underscores.
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "app"
    return clean[:48]


def _validate_path_security(path_str: str) -> Tuple[bool, str]:
    """
    Validates a file or directory path against injection risks (null bytes, control chars, shell operators).
    """
    if not path_str or not isinstance(path_str, str):
        return False, "Path must be a non-empty string."

    clean = path_str.strip().strip('"').strip("'")
    if not clean:
        return False, "Path cannot be empty."

    # Forbidden injection characters
    forbidden_chars = ['\x00', '\n', '\r', ';', '&', '|', '`', '$', '<', '>', '^']
    for ch in forbidden_chars:
        if ch in clean:
            return False, f"Path contains illegal character ({repr(ch)})."

    return True, clean


def check_elevated_task_exists(task_name: str, runner: Optional[Callable] = None) -> bool:
    """
    Checks if a scheduled task exists in Windows Task Scheduler using `schtasks /query`.
    """
    if not task_name:
        return False

    cmd = ["schtasks", "/query", "/tn", task_name]
    try:
        if runner is not None:
            res = runner(cmd)
        else:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        return (res.returncode == 0) and (task_name.lower() in res.stdout.lower())
    except Exception:
        return False


def register_elevated_task(
    task_name: str,
    executable_path: str,
    working_dir: Optional[str] = None,
    shell_executor: Optional[Callable] = None,
    wait_timeout_sec: float = 8.0,
    check_runner: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Auto-registration helper for elevated tasks:
    Builds a PowerShell registration command with `-RunLevel Highest` and `-LogonType Interactive`.
    Invokes via ShellExecuteW with the 'runas' verb to prompt the user once for UAC approval during setup.
    """
    # 1. Path validation
    is_valid, clean_exe = _validate_path_security(executable_path)
    if not is_valid:
        return {
            "status": "VALIDATION_ERROR",
            "registered": False,
            "error": f"Invalid executable path: {clean_exe}",
            "output": f"[SECURITY REJECTION] Executable path rejected: {clean_exe}",
        }

    clean_wd = None
    if working_dir:
        wd_valid, clean_wd = _validate_path_security(working_dir)
        if not wd_valid:
            return {
                "status": "VALIDATION_ERROR",
                "registered": False,
                "error": f"Invalid working directory: {clean_wd}",
                "output": f"[SECURITY REJECTION] Working directory rejected: {clean_wd}",
            }

    # Normalize paths
    abs_exe = os.path.abspath(clean_exe)
    if not os.path.exists(abs_exe):
        return {
            "status": "NOT_FOUND",
            "registered": False,
            "error": f"Target executable does not exist on disk: {abs_exe}",
            "output": f"[REGISTRATION ERROR] File not found: {abs_exe}",
        }

    wd_arg = f"-WorkingDirectory '{os.path.abspath(clean_wd)}'" if clean_wd else ""

    # Build PowerShell command
    ps_cmd = (
        f"$action = New-ScheduledTaskAction -Execute '{abs_exe}' {wd_arg}; "
        f"$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest; "
        f"Register-ScheduledTask -TaskName '{task_name}' -Action $action -Principal $principal -Force"
    )

    try:
        if shell_executor is not None:
            exec_code = shell_executor(
                None,
                "runas",
                "powershell.exe",
                f"-NoProfile -WindowStyle Hidden -Command \"{ps_cmd}\"",
                clean_wd or os.path.dirname(abs_exe),
                0,
            )
        else:
            exec_code = ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                "powershell.exe",
                f"-NoProfile -WindowStyle Hidden -Command \"{ps_cmd}\"",
                clean_wd or os.path.dirname(abs_exe),
                0,
            )

        if exec_code <= 32 and shell_executor is None:
            # Fallback to COM Shell.Application in a daemon thread to prevent blocking
            def _async_com_register():
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    import win32com.client
                    sh = win32com.client.Dispatch("Shell.Application")
                    sh.ShellExecute(
                        "powershell.exe",
                        f"-NoProfile -WindowStyle Hidden -Command \"{ps_cmd}\"",
                        clean_wd or os.path.dirname(abs_exe),
                        "runas",
                        0,
                    )
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

            import threading
            t = threading.Thread(target=_async_com_register, daemon=True)
            t.start()
            t.join(timeout=1.5)
            exec_code = 42  # Handed off to Windows Shell COM server

        if exec_code <= 32:
            return {
                "status": "UAC_DENIED",
                "registered": False,
                "exec_code": exec_code,
                "output": f"[UAC DENIED] User declined or Windows prevented task registration elevation (Code: {exec_code}).",
            }

        # Poll for task appearance
        start_t = time.time()
        while (time.time() - start_t) < wait_timeout_sec:
            if check_elevated_task_exists(task_name, runner=check_runner):
                return {
                    "status": "SUCCESS",
                    "registered": True,
                    "task_name": task_name,
                    "output": f"[ELEVATED REGISTRATION SUCCESS] Task '{task_name}' registered successfully with RunLevel Highest.",
                }
            time.sleep(0.5)

        return {
            "status": "PENDING_OR_TIMEOUT",
            "registered": False,
            "task_name": task_name,
            "output": f"[TIMEOUT] Task registration was dispatched but task '{task_name}' was not detected within {wait_timeout_sec}s.",
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "registered": False,
            "error": str(e),
            "output": f"[REGISTRATION EXCEPTION] Failed to execute elevated registration: {e}",
        }


def launch_elevated_app(
    app_target: str,
    working_dir: Optional[str] = None,
    auto_register: bool = True,
    task_name: Optional[str] = None,
    task_runner: Optional[Callable] = None,
    shell_executor: Optional[Callable] = None,
    register_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Elevated Application Launcher:
    1. Checks if a scheduled task named Miku_Elevated_<AppKey> exists.
    2. If it exists, executes it silently via `schtasks /run /tn <task_name>` (zero UAC prompts).
    3. If missing and auto_register is True, prompts user once for UAC elevation to register the bridge task,
       then executes the task.
    4. If the task bridge is unavailable, falls back to direct ShellExecuteW with 'runas' verb.
    """
    from tools.app_launcher import resolve_app_path

    # 1. Resolve executable path
    resolved = resolve_app_path(app_target)
    if resolved:
        resolved_name, target_path = resolved
    else:
        resolved_name = app_target.strip()
        target_path = app_target.strip()

    # Unwrap .lnk shortcut target if applicable
    if target_path.lower().endswith(".lnk") and os.path.exists(target_path):
        try:
            import win32com.client
            sh = win32com.client.Dispatch("WScript.Shell")
            sc = sh.CreateShortcut(target_path)
            if sc.TargetPath and os.path.exists(sc.TargetPath):
                if not working_dir and sc.WorkingDirectory and os.path.exists(sc.WorkingDirectory):
                    working_dir = sc.WorkingDirectory
                target_path = sc.TargetPath
        except Exception:
            pass

    is_valid, clean_target = _validate_path_security(target_path)
    if not is_valid:
        return {
            "status": "INVALID_TARGET",
            "executed": False,
            "error": clean_target,
            "output": f"[LAUNCH REJECTION] Invalid app target path: {clean_target}",
        }

    app_key = _sanitize_app_key(resolved_name)
    task_name = task_name or f"Miku_Elevated_{app_key}"

    # Determine default working directory
    target_wd = working_dir
    if not target_wd and os.path.isabs(clean_target):
        target_wd = os.path.dirname(clean_target)

    # 2. Check if bridge task exists
    task_exists = check_elevated_task_exists(task_name, runner=task_runner)

    if task_exists:
        # Silently execute via schtasks /run
        run_cmd = ["schtasks", "/run", "/tn", task_name]
        try:
            if task_runner is not None:
                run_res = task_runner(run_cmd)
            else:
                run_res = subprocess.run(run_cmd, capture_output=True, text=True, timeout=5)

            if run_res.returncode == 0:
                return {
                    "status": "SUCCESS",
                    "mode": "TASK_SCHEDULER",
                    "executed": True,
                    "task_name": task_name,
                    "app_name": resolved_name,
                    "target": clean_target,
                    "output": f"[ELEVATED LAUNCH SUCCESS] Silently executed '{resolved_name}' via Task Scheduler bridge '{task_name}'.",
                }
        except Exception:
            pass

    # 3. Task does not exist: attempt registration if auto_register is True
    if auto_register:
        reg_fn = register_fn if register_fn is not None else register_elevated_task
        reg_res = reg_fn(
            task_name=task_name,
            executable_path=clean_target,
            working_dir=target_wd,
            shell_executor=shell_executor,
            check_runner=task_runner,
        )

        if reg_res.get("registered") or reg_res.get("status") == "SUCCESS":
            # Task registered! Run it immediately
            run_cmd = ["schtasks", "/run", "/tn", task_name]
            try:
                if task_runner is not None:
                    run_res = task_runner(run_cmd)
                else:
                    run_res = subprocess.run(run_cmd, capture_output=True, text=True, timeout=5)

                if run_res.returncode == 0:
                    return {
                        "status": "SUCCESS",
                        "mode": "TASK_SCHEDULER_REGISTERED_AND_RUN",
                        "executed": True,
                        "task_name": task_name,
                        "app_name": resolved_name,
                        "target": clean_target,
                        "output": f"[ELEVATED REGISTRATION & LAUNCH SUCCESS] Registered and launched '{resolved_name}' via Task Scheduler bridge.",
                    }
            except Exception:
                pass

    # 4. Fallback Logic: Direct ShellExecuteW with "runas" verb
    try:
        if shell_executor is not None:
            exec_code = shell_executor(
                None,
                "runas",
                clean_target,
                None,
                target_wd,
                1,
            )
        else:
            exec_code = ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                clean_target,
                None,
                target_wd,
                1,
            )
            if exec_code <= 32:
                def _async_com_fallback():
                    try:
                        import pythoncom
                        pythoncom.CoInitialize()
                        import win32com.client
                        sh = win32com.client.Dispatch("Shell.Application")
                        sh.ShellExecute(
                            clean_target,
                            "",
                            target_wd or "",
                            "runas",
                            1,
                        )
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

                import threading
                t = threading.Thread(target=_async_com_fallback, daemon=True)
                t.start()
                t.join(timeout=1.5)
                exec_code = 42

        if exec_code > 32:
            return {
                "status": "FALLBACK_DIRECT_ELEVATION",
                "mode": "DIRECT_RUNAS",
                "executed": True,
                "app_name": resolved_name,
                "target": clean_target,
                "output": (
                    f"[FALLBACK DIRECT ELEVATION] Task bridge unavailable for '{resolved_name}'. "
                    f"Dispatched direct elevated launch (Code: {exec_code}); manual UAC confirmation expected."
                ),
            }
        else:
            return {
                "status": "ELEVATION_FAILED",
                "mode": "DIRECT_RUNAS",
                "executed": False,
                "error_code": exec_code,
                "output": f"[ELEVATION FAILED] ShellExecuteW failed with error code {exec_code} for '{resolved_name}'.",
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "mode": "FAILED",
            "executed": False,
            "error": str(e),
            "output": f"[LAUNCH ERROR] Failed to launch elevated application: {e}",
        }


__all__ = [
    "CONFIDENCE_MARGIN",
    "CONFIDENCE_RATIO",
    "RiskLevel",
    "TOOL_DESCRIPTORS",
    "TOOL_REGISTRY",
    "ToolCall",
    "ToolResult",
    "detect_query_argument",
    "dispatch_tool",
    "extract_deterministic_slot",
    "parse_tool_call",
    "rank_with_structure",
    "resolve_intent_anchor",
    "resolve_intent_with_confidence",
    "route_media_play",
    "_sanitize_app_key",
    "check_elevated_task_exists",
    "register_elevated_task",
    "launch_elevated_app",
]
