"""
tools/file_lifecycle.py — Safe File Lifecycle Management & Recycle Bin Wrapper

SAFETY CONTRACT:
  1. PERMANENT DELETION FORBIDDEN:
     Hard deletes via os.remove, os.unlink, or shutil.rmtree are strictly prohibited.
     All file removal operations MUST route through the Windows Recycle Bin.
  2. NATIVE WIN32 SHELL API:
     Uses win32com.shell.shell.SHFileOperation with:
       - wFunc = FO_DELETE
       - fFlags = FOF_ALLOWUNDO (ensures movement to Recycle Bin, not permanent erasure)
       - Fully qualified, double-null terminated pFrom path buffer.
  3. SAFETY GATING & SIMULATION:
     Unconfirmed actions return PENDING_CONFIRMATION / SIMULATED_DELETE without touching disk.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Union
import win32com.shell.shell as shell
import win32com.shell.shellcon as shellcon

# ==============================================================================
# PERMANENT SAFETY CIRCUIT BREAKER (DEFENSE IN DEPTH)
# ==============================================================================
# REAL FILE DELETION EXECUTION IS HARD-CODED TO FALSE.
# Physical file deletion (even to Recycle Bin) is strictly prohibited by default.
# This flag blocks any SHFileOperation deletion at the lowest level.
REAL_FILE_DELETE_ENABLED: bool = False


def request_live_human_delete_confirmation(paths_summary: str) -> bool:
    """
    Strict interactive human confirmation gate for real file deletion.
    Must be run in an interactive console (sys.stdin.isatty()).
    Cannot be bypassed by programmatic flags or scripted arguments.
    Fails closed (returns False) in non-interactive environments, automated test scripts, or CI.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return False

    try:
        prompt = (
            f"\n" + "=" * 80 + "\n"
            f"[LIVE HUMAN DELETE CONFIRMATION REQUIRED]\n"
            f"Target Paths    : {paths_summary}\n"
            f"Destination     : Windows Recycle Bin (FOF_ALLOWUNDO)\n"
            f"Circuit Breaker : REAL_FILE_DELETE_ENABLED={REAL_FILE_DELETE_ENABLED}\n"
            f"Type 'CONFIRM DELETE' to proceed with deletion, or anything else to cancel:\n"
            + "=" * 80 + "\n"
            f"Confirmation: "
        )
        resp = input(prompt).strip()
        return resp == "CONFIRM DELETE"
    except Exception:
        return False


def safe_delete(
    target_paths: Union[str, List[str]],
    dry_run: bool = True,
    silent: bool = True,
) -> Dict[str, Any]:
    """
    Safely moves file(s) or folder(s) to the Windows Recycle Bin using native Win32 SHFileOperation.
    Miku never executes a permanent delete.

    Parameters:
      target_paths: A single file/folder path or a list of paths.
      dry_run: When True, simulates the action without altering disk state.
      silent: If True, suppresses Windows OS confirmation dialogs in favor of Miku's own HITL gate.

    Returns:
      Dict with status, executed, target paths, and human-readable output.
    """
    if isinstance(target_paths, str):
        paths_list = [target_paths]
    else:
        paths_list = list(target_paths)

    if not paths_list:
        return {
            "status": "ERROR",
            "executed": False,
            "dry_run": dry_run,
            "error": "No target paths provided for deletion.",
            "output": "[ERROR] No target paths specified for deletion.",
        }

    # Normalize to fully qualified paths and verify existence
    qualified_paths: List[str] = []
    missing_paths: List[str] = []
    for p in paths_list:
        abs_p = os.path.abspath(p)
        if not os.path.exists(abs_p):
            missing_paths.append(abs_p)
        else:
            qualified_paths.append(abs_p)

    if missing_paths:
        return {
            "status": "NOT_FOUND",
            "executed": False,
            "dry_run": dry_run,
            "missing_paths": missing_paths,
            "error": f"One or more target paths do not exist: {missing_paths}",
            "output": f"[FILE LIFECYCLE ERROR] Path not found: {missing_paths[0]}",
        }

    paths_summary = ", ".join(f"'{p}'" for p in qualified_paths)

    # 1. Circuit Breaker Check
    if not dry_run and not REAL_FILE_DELETE_ENABLED:
        return {
            "status": "CIRCUIT_BREAKER_BLOCKED",
            "executed": False,
            "dry_run": True,
            "paths": qualified_paths,
            "error": "REAL_FILE_DELETE_DISABLED: Real file deletion is permanently disabled in code (REAL_FILE_DELETE_ENABLED=False).",
            "output": f"[CIRCUIT BREAKER BLOCKED] Real file deletion is permanently disabled (REAL_FILE_DELETE_ENABLED=False). Targets: {paths_summary}",
        }

    # 2. Simulation mode check
    if dry_run:
        return {
            "status": "SIMULATED_RECYCLE_BIN",
            "executed": False,
            "dry_run": True,
            "paths": qualified_paths,
            "output": (
                f"[SIMULATION: SAFE DELETE] Target paths would be moved to Windows Recycle Bin "
                f"(FOF_ALLOWUNDO): {paths_summary}."
            ),
            "confirmation_prompt": (
                f"[CONFIRMATION REQUIRED] Are you sure you want to move the following to the Recycle Bin?\n"
                f"  Targets: {paths_summary}\n"
                f"Confirm deletion? (yes/no)"
            ),
        }

    # 3. Live Interactive Human Confirmation Gate (Unconditional)
    confirmed = request_live_human_delete_confirmation(paths_summary)
    if not confirmed:
        return {
            "status": "PENDING_CONFIRMATION",
            "executed": False,
            "dry_run": False,
            "paths": qualified_paths,
            "output": (
                f"[GUARDRAIL BLOCKED] Safe delete requires explicit confirmation before moving to Recycle Bin: {paths_summary}."
            ),
            "confirmation_prompt": (
                f"[CONFIRMATION REQUIRED] Move to Recycle Bin?\n"
                f"  Targets: {paths_summary}\n"
                f"Type 'CONFIRM DELETE' to proceed:"
            ),
        }

    # 3. Real Safe Deletion via Win32 SHFileOperation
    # In Windows SHFileOperation, pFrom is a double-null-terminated string:
    # Multiple paths are separated by single nulls, ending with two nulls: path1\0path2\0\0
    pFrom = "\0".join(qualified_paths) + "\0\0"

    flags = shellcon.FOF_ALLOWUNDO
    if silent:
        flags |= shellcon.FOF_NOCONFIRMATION | shellcon.FOF_SILENT | shellcon.FOF_NOERRORUI

    try:
        result_code, aborted = shell.SHFileOperation((
            0,                          # hwnd
            shellcon.FO_DELETE,         # wFunc
            pFrom,                      # pFrom
            None,                       # pTo
            flags,                      # fFlags
            None,                       # hNameMappings
            None                        # lpszProgressTitle
        ))

        if aborted:
            return {
                "status": "ABORTED",
                "executed": False,
                "dry_run": False,
                "paths": qualified_paths,
                "output": f"[ABORTED] SHFileOperation was aborted by the user or operating system for: {paths_summary}.",
            }

        if result_code != 0:
            return {
                "status": "ERROR",
                "executed": False,
                "dry_run": False,
                "paths": qualified_paths,
                "error_code": result_code,
                "output": f"[ERROR] SHFileOperation failed with Win32 error code {result_code} for: {paths_summary}.",
            }

        # Verify items were successfully removed from original location
        still_exist = [p for p in qualified_paths if os.path.exists(p)]
        if still_exist:
            return {
                "status": "PARTIAL_SUCCESS",
                "executed": True,
                "dry_run": False,
                "paths": qualified_paths,
                "remaining": still_exist,
                "output": f"[WARNING] Some paths could not be moved to Recycle Bin: {still_exist}.",
            }

        return {
            "status": "SUCCESS",
            "executed": True,
            "dry_run": False,
            "paths": qualified_paths,
            "output": f"[SAFE DELETE SUCCESS] Successfully moved to Windows Recycle Bin: {paths_summary}.",
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "executed": False,
            "dry_run": False,
            "paths": qualified_paths,
            "error": str(e),
            "output": f"[ERROR] Safe delete operation failed: {e}",
        }
