"""
Safe File Lifecycle Management for Miku OS Assistant.

Provides safe file deletion by dispatching files to the Windows Recycle Bin
via Win32 SHFileOperation (FO_DELETE + FOF_ALLOWUNDO). Permanent deletion
(os.remove / shutil.rmtree) is strictly forbidden in this module.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import win32com.shell.shell as shell
import win32com.shell.shellcon as shellcon

from tools.screen_inspector import check_autonomous_authorization


def request_live_human_delete_confirmation(target_path: str) -> bool:
    """
    Strict authorization gate for live file deletion / recycling.
    If MIKU_AUTONOMOUS_MODE=True and MIKU_LIVE_EXECUTION=True, permits execution non-blockingly.
    Otherwise requires an interactive console (sys.stdin.isatty()) and typing 'CONFIRM DELETE'.
    """
    if check_autonomous_authorization("DELETE"):
        return True

    if not sys.stdin or not sys.stdin.isatty():
        return False

    print("\n" + "=" * 64, flush=True)
    print(" [!] CAUTION: LIVE FILE DELETION REQUESTED", flush=True)
    print(f" Target: {target_path}", flush=True)
    print(" Action: Move to Windows Recycle Bin (FOF_ALLOWUNDO)", flush=True)
    print("=" * 64, flush=True)

    try:
        token = input("Type 'CONFIRM DELETE' to proceed with recycling: ").strip()
        return token == "CONFIRM DELETE"
    except (EOFError, KeyboardInterrupt):
        return False


def format_double_null_terminated_path(path: str) -> str:
    """
    Formats a file path for Win32 SHFileOperation:
    Fully qualified (absolute) and double-null terminated.
    """
    abs_path = os.path.abspath(path)
    # Double-null termination for Win32 MULTI_SZ / SHFileOperation buffer
    return f"{abs_path}\0\0"


def safe_recycle_file(
    path: str,
    force_dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Safely moves a target file or directory to the Windows Recycle Bin using
    SHFileOperation with FO_DELETE and FOF_ALLOWUNDO.

    Parameters:
      path: Path to the target file or directory.
      force_dry_run: If True, forces simulated dry-run regardless of env vars.

    Returns:
      Dict containing 'success', 'status', 'path', and diagnostic details.
    """
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return {
            "success": False,
            "status": "FILE_NOT_FOUND",
            "path": abs_path,
            "message": f"Target path does not exist: '{abs_path}'",
        }

    # Determine execution mode:
    # If force_dry_run is explicitly requested, or live execution is disabled, simulate.
    live_exec = os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes")

    if force_dry_run is True or (force_dry_run is None and not live_exec):
        print(f"[*] [SIMULATED RECYCLE] Would move '{abs_path}' to Recycle Bin (FOF_ALLOWUNDO).", flush=True)
        return {
            "success": True,
            "status": "SIMULATED_RECYCLE",
            "path": abs_path,
            "message": f"[SIMULATED DELETE] Target '{abs_path}' validated and safe for recycling.",
        }

    # Check authorization gate before physical OS action
    authorized = request_live_human_delete_confirmation(abs_path)
    if not authorized:
        print(f"[!] [SAFETY GATE REJECTION] File deletion aborted: Not authorized for '{abs_path}'.", flush=True)
        return {
            "success": False,
            "status": "ABORT_UNAUTHORIZED",
            "path": abs_path,
            "message": f"File recycling unconfirmed or rejected for '{abs_path}'.",
        }

    # Format double-null terminated path
    p_from = format_double_null_terminated_path(abs_path)

    # Configure SHFileOperation flags: strictly enforce FOF_ALLOWUNDO
    flags = shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION | shellcon.FOF_SILENT

    try:
        ret_code, aborted = shell.SHFileOperation(
            (
                0,                          # hwnd (0 = desktop / no parent)
                shellcon.FO_DELETE,         # wFunc (FO_DELETE)
                p_from,                     # pFrom (double-null terminated)
                None,                       # pTo
                flags,                      # fFlags (FOF_ALLOWUNDO)
                None,                       # hNameMappings
                None,                       # lpszProgressTitle
            )
        )

        if ret_code == 0 and not aborted:
            print(f"[+] [FILE RECYCLED] Safely moved '{abs_path}' to Windows Recycle Bin.", flush=True)
            return {
                "success": True,
                "status": "RECYCLED_SUCCESS",
                "path": abs_path,
                "return_code": ret_code,
                "aborted": aborted,
                "message": f"File '{abs_path}' successfully moved to Recycle Bin.",
            }
        else:
            print(f"[!] [RECYCLE FAILED] SHFileOperation code {ret_code}, aborted={aborted}.", flush=True)
            return {
                "success": False,
                "status": "RECYCLE_ERROR",
                "path": abs_path,
                "return_code": ret_code,
                "aborted": aborted,
                "message": f"SHFileOperation failed with return code {ret_code}.",
            }

    except Exception as exc:
        print(f"[!] [RECYCLE EXCEPTION] Exception during SHFileOperation: {exc}", flush=True)
        return {
            "success": False,
            "status": "EXCEPTION",
            "path": abs_path,
            "error": str(exc),
            "message": f"Recycle operation encountered exception: {exc}",
        }
