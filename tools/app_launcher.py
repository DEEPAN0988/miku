"""
tools/app_launcher.py — Windows Application Discovery & Lifecycle Management (Phase 10)

Provides verified, zero-external-dependency discovery and launching of Windows applications:
1. Scans Start Menu shortcuts (.lnk) in APPDATA and ALLUSERSPROFILE.
2. Scans Registry App Paths (HKLM & HKCU \\ Software \\ Microsoft \\ Windows \\ CurrentVersion \\ App Paths).
3. Resolves common built-in Windows utilities (calc, notepad, cmd, explorer, etc.).
4. Launches verified applications via os.startfile.
5. Focuses running application windows via ctypes.windll.user32.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
import winreg
from typing import Any, Dict, List, Optional, Tuple


BUILTIN_APPS = {
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "notepad": "notepad.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "command prompt": "cmd.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "control panel": "control.exe",
}


def scan_start_menu_shortcuts() -> Dict[str, str]:
    """
    Scans Start Menu folders for installed application shortcuts (.lnk).
    Returns mapping of lowercase application name -> full .lnk path.
    """
    shortcuts: Dict[str, str] = {}
    candidate_dirs = [
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%ALLUSERSPROFILE%\Microsoft\Windows\Start Menu\Programs"),
    ]
    for base_dir in candidate_dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".lnk"):
                    app_name = os.path.splitext(file)[0].strip().lower()
                    shortcuts[app_name] = os.path.join(root, file)
    return shortcuts


def scan_registry_app_paths() -> Dict[str, str]:
    """
    Scans HKLM and HKCU 'App Paths' registry keys for registered application executables.
    Returns mapping of lowercase application name/executable -> executable path.
    """
    app_paths: Dict[str, str] = {}
    for hkey in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hkey, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") as key:
                count = winreg.QueryInfoKey(key)[0]
                for i in range(count):
                    name = winreg.EnumKey(key, i)
                    try:
                        with winreg.OpenKey(key, name) as app_key:
                            val, _ = winreg.QueryValueEx(app_key, "")
                            if val:
                                expanded = os.path.expandvars(val.strip('"'))
                                base_name = os.path.splitext(name)[0].lower()
                                app_paths[name.lower()] = expanded
                                app_paths[base_name] = expanded
                    except Exception:
                        continue
        except Exception:
            continue
    return app_paths


def resolve_app_path(app_name: str) -> Optional[Tuple[str, str]]:
    """
    Resolves an application name to a target executable or shortcut path.
    Returns (resolved_name, path_or_command) or None.
    """
    cleaned = app_name.strip().lower()
    if not cleaned:
        return None

    # 1. Built-in Windows executables
    if cleaned in BUILTIN_APPS:
        return cleaned, BUILTIN_APPS[cleaned]

    # 2. Start Menu shortcuts
    shortcuts = scan_start_menu_shortcuts()
    if cleaned in shortcuts:
        return cleaned, shortcuts[cleaned]

    # Partial / substring match in Start Menu shortcuts
    for name, path in shortcuts.items():
        if cleaned == name or cleaned in name.split():
            return name, path

    # 3. Registry App Paths
    app_paths = scan_registry_app_paths()
    if cleaned in app_paths:
        return cleaned, app_paths[cleaned]
    if f"{cleaned}.exe" in app_paths:
        return cleaned, app_paths[f"{cleaned}.exe"]

    # Partial match in App Paths
    for name, path in app_paths.items():
        if cleaned in name:
            return name, path

    return None


def find_installed_apps(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Searches for installed applications matching query string.
    Returns list of {'name': app_name, 'path': path}.
    """
    q = query.strip().lower()
    results: List[Dict[str, str]] = []
    seen = set()

    # Search built-ins
    for name, exe in BUILTIN_APPS.items():
        if q in name or name in q:
            if name not in seen:
                seen.add(name)
                results.append({"name": name, "path": exe, "source": "builtin"})

    # Search Start Menu
    shortcuts = scan_start_menu_shortcuts()
    for name, path in sorted(shortcuts.items()):
        if q in name or name in q:
            if name not in seen:
                seen.add(name)
                results.append({"name": name, "path": path, "source": "start_menu"})
                if len(results) >= limit:
                    return results

    # Search App Paths
    app_paths = scan_registry_app_paths()
    for name, path in sorted(app_paths.items()):
        if q in name or name in q:
            if name not in seen:
                seen.add(name)
                results.append({"name": name, "path": path, "source": "registry_app_paths"})
                if len(results) >= limit:
                    return results

    return results


def launch_app(app_name: str) -> Dict[str, Any]:
    """
    Launches an application by name via os.startfile.
    Returns status dictionary with execution details.
    """
    res = resolve_app_path(app_name)
    if not res:
        # Fallback attempt: try directly launching with .exe
        target = f"{app_name.strip()}.exe"
        resolved_name = app_name.strip()
    else:
        resolved_name, target = res

    try:
        os.startfile(target)
        return {
            "status": "SUCCESS",
            "app_name": resolved_name,
            "target": target,
            "output": f"Launched application '{resolved_name}' via target '{target}'.",
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "app_name": app_name,
            "target": target,
            "error": str(e),
            "output": f"Failed to launch '{app_name}': {e}",
        }


def restart_app(app_name: str) -> Dict[str, Any]:
    """
    Terminates matching process and relaunches application.
    """
    clean_name = app_name.strip().lower()
    proc_target = f"{clean_name}.exe"

    # Terminate existing instances
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", proc_target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        pass

    time.sleep(0.5)
    return launch_app(app_name)


def focus_app(app_name: str) -> Dict[str, Any]:
    """
    Brings an existing application window matching app_name to the foreground.
    """
    target = app_name.strip().lower()
    found_hwnds: List[Tuple[int, str]] = []

    user32 = ctypes.windll.user32
    hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
    if hdesk:
        user32.SetThreadDesktop(hdesk)

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

    def enum_cb(hwnd, lparam):
        if user32.IsWindow(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value
                if target in title.lower() and "gdi+" not in title.lower():
                    found_hwnds.append((hwnd, title))
        return True

    cb = WNDENUMPROC(enum_cb)
    user32.EnumWindows(cb, 0)

    if not found_hwnds:
        return {
            "status": "NOT_FOUND",
            "app_name": app_name,
            "output": f"No active window found matching '{app_name}'.",
        }

    hwnd, title = found_hwnds[0]
    # SW_RESTORE = 9
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    return {
        "status": "SUCCESS",
        "app_name": app_name,
        "window_title": title,
        "output": f"Brought window '{title}' to foreground.",
    }
