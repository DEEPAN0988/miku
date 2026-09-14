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

    # Partial / substring match in Start Menu shortcuts (deprioritize uninstaller shortcuts)
    candidates = []
    for name, path in shortcuts.items():
        if "uninstall" in name or "remove" in name:
            continue
        if cleaned == name or cleaned in name or name in cleaned or cleaned in name.split():
            candidates.append((name, path))

    if candidates:
        candidates.sort(key=lambda x: abs(len(x[0]) - len(cleaned)))
        return candidates[0]

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


def find_direct_game_binary(launcher_path: str) -> Optional[str]:
    """
    If a launcher executable is provided, checks if a direct game binary exists
    in a sibling or sub-directory (e.g., 'Wuthering Waves Game\\Wuthering Waves.exe').
    """
    if not launcher_path or not os.path.exists(launcher_path):
        return None
    base_dir = os.path.dirname(launcher_path)
    base_name = os.path.basename(base_dir).lower()
    excluded = ("uninst", "clearthirdparty", "crashreport", "unitycrashhandler", "update", "patch")
    candidates = []
    try:
        for entry in os.listdir(base_dir):
            sub_path = os.path.join(base_dir, entry)
            if os.path.isdir(sub_path) and "game" in entry.lower():
                for f in os.listdir(sub_path):
                    if f.lower().endswith(".exe") and not any(x in f.lower() for x in excluded):
                        cand = os.path.join(sub_path, f)
                        if os.path.isfile(cand):
                            candidates.append(cand)
        if candidates:
            # Prioritize candidate that shares words with the base game folder name
            candidates.sort(
                key=lambda c: len(set(os.path.splitext(os.path.basename(c))[0].lower().split()) & set(base_name.split())),
                reverse=True,
            )
            return candidates[0]
    except Exception:
        pass
    return None


def _verify_process_alive(
    target: str,
    min_duration: float = 0.5,
    expected_image: Optional[str] = None,
    timeout: Optional[float] = None,
) -> bool:
    """
    Verifies if a process matching target image name is actively running in tasklist
    and remains alive after a stabilization duration.
    """
    wait_time = timeout if timeout is not None else min_duration
    if wait_time > 0:
        time.sleep(wait_time)

    base_name = os.path.basename(target).lower()
    images_to_check = {base_name}
    if expected_image:
        images_to_check.add(os.path.basename(expected_image).lower())

    ALIAS_MAP = {
        "calc.exe": "calculatorapp.exe",
        "calc": "calculatorapp.exe",
        "calculator.exe": "calculatorapp.exe",
        "calculator": "calculatorapp.exe",
        "mspaint.exe": "mspaint.exe",
        "control.exe": "explorer.exe",
    }
    for alias_k, alias_v in ALIAS_MAP.items():
        if alias_k in images_to_check:
            images_to_check.add(alias_v)

    if base_name.endswith(".msc"):
        images_to_check.add("mmc.exe")

    try:
        res = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = [p.strip('"') for p in line.split('","')]
                if parts:
                    proc_name = parts[0].lower()
                    for img in images_to_check:
                        if img and (proc_name == img or proc_name == f"{img}.exe"):
                            return True
    except Exception:
        pass
    return False


def launch_app(app_name: str, max_retries: int = 3, retry_delay: float = 0.5) -> Dict[str, Any]:
    """
    Launches an application by name via a resilient multi-tier fallback loop:
      Tier 1: Direct shell execution via os.startfile (verified with tasklist).
      Tier 2: Windows Application Compatibility Layer (RunAsInvoker shim).
      Tier 3: Windows Task Scheduler Bridge (zero-prompt execution).
      Tier 4: Auto-registration & elevated fallback.
    """
    res = resolve_app_path(app_name)
    if not res:
        target = f"{app_name.strip()}.exe"
        resolved_name = app_name.strip()
    else:
        resolved_name, target = res

    original_target = target
    working_dir: Optional[str] = None
    shortcut_args: Optional[str] = None
    target_exe: str = target

    # Unwrap .lnk shortcut target if applicable
    if target.lower().endswith(".lnk") and os.path.exists(target):
        try:
            import win32com.client
            sh = win32com.client.Dispatch("WScript.Shell")
            sc = sh.CreateShortcut(target)
            if sc.TargetPath:
                target_exe = sc.TargetPath
                shortcut_args = sc.Arguments
                if sc.WorkingDirectory and os.path.exists(sc.WorkingDirectory):
                    working_dir = sc.WorkingDirectory
                elif os.path.isabs(target_exe):
                    working_dir = os.path.dirname(target_exe)
        except Exception:
            pass

    if not working_dir and os.path.isabs(target_exe):
        working_dir = os.path.dirname(target_exe)

    # Check for direct game binary if target is a launcher
    direct_game_binary = find_direct_game_binary(target_exe)

    # Tier 1 candidates: If original target is a .lnk, prioritize launching the .lnk
    # because Windows Shell natively resolves all shortcut parameters and working directories.
    tier1_candidates = []
    if original_target.lower().endswith(".lnk") and os.path.exists(original_target):
        tier1_candidates.append(original_target)
    if target_exe and target_exe not in tier1_candidates:
        tier1_candidates.append(target_exe)
    if direct_game_binary and direct_game_binary not in tier1_candidates:
        tier1_candidates.append(direct_game_binary)

    # Execution candidates for direct process invocation (Tier 2)
    tier2_candidates = []
    if direct_game_binary:
        tier2_candidates.append(direct_game_binary)
    if target_exe and target_exe.lower().endswith(".exe"):
        tier2_candidates.append(target_exe)
    if not tier2_candidates and tier1_candidates:
        tier2_candidates.extend(tier1_candidates)

    expected_img = os.path.basename(target_exe).lower() if target_exe else None

    # Tier 1: Direct execution via os.startfile with process verification
    for cand in tier1_candidates:
        try:
            os.startfile(cand)
            if _verify_process_alive(cand, min_duration=0.8, expected_image=expected_img):
                return {
                    "status": "SUCCESS",
                    "mode": "DIRECT_STARTFILE",
                    "app_name": resolved_name,
                    "target": cand,
                    "output": f"Launched application '{resolved_name}' via target '{cand}'.",
                }
        except OSError as e:
            if getattr(e, "winerror", None) == 740:
                break
        except Exception:
            pass

    # Tier 2: Application Compatibility Layer (RunAsInvoker shim)
    import shlex
    for cand in tier2_candidates:
        if not cand.lower().endswith(".exe") and not os.path.isfile(cand):
            continue
        cand_wd = os.path.dirname(cand) if os.path.isabs(cand) else working_dir
        cmd = [cand]
        if shortcut_args and cand == target_exe:
            try:
                cmd.extend(shlex.split(shortcut_args))
            except Exception:
                pass
        for _ in range(max_retries):
            try:
                env = os.environ.copy()
                env["__COMPAT_LAYER"] = "RunAsInvoker"
                p = subprocess.Popen(cmd, cwd=cand_wd, env=env)
                if _verify_process_alive(cand, min_duration=0.8, expected_image=expected_img) and p.poll() is None:
                    return {
                        "status": "SUCCESS",
                        "mode": "COMPAT_RUNASINVOKER",
                        "app_name": resolved_name,
                        "target": cand,
                        "pid": p.pid,
                        "output": f"Launched application '{resolved_name}' via RunAsInvoker compatibility shim (PID {p.pid}).",
                    }
            except Exception:
                pass

    # Tier 3: Task Scheduler Bridge via launch_elevated_app
    try:
        from tools.system_dispatcher import launch_elevated_app
        return launch_elevated_app(app_name, working_dir=working_dir)
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

    import win32gui

    def _cb(hwnd, _):
        if win32gui.IsWindow(hwnd):
            try:
                title = win32gui.GetWindowText(hwnd)
                cls_name = win32gui.GetClassName(hwnd)
                if (target in title.lower() or target in cls_name.lower()) and "gdi+" not in title.lower():
                    found_hwnds.append((hwnd, title))
            except Exception:
                pass
        return True

    try:
        win32gui.EnumWindows(_cb, None)
    except Exception:
        pass

    if not found_hwnds:
        return {
            "status": "NOT_FOUND",
            "app_name": app_name,
            "found": False,
            "hwnd": 0,
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
        "found": True,
        "hwnd": hwnd,
        "output": f"Brought window '{title}' to foreground.",
    }
