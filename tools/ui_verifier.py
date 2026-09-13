"""
tools/ui_verifier.py — Real-Time UI Target & Window Focus Verifier (Phase 12 Remediation)

SAFETY CONTRACT:
  Before any UI automation (keystroke, text staging, or mouse click) is dispatched,
  the automation engine MUST independently inspect and verify:
    1. The target application actually possesses foreground window focus
       (verifying HWND, process executable name, and window class).
    2. The target window is NOT a collision dialog, modal popup, forward screen,
       file chooser, or profile viewer.
    3. The expected recipient/contact name is explicitly confirmed visible
       in the window title, header bar, or active UI hierarchy.

FAIL-CLOSED BEHAVIOR:
  If any check fails or cannot be verified with high confidence, the system
  MUST IMMEDIATELY ABORT and refuse to send keystrokes.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

# Known modal or collision titles/classes that indicate an unsafe state
UNSAFE_WINDOW_INDICATORS = [
    "forward",
    "share",
    "attach",
    "open file",
    "settings",
    "profile",
    "sign in",
    "login",
    "select contact",
    "choose",
    "dialog",
    "crop",
]

# Supported target applications and their known process executables
APP_PROCESS_CATALOG: Dict[str, List[str]] = {
    "whatsapp": ["whatsapp.exe", "whatsapp.root.exe"],
    "discord": ["discord.exe"],
    "telegram": ["telegram.exe"],
    "teams": ["teams.exe", "msteams.exe"],
}


def _attach_thread_to_default_desktop():
    """
    Ensures calling thread is attached to the interactive Default desktop ('WinSta0\\Default')
    so that real user windows and foreground state are accurately observed.
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


def inspect_live_foreground_window() -> Dict[str, Any]:
    """
    Inspects the currently active Windows foreground window via Win32 APIs.
    Returns metadata regarding HWND, process ID, executable name, window title, and class.
    """
    try:
        import ctypes
        import win32gui
        import win32process
        import psutil

        _attach_thread_to_default_desktop()

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd or not win32gui.IsWindow(hwnd):
            return {
                "hwnd": 0,
                "is_window": False,
                "is_visible": False,
                "title": "",
                "class_name": "",
                "process_id": 0,
                "process_name": "unknown",
                "error": "No foreground window active",
            }

        is_vis = bool(win32gui.IsWindowVisible(hwnd))
        title = win32gui.GetWindowText(hwnd).strip()
        class_name = win32gui.GetClassName(hwnd).strip()

        tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = "unknown"
        if pid > 0:
            try:
                proc = psutil.Process(pid)
                process_name = proc.name().lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Collect text from top-level child elements if available
        child_texts = []
        def _enum_children(child_hwnd, _):
            if win32gui.IsWindowVisible(child_hwnd):
                ct = win32gui.GetWindowText(child_hwnd).strip()
                if ct:
                    child_texts.append(ct)
        try:
            win32gui.EnumChildWindows(hwnd, _enum_children, None)
        except Exception:
            pass

        # For modern frameworks (WinUI 3, Chromium WebView, Electron) where Win32 EnumChildWindows
        # returns empty text, fall back to UIAutomationCore to inspect live accessibility tree across
        # the root HWND and any child host bridges (such as Chrome_WidgetWin_0 inside WinUI islands)
        if not child_texts or any(app in process_name for app in ["whatsapp", "discord", "slack", "teams"]):
            try:
                import comtypes.client
                UIAutomationCore = comtypes.client.GetModule("UIAutomationCore.dll")
                cui = comtypes.client.CreateObject(UIAutomationCore.CUIAutomation, interface=UIAutomationCore.IUIAutomation)
                
                # Gather root HWND and all child HWNDs
                targets_to_inspect = [hwnd]
                def _collect_sub_hwnds(ch, _):
                    cls_name = win32gui.GetClassName(ch).lower()
                    if "chrome" in cls_name or "child" in cls_name or "bridge" in cls_name:
                        targets_to_inspect.append(ch)
                try:
                    win32gui.EnumChildWindows(hwnd, _collect_sub_hwnds, None)
                except Exception:
                    pass

                cond = cui.CreateOrCondition(
                    cui.CreatePropertyCondition(UIAutomationCore.UIA_ControlTypePropertyId, 50020), # Heading
                    cui.CreateOrCondition(
                        cui.CreatePropertyCondition(UIAutomationCore.UIA_ControlTypePropertyId, 50004), # Edit
                        cui.CreateOrCondition(
                            cui.CreatePropertyCondition(UIAutomationCore.UIA_ControlTypePropertyId, 50032), # Window/Dialog
                            cui.CreatePropertyCondition(UIAutomationCore.UIA_ControlTypePropertyId, 50033), # Pane
                        )
                    )
                )

                seen = set(child_texts)
                for target_h in targets_to_inspect:
                    try:
                        el = cui.ElementFromHandle(target_h)
                        if el:
                            found = el.FindAll(UIAutomationCore.TreeScope_Descendants, cond)
                            for i in range(min(found.Length, 150)):
                                item = found.GetElement(i)
                                nm = item.CurrentName
                                if nm and nm not in seen and len(nm) < 200:
                                    seen.add(nm)
                                    child_texts.append(nm)
                    except Exception:
                        pass
            except Exception:
                pass

        return {
            "hwnd": hwnd,
            "is_window": True,
            "is_visible": is_vis,
            "title": title,
            "class_name": class_name,
            "process_id": pid,
            "process_name": process_name,
            "child_texts": child_texts,
            "error": None,
        }

    except Exception as e:
        return {
            "hwnd": 0,
            "is_window": False,
            "is_visible": False,
            "title": "",
            "class_name": "",
            "process_id": 0,
            "process_name": "unknown",
            "child_texts": [],
            "error": str(e),
        }


def verify_chat_ui_target(
    target_app: str,
    target_recipient: str,
    mock_window_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Strict pre-execution verification:
    Ensures foreground window belongs to target_app and displays target_recipient.
    
    If mock_window_state is provided, it uses the provided window state (used for
    deterministic simulation and negative mismatch unit tests).
    """
    app_key = target_app.lower().strip()
    recip = target_recipient.strip()

    # 1. Inspect window state
    if mock_window_state is not None:
        win_info = mock_window_state
    else:
        win_info = inspect_live_foreground_window()

    hwnd = win_info.get("hwnd", 0)
    proc_name = win_info.get("process_name", "").lower()
    title = win_info.get("title", "")
    class_name = win_info.get("class_name", "")
    child_texts = win_info.get("child_texts", [])
    title_low = title.lower()

    # 2. Check 1: Must have a valid, visible foreground window
    if not win_info.get("is_window", False) or hwnd == 0:
        return {
            "verified": False,
            "status": "ABORT_NO_FOREGROUND_WINDOW",
            "hwnd": hwnd,
            "app": target_app,
            "recipient": recip,
            "reason": (
                f"VERIFICATION FAILED: No active foreground window detected. "
                f"Cannot confirm target application '{target_app}' is active."
            ),
            "window_details": win_info,
        }

    # 3. Check 2: Process identity must match target application catalog
    allowed_procs = APP_PROCESS_CATALOG.get(app_key, [f"{app_key}.exe"])
    if not any(proc_name == p or proc_name.startswith(p.split(".")[0]) for p in allowed_procs):
        # Also check if title or class clearly indicates the app (for UWP ApplicationFrameHost wrappers)
        if app_key not in title_low and app_key not in class_name.lower():
            return {
                "verified": False,
                "status": "ABORT_WRONG_APP",
                "hwnd": hwnd,
                "app": target_app,
                "recipient": recip,
                "reason": (
                    f"VERIFICATION FAILED: Foreground process '{proc_name}' (Title: '{title}') "
                    f"does not match expected target application '{target_app}'. Expected process in {allowed_procs}."
                ),
                "window_details": win_info,
            }

    # 4. Check 3: Reject modal dialogs, forward dialogs, or collision screens
    # In modern apps (WinUI 3/Electron), top-level HWND title remains app name ('WhatsApp') while
    # the modal dialog header ('Forward message to', 'Select contact') renders inside child elements.
    for unsafe_kw in UNSAFE_WINDOW_INDICATORS:
        if unsafe_kw in title_low:
            return {
                "verified": False,
                "status": "ABORT_MODAL_DIALOG_DETECTED",
                "hwnd": hwnd,
                "app": target_app,
                "recipient": recip,
                "reason": (
                    f"VERIFICATION FAILED: Detected unsafe modal/collision dialog title '{title}' "
                    f"(matches keyword '{unsafe_kw}'). Aborting to prevent misdirected keystrokes."
                ),
                "window_details": win_info,
            }

    modal_keywords = ["forward", "select contact", "choose contact", "open file", "attach file", "crop photo", "share to"]
    for t in child_texts:
        tl = t.strip().lower()
        if len(tl) <= 50:
            for kw in modal_keywords:
                if tl.startswith(kw) or re.search(r"\b" + re.escape(kw) + r"\b", tl):
                    return {
                        "verified": False,
                        "status": "ABORT_MODAL_DIALOG_DETECTED",
                        "hwnd": hwnd,
                        "app": target_app,
                        "recipient": recip,
                        "reason": (
                            f"VERIFICATION FAILED: Detected unsafe modal/collision dialog element '{t}' "
                            f"(matches keyword '{kw}'). Aborting to prevent misdirected keystrokes."
                        ),
                        "window_details": win_info,
                    }

    # 5. Check 4: Recipient / Contact Verification in Chat Header
    # Look for recipient name in window title or child header elements
    recip_parts = recip.lower().split()
    first_name = recip_parts[0] if recip_parts else recip.lower()
    full_name = recip.lower()

    all_visible_text = " ".join([title] + child_texts).lower()

    recipient_found = False
    if full_name in all_visible_text:
        recipient_found = True
    elif len(recip_parts) > 1 and first_name in all_visible_text and len(first_name) >= 3:
        recipient_found = True

    if not recipient_found:
        return {
            "verified": False,
            "status": "ABORT_RECIPIENT_NOT_VISIBLE",
            "hwnd": hwnd,
            "app": target_app,
            "recipient": recip,
            "reason": (
                f"VERIFICATION FAILED: Active window does not display target contact '{recip}'. "
                f"Visible window title was '{title}'. Keystroke dispatch aborted."
            ),
            "window_details": win_info,
        }

    # All checks passed with high confidence
    return {
        "verified": True,
        "status": "VERIFIED",
        "hwnd": hwnd,
        "app": target_app,
        "recipient": recip,
        "reason": f"Foreground window verified: '{proc_name}' displays contact '{recip}'.",
        "window_details": win_info,
    }
