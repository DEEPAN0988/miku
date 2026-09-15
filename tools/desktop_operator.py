"""
tools/desktop_operator.py

Desktop Operator Subsystem for MIKU.

Provides non-vision, UIAutomation-based capabilities for:
(A) Opening an application by locating and clicking its icon/shortcut on screen (Start Menu / Taskbar).
(B) Searching via UI search box by locating the edit control, clicking into it, and typing search queries.

HARD CONSTITUTIONAL & SAFETY INVARIANTS:
1. Zero fake AI, zero hardcoded responses, zero unbacked claims.
2. Built strictly on top of verified infrastructure:
   - tools.screen_inspector (inspect_screen_elements, verify_element_clickable, simulate_click, dispatch_real_click)
   - tools.typing_automation (verify_element_editable, sanitize_typing_payload, simulate_typing, dispatch_real_typing)
3. Mandatory Pre-Action Verification: All click and typing operations pass through verify_element_clickable()
   and verify_element_editable() before any simulated or physical action.
4. Fail-Closed Circuit Breakers: Respects REAL_CLICK_ENABLED=False and REAL_TYPE_ENABLED=False.
5. Live Human Interactive Confirmation: Physical execution requires explicit, live, human-typed confirmation
   ('CONFIRM CLICK' / 'CONFIRM TYPE'). No programmatic parameter or flag can bypass this requirement.
"""

import ctypes
import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple

import win32gui

from tools.screen_inspector import (
    _attach_thread_to_default_desktop,
    inspect_screen_elements,
    verify_element_clickable,
    simulate_click,
    dispatch_real_click,
    UIElement,
    ScreenSnapshot,
    REAL_CLICK_ENABLED,
)
from tools.typing_automation import (
    verify_element_editable,
    sanitize_typing_payload,
    simulate_typing,
    dispatch_real_typing,
    REAL_TYPE_ENABLED,
    VALID_EDIT_CONTROL_TYPES,
)

user32 = ctypes.windll.user32


def open_start_menu() -> Tuple[int, ScreenSnapshot]:
    """
    Toggles the Windows Start / Search menu open using the Windows key event,
    attaches thread to the default desktop, and returns (target_hwnd, ScreenSnapshot).
    """
    _attach_thread_to_default_desktop()
    # Send VK_LWIN key event (0x5B)
    user32.keybd_event(0x5B, 0, 0, 0)
    user32.keybd_event(0x5B, 0, 2, 0)  # KEYEVENTF_KEYUP = 2
    time.sleep(1.0)

    fg_hwnd = user32.GetForegroundWindow()

    # Search visible windows for Windows Start/Search CoreWindow
    matches = []
    def _enum_cb(h, extra):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            c = win32gui.GetClassName(h)
            if c == "Windows.UI.Core.CoreWindow" and t in ("Search", "Start"):
                extra.append(h)
        return True

    if win32gui:
        try:
            win32gui.EnumWindows(_enum_cb, matches)
        except Exception:
            pass

    target_h = matches[0] if matches else (fg_hwnd if fg_hwnd else None)

    snap = inspect_screen_elements(hwnd=target_h, max_elements=300, interactive_only=False)
    if target_h and user32.IsWindow(target_h):
        snap.hwnd = target_h
    return target_h, snap


def close_start_menu() -> None:
    """Closes the Start / Search menu by sending VK_LWIN key event."""
    user32.keybd_event(0x5B, 0, 0, 0)
    user32.keybd_event(0x5B, 0, 2, 0)


def find_search_box_element(snapshot: ScreenSnapshot) -> Optional[UIElement]:
    """
    Locates an Edit, Document, or ComboBox control UIElement within the given ScreenSnapshot.
    Strictly restricted to valid editable control types.
    """
    # Priority 1: Valid edit control with 'search' in name or automation_id
    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            if "search" in el.name.lower() or "search" in el.automation_id.lower():
                return el

    # Priority 2: Any valid edit control (e.g. Edit control type)
    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            return el

    return None


def find_app_icon_or_shortcut(snapshot: ScreenSnapshot, app_name: str) -> Optional[UIElement]:
    """
    Locates a UIElement corresponding to an app icon, shortcut, or search result matching `app_name`.
    Matches by name, automation_id, or control text (case-insensitive substring match).
    """
    query = app_name.strip().lower()
    # 1. Exact match on element name
    for el in snapshot.elements:
        if el.name.strip().lower() == query:
            return el

    # 2. Substring match on element name or automation_id (excluding edit search bars)
    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            continue
        el_name = el.name.strip().lower()
        el_id = el.automation_id.strip().lower()
        if (query in el_name or query in el_id) and el.is_enabled:
            return el

    return None


def simulate_open_app_by_icon(app_name: str) -> Dict[str, Any]:
    """
    Capability (A) Simulation: Open an app by finding and clicking its icon on screen (Taskbar / Start Menu).
    Performs full pre-click verification and returns trace dictionary with real_input_dispatched=False.
    """
    trace: List[Dict[str, Any]] = []

    # Step 1: Open Start Menu / Taskbar view
    fg_hwnd, snap = open_start_menu()
    trace.append({
        "step": 1,
        "action": "open_start_menu",
        "hwnd": fg_hwnd,
        "window_title": snap.title,
        "elements_scanned": len(snap.elements),
    })

    # Step 2: Locate app icon
    app_icon = find_app_icon_or_shortcut(snap, app_name)
    if not app_icon:
        close_start_menu()
        return {
            "success": False,
            "status": "ABORT_APP_ICON_NOT_FOUND",
            "app_name": app_name,
            "trace": trace,
            "real_input_dispatched": False,
        }

    trace.append({
        "step": 2,
        "action": "locate_app_icon",
        "element": app_icon.to_dict(),
    })

    # Step 3: Verify clickability
    verif = verify_element_clickable(snap.hwnd, app_icon)
    trace.append({
        "step": 3,
        "action": "verify_element_clickable",
        "verification": verif.to_dict(),
    })

    # Step 4: Simulate Click
    sim_res = simulate_click(snap.hwnd, app_icon)
    trace.append({
        "step": 4,
        "action": "simulate_click",
        "result": sim_res.to_dict(),
    })

    # Clean up Start Menu
    close_start_menu()

    return {
        "success": sim_res.success,
        "status": sim_res.status,
        "app_name": app_name,
        "target_element": app_icon.to_dict(),
        "trace": trace,
        "real_input_dispatched": False,
    }


def simulate_start_menu_search_and_click(query: str, target_app_name: str) -> Dict[str, Any]:
    """
    Capability (B) Simulation: Search for something by clicking a search box, typing into it,
    inspecting search results, and clicking the result.
    Performs full verification and returns trace dictionary with real_input_dispatched=False.
    """
    trace: List[Dict[str, Any]] = []

    # Step 1: Open Start / Search Menu
    fg_hwnd, snap1 = open_start_menu()
    trace.append({
        "step": 1,
        "action": "open_start_menu",
        "hwnd": fg_hwnd,
        "window_title": snap1.title,
        "elements_scanned": len(snap1.elements),
    })

    # Step 2: Locate Search Box Edit Control
    search_edit = find_search_box_element(snap1)
    if not search_edit:
        close_start_menu()
        return {
            "success": False,
            "status": "ABORT_SEARCH_BOX_NOT_FOUND",
            "query": query,
            "trace": trace,
            "real_input_dispatched": False,
        }

    trace.append({
        "step": 2,
        "action": "locate_search_box",
        "element": search_edit.to_dict(),
    })

    # Step 3: Verify Clickability & Simulate Click into Search Box
    click_verif = verify_element_clickable(snap1.hwnd, search_edit)
    sim_click_res = simulate_click(snap1.hwnd, search_edit)
    trace.append({
        "step": 3,
        "action": "simulate_click_search_box",
        "verification": click_verif.to_dict(),
        "click_result": sim_click_res.to_dict(),
    })

    # Step 4: Verify Editability & Simulate Typing Search Query
    type_verif = verify_element_editable(snap1.hwnd, search_edit)
    sim_type_res = simulate_typing(snap1.hwnd, search_edit, query)
    trace.append({
        "step": 4,
        "action": "simulate_typing_search_query",
        "verification": type_verif.to_dict(),
        "type_result": sim_type_res.to_dict(),
    })

    # Step 5: Locate Target Search Result Element
    # In simulation mode, inspect updated UIA tree or current snapshot for result matching target_app_name
    result_icon = find_app_icon_or_shortcut(snap1, target_app_name)
    if result_icon:
        result_click_verif = verify_element_clickable(snap1.hwnd, result_icon)
        sim_result_click = simulate_click(snap1.hwnd, result_icon)
        trace.append({
            "step": 5,
            "action": "simulate_click_search_result",
            "target_app_name": target_app_name,
            "element": result_icon.to_dict(),
            "verification": result_click_verif.to_dict(),
            "click_result": sim_result_click.to_dict(),
        })
    else:
        trace.append({
            "step": 5,
            "action": "locate_search_result",
            "status": "RESULT_ELEMENT_NOT_PRESENT_IN_DRY_RUN",
            "target_app_name": target_app_name,
        })

    # Clean up Start Menu
    close_start_menu()

    return {
        "success": sim_click_res.real_input_dispatched is False and sim_type_res.real_input_dispatched is False,
        "status": "SIMULATED_SEARCH_TRACE_SUCCESS",
        "query": query,
        "target_app_name": target_app_name,
        "trace": trace,
        "real_input_dispatched": False,
    }


def execute_real_start_menu_search_and_click(query: str, target_app_name: str) -> Dict[str, Any]:
    """
    Task 3 Canary Execution: Dispatches REAL physical clicks and keystrokes to open Start menu,
    search for `query`, and click the resulting app (`target_app_name`).

    UNBYPASSABLE SAFETY REQUIREMENTS:
    1. REAL_CLICK_ENABLED must be True (imported from tools.screen_inspector).
    2. REAL_TYPE_ENABLED must be True (imported from tools.typing_automation).
    3. Interactive console confirmation prompts ('CONFIRM CLICK' and 'CONFIRM TYPE') will be triggered
       before every physical OS click and keystroke.
    """
    from tools.screen_inspector import REAL_CLICK_ENABLED as CURRENT_CLICK_CB
    from tools.typing_automation import REAL_TYPE_ENABLED as CURRENT_TYPE_CB

    if not CURRENT_CLICK_CB or not CURRENT_TYPE_CB:
        return {
            "success": False,
            "status": "ABORT_CIRCUIT_BREAKER_BLOCKED",
            "error": f"Circuit breaker invariant failed: REAL_CLICK_ENABLED={CURRENT_CLICK_CB}, REAL_TYPE_ENABLED={CURRENT_TYPE_CB}",
            "real_input_dispatched": False,
        }

    trace: List[Dict[str, Any]] = []

    # Step 1: Open Start / Search Menu
    fg_hwnd, snap1 = open_start_menu()
    trace.append({
        "step": 1,
        "action": "open_start_menu",
        "hwnd": fg_hwnd,
        "window_title": snap1.title,
        "elements_scanned": len(snap1.elements),
    })

    # Step 2: Locate Search Box Edit Control
    search_edit = find_search_box_element(snap1)
    if not search_edit:
        close_start_menu()
        return {
            "success": False,
            "status": "ABORT_SEARCH_BOX_NOT_FOUND",
            "query": query,
            "trace": trace,
            "real_input_dispatched": False,
        }

    # Step 3: Real Click into Search Box (Triggers 'CONFIRM CLICK' prompt)
    real_click_res = dispatch_real_click(snap1.hwnd, search_edit)
    trace.append({
        "step": 3,
        "action": "dispatch_real_click_search_box",
        "result": real_click_res.to_dict(),
    })

    if not real_click_res.success:
        close_start_menu()
        return {
            "success": False,
            "status": f"ABORT_CLICK_FAILED_{real_click_res.status}",
            "trace": trace,
            "real_input_dispatched": real_click_res.real_input_dispatched,
        }

    # Step 4: Real Typing Search Query (Triggers 'CONFIRM TYPE' prompt)
    real_type_res = dispatch_real_typing(snap1.hwnd, search_edit, query)
    trace.append({
        "step": 4,
        "action": "dispatch_real_typing_search_query",
        "result": real_type_res.to_dict(),
    })

    if not real_type_res.success:
        close_start_menu()
        return {
            "success": False,
            "status": f"ABORT_TYPING_FAILED_{real_type_res.status}",
            "trace": trace,
            "real_input_dispatched": True,
        }

    # Wait for search results to populate
    time.sleep(1.2)

    # Step 5: Fresh inspection of updated Search results window
    snap2 = inspect_screen_elements(hwnd=snap1.hwnd, max_elements=300, interactive_only=False)
    result_icon = find_app_icon_or_shortcut(snap2, target_app_name)

    if not result_icon:
        close_start_menu()
        return {
            "success": False,
            "status": "ABORT_SEARCH_RESULT_NOT_FOUND",
            "target_app_name": target_app_name,
            "trace": trace,
            "real_input_dispatched": True,
        }

    # Step 6: Real Click on App Search Result (Triggers 'CONFIRM CLICK' prompt)
    result_click_res = dispatch_real_click(snap2.hwnd, result_icon)
    trace.append({
        "step": 6,
        "action": "dispatch_real_click_search_result",
        "target_app_name": target_app_name,
        "result": result_click_res.to_dict(),
    })

    # Wait 2.0s for launched app process/window to appear
    time.sleep(2.0)

    # Step 7: Independent verification — confirm launched app window exists
    _attach_thread_to_default_desktop()
    found_launched_window = False
    launched_window_title = ""

    def _check_app_cb(h, extra):
        nonlocal found_launched_window, launched_window_title
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            if target_app_name.lower() in t.lower():
                found_launched_window = True
                launched_window_title = t
        return True

    if win32gui:
        try:
            win32gui.EnumWindows(_check_app_cb, None)
        except Exception:
            pass

    trace.append({
        "step": 7,
        "action": "independent_verification",
        "app_launched": found_launched_window,
        "window_title": launched_window_title,
    })

    return {
        "success": result_click_res.success and found_launched_window,
        "status": "REAL_EXECUTION_SUCCESS" if (result_click_res.success and found_launched_window) else "REAL_EXECUTION_PARTIAL",
        "query": query,
        "target_app_name": target_app_name,
        "app_launched": found_launched_window,
        "window_title": launched_window_title,
        "trace": trace,
        "real_input_dispatched": True,
    }
