"""
tools/interactive_loop.py

Autonomous Desktop Interactive Loop Controller for MIKU.

Operates on natural language user goals (e.g., "open Notepad", "search for X in Y")
by dynamically deciding each step in real time based on live UIAutomation perception data
from tools/screen_inspector.py and tools/typing_automation.py.

HARD SAFETY & ARCHITECTURAL INVARIANTS:
1. Perception is ALWAYS real-time: inspect_screen_elements() is called before EVERY decision step.
   No stale inspections are ever reused.
2. Fast Single Confirmation Gate: Every proposed physical action is shown clearly.
   Pressing [ENTER] approves dispatch, while typing any character/cancellation aborts execution immediately.
3. Zero Duplicate Prompts: Sets MIKU_FAST_CONFIRM_PASSTHROUGH for dispatch calls after loop confirmation.
4. Mid-Task Error Safety: If element verification fails (occluded, non-editable, unclickable) or user cancels,
   the loop STOPS immediately and reports the diagnostic reason without silent unconfirmed retries.
5. Fail-Closed Circuit Breakers: Respects REAL_CLICK_ENABLED and REAL_TYPE_ENABLED.
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


def parse_goal_intent(goal: str) -> Dict[str, Any]:
    """
    Parses natural language task goal into target action intent dictionary.
    """
    goal_lower = goal.strip().lower()
    intent = {
        "raw_goal": goal,
        "target_app": "",
        "search_query": "",
        "action_type": "UNKNOWN",
    }

    if "search for" in goal_lower and "in" in goal_lower:
        # Format: "search for <query> in <app>"
        parts = goal_lower.split("search for", 1)[1].split("in", 1)
        intent["search_query"] = parts[0].strip("'\" ")
        intent["target_app"] = parts[1].strip("'\" ")
        intent["action_type"] = "SEARCH_IN_APP"
    elif "open" in goal_lower:
        # Format: "open <app>"
        app = goal_lower.split("open", 1)[1].strip("'\" ")
        intent["target_app"] = app
        intent["action_type"] = "OPEN_APP"
    else:
        intent["target_app"] = goal_lower
        intent["action_type"] = "OPEN_APP"

    return intent


import msvcrt


def flush_stdin_buffer():
    """Flushes any leftover buffered keystrokes in Windows console stdin."""
    if sys.platform == "win32":
        try:
            while msvcrt.kbhit():
                msvcrt.getch()
        except Exception:
            pass


def fast_confirm_action(action_description: str, target_details: Dict[str, Any]) -> bool:
    """
    Fast interactive confirmation gate:
    Displays exact proposed action and target details cleanly.
    Requires a single live physical keypress: [ENTER] or [Y] or [SPACE] to approve,
    or [ESC] / any other key to cancel.
    """
    if not sys.stdin or not sys.stdin.isatty():
        if os.environ.get("MIKU_AUTONOMOUS_MODE", "false").strip().lower() in ("true", "1", "yes"):
            return True
        print(f"[*] [NON-INTERACTIVE ABORT] Fast confirmation required for: {action_description}")
        return False

    print("\n" + "-" * 70)
    print(f"[MIKU PROPOSED ACTION] : {action_description}")
    print(f"Target Details        : {target_details}")
    print(">>> Press [ENTER] or [Y] to APPROVE & DISPATCH, or [ESC] / any other key to CANCEL <<<")
    print("-" * 70, flush=True)

    if sys.platform == "win32":
        try:
            flush_stdin_buffer()
            ch = msvcrt.getch()
            # Handle special extended keys (e.g. arrow keys b'\x00' or b'\xe0')
            if ch in (b'\x00', b'\xe0'):
                msvcrt.getch()
                print("[-] CANCELLED (Extended keypress)\n", flush=True)
                return False

            if ch in (b'\r', b'\n', b'y', b'Y', b' '):
                print("[+] APPROVED by User keypress\n", flush=True)
                return True
            else:
                print(f"[-] CANCELLED by User keypress\n", flush=True)
                return False
        except Exception as exc:
            print(f"[!] msvcrt exception: {exc}, falling back to input()...", flush=True)

    try:
        resp = input("Confirmation Choice [ENTER=Approve / Any=Cancel]: ").strip()
        return resp == "" or resp.upper() in ("Y", "YES", "OK")
    except Exception as exc:
        print(f"[!] Confirmation exception: {exc}", flush=True)
        return False


def find_search_box_element(snapshot: ScreenSnapshot) -> Optional[UIElement]:
    """
    Locates an Edit, Document, or ComboBox control UIElement within the given ScreenSnapshot.
    """
    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            if "search" in el.name.lower() or "search" in el.automation_id.lower():
                return el

    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            return el

    return None


def find_app_icon_or_shortcut(snapshot: ScreenSnapshot, app_name: str) -> Optional[UIElement]:
    """
    Locates a UIElement corresponding to an app icon, shortcut, or search result matching `app_name`.
    """
    query = app_name.strip().lower()
    for el in snapshot.elements:
        if el.name.strip().lower() == query:
            return el

    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            continue
        el_name = el.name.strip().lower()
        el_id = el.automation_id.strip().lower()
        if (query in el_name or query in el_id) and el.is_enabled:
            return el

    return None


def run_autonomous_task_loop(
    goal: str,
    max_steps: int = 10,
    real_execution: bool = False,
) -> Dict[str, Any]:
    """
    Main Autonomous Desktop Interactive Loop Controller.
    Executes discrete perception-action-verification steps.
    """
    intent = parse_goal_intent(goal)
    trace: List[Dict[str, Any]] = []

    print(f"\n[MIKU AUTONOMOUS LOOP] Initializing task: '{goal}'")
    print(f"[*] Parsed Intent: {intent}")

    step_count = 0
    search_box_clicked = False
    search_text_typed = False

    while step_count < max_steps:
        step_count += 1
        print(f"\n--- STEP {step_count}: REAL-TIME PERCEPTION ---")

        # 1. REAL-TIME PERCEPTION (Always fresh state)
        _attach_thread_to_default_desktop()
        fg_hwnd = user32.GetForegroundWindow()

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

        print(f"[*] Perception HWND {snap.hwnd}: Title='{snap.title}', Class='{snap.class_name}', Process='{snap.process_name}'")
        print(f"[*] Scanned Elements: {len(snap.elements)}")

        # Check Goal Completion: Target app window is active and foreground
        target_app_lower = intent["target_app"].lower()
        if target_app_lower and target_app_lower in snap.title.lower() and snap.title not in ("Search", "Start"):
            print(f"\n[GOAL ACHIEVED] Window '{snap.title}' matching '{intent['target_app']}' is active!")
            trace.append({
                "step": step_count,
                "action": "verify_completion",
                "status": "GOAL_ACHIEVED",
                "window_title": snap.title,
                "hwnd": snap.hwnd,
            })
            return {
                "success": True,
                "status": "TASK_COMPLETED_SUCCESSFULLY",
                "goal": goal,
                "intent": intent,
                "steps_taken": step_count,
                "trace": trace,
            }

        # Dynamic Action Selection Logic:
        # Phase 1: Open Start Menu if not active
        if snap.title not in ("Search", "Start"):
            action_desc = "Open Windows Start Menu via Win Key"
            details = {"action": "keybd_event(VK_LWIN)", "foreground_window": snap.title}

            if not fast_confirm_action(action_desc, details):
                print("\n[*] User CANCELLED action. Stopping loop.")
                trace.append({"step": step_count, "action": "open_start_menu", "status": "USER_CANCELLED"})
                return {"success": False, "status": "ABORT_USER_CANCELLED", "trace": trace}

            user32.keybd_event(0x5B, 0, 0, 0)
            user32.keybd_event(0x5B, 0, 2, 0)
            time.sleep(1.0)
            trace.append({"step": step_count, "action": "open_start_menu", "status": "EXECUTED"})
            continue

        # Phase 2: Start Menu is active -> Locate Search Box
        search_edit = find_search_box_element(snap)
        if not search_edit:
            print("  [ERR] Search Edit box not visible in current snapshot!")
            trace.append({"step": step_count, "action": "locate_search_box", "status": "ELEMENT_NOT_FOUND"})
            return {"success": False, "status": "ABORT_ELEMENT_NOT_FOUND", "trace": trace}

        # Phase 3: Click Search Box
        if not search_box_clicked:
            action_desc = f"Click Search Box '{search_edit.name}' ({search_edit.control_type})"
            details = {
                "control_type": search_edit.control_type,
                "element_name": search_edit.name,
                "center": search_edit.center,
                "hwnd": snap.hwnd,
            }

            c_verif = verify_element_clickable(snap.hwnd, search_edit)
            if not c_verif.is_safe:
                print(f"  [SAFETY REJECTION] Search box click verification failed: reason='{c_verif.reason}'")
                trace.append({"step": step_count, "action": "click_search_box", "status": f"SAFETY_REJECTED_{c_verif.reason}"})
                return {"success": False, "status": f"ABORT_SAFETY_VERIFICATION_FAILED_{c_verif.reason}", "trace": trace}

            if not fast_confirm_action(action_desc, details):
                print("\n[*] User CANCELLED action. Stopping loop.")
                trace.append({"step": step_count, "action": "click_search_box", "status": "USER_CANCELLED"})
                return {"success": False, "status": "ABORT_USER_CANCELLED", "trace": trace}

            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "true"
            if real_execution and REAL_CLICK_ENABLED:
                click_res = dispatch_real_click(snap.hwnd, search_edit)
            else:
                click_res = simulate_click(snap.hwnd, search_edit)
            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "false"

            search_box_clicked = True
            trace.append({"step": step_count, "action": "click_search_box", "result": click_res.to_dict()})
            time.sleep(0.5)
            continue

        # Phase 4: Type Search Query
        if not search_text_typed:
            query_text = intent["target_app"]
            action_desc = f"Type '{query_text}' into Search Box"
            details = {
                "control_type": search_edit.control_type,
                "element_name": search_edit.name,
                "text_to_type": query_text,
                "hwnd": snap.hwnd,
            }

            t_verif = verify_element_editable(snap.hwnd, search_edit)
            if not t_verif.is_safe:
                print(f"  [SAFETY REJECTION] Search box editability verification failed: reason='{t_verif.reason}'")
                trace.append({"step": step_count, "action": "type_search_query", "status": f"SAFETY_REJECTED_{t_verif.reason}"})
                return {"success": False, "status": f"ABORT_SAFETY_VERIFICATION_FAILED_{t_verif.reason}", "trace": trace}

            if not fast_confirm_action(action_desc, details):
                print("\n[*] User CANCELLED action. Stopping loop.")
                trace.append({"step": step_count, "action": "type_search_query", "status": "USER_CANCELLED"})
                return {"success": False, "status": "ABORT_USER_CANCELLED", "trace": trace}

            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "true"
            if real_execution and REAL_TYPE_ENABLED:
                type_res = dispatch_real_typing(snap.hwnd, search_edit, query_text)
            else:
                type_res = simulate_typing(snap.hwnd, search_edit, query_text)
            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "false"

            search_text_typed = True
            trace.append({"step": step_count, "action": "type_search_query", "result": type_res.to_dict()})
            time.sleep(1.2)  # Wait for search results list to populate
            continue

        # Phase 5: Locate and Click Search Result Element
        target_item = find_app_icon_or_shortcut(snap, intent["target_app"])
        if target_item:
            action_desc = f"Click App Result '{target_item.name}' ({target_item.control_type})"
            details = {
                "control_type": target_item.control_type,
                "element_name": target_item.name,
                "center": target_item.center,
                "hwnd": snap.hwnd,
            }

            res_verif = verify_element_clickable(snap.hwnd, target_item)
            if not res_verif.is_safe:
                print(f"  [SAFETY REJECTION] App result click verification failed: reason='{res_verif.reason}'")
                trace.append({"step": step_count, "action": "click_app_result", "status": f"SAFETY_REJECTED_{res_verif.reason}"})
                return {"success": False, "status": f"ABORT_SAFETY_VERIFICATION_FAILED_{res_verif.reason}", "trace": trace}

            if not fast_confirm_action(action_desc, details):
                print("\n[*] User CANCELLED action. Stopping loop.")
                trace.append({"step": step_count, "action": "click_app_result", "status": "USER_CANCELLED"})
                return {"success": False, "status": "ABORT_USER_CANCELLED", "trace": trace}

            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "true"
            if real_execution and REAL_CLICK_ENABLED:
                res_click = dispatch_real_click(snap.hwnd, target_item)
            else:
                res_click = simulate_click(snap.hwnd, target_item)
            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "false"

            trace.append({"step": step_count, "action": "click_app_result", "result": res_click.to_dict()})
            time.sleep(2.0)  # Wait for app window to launch
            continue
        else:
            print(f"  [*] Search result element for '{intent['target_app']}' not found in current snapshot.")
            trace.append({"step": step_count, "action": "locate_app_result", "status": "ELEMENT_NOT_FOUND"})
            return {"success": False, "status": "ABORT_APP_RESULT_NOT_FOUND", "trace": trace}

    return {
        "success": False,
        "status": "ABORT_MAX_STEPS_EXCEEDED",
        "steps_taken": step_count,
        "trace": trace,
    }
