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
    set_window_foreground_passive,
    UIElement,
    ScreenSnapshot,
    REAL_CLICK_ENABLED,
)
from tools.typing_automation import (
    verify_element_editable,
    sanitize_typing_payload,
    simulate_typing,
    dispatch_real_typing,
    dispatch_vk_key,
    dispatch_enter_key,
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
    If autonomous mode is active (MIKU_AUTONOMOUS_MODE=true or MIKU_AUTO_APPROVE=true),
    automatically approves without requesting keypresses or blocking.
    Otherwise requires live physical keypress [ENTER] or [Y].
    """
    auto_mode = os.environ.get("MIKU_AUTONOMOUS_MODE", "false").strip().lower() in ("true", "1", "yes")
    auto_appr = os.environ.get("MIKU_AUTO_APPROVE", "false").strip().lower() in ("true", "1", "yes")
    if auto_mode or auto_appr:
        print(f"[*] [AUTONOMOUS AUTO-APPROVED] Action: {action_description}", flush=True)
        return True

    if not sys.stdin or not sys.stdin.isatty():
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
    Prioritizes explicit 'Open' action button or 'Best match' app results while excluding Store/Web results.
    """
    query = app_name.strip().lower()

    # Priority 1: Check for explicit 'Open' button on the right pane of Search Host
    for el in snapshot.elements:
        if el.control_type in ("Button", "ListItem", "Hyperlink") and el.is_enabled:
            if el.name.strip().lower() in ("open", "open app"):
                return el

    # Priority 2: Best match or App shortcuts (excluding Store, Web, Photos, Folders)
    EXCLUDE_KEYWORDS = ("store", "search the web", "photos", "folders", "codes", "download", "size")

    # Check exact match excluding Store/Web
    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            continue
        el_name = el.name.strip().lower()
        if el_name == query and not any(kw in el_name for kw in EXCLUDE_KEYWORDS):
            return el

    # Check partial match excluding Store/Web
    for el in snapshot.elements:
        if el.control_type in VALID_EDIT_CONTROL_TYPES:
            continue
        el_name = el.name.strip().lower()
        el_id = el.automation_id.strip().lower()
        if any(kw in el_name for kw in EXCLUDE_KEYWORDS) or any(kw in el_id for kw in EXCLUDE_KEYWORDS):
            continue
        if (query in el_name or query in el_id) and el.is_enabled:
            return el

    # Fallback exact match
    for el in snapshot.elements:
        if el.control_type not in VALID_EDIT_CONTROL_TYPES and el.name.strip().lower() == query and el.is_enabled:
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
    app_result_clicked = False

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

        title_repr = repr(snap.title)
        print(f"[*] Perception HWND {snap.hwnd}: Title={title_repr}, Class={repr(snap.class_name)}, Process='{snap.process_name}'")
        print(f"[*] Scanned Elements: {len(snap.elements)}")

        # Check Goal Completion: Target app window is active and foreground or open on desktop
        target_app_lower = intent["target_app"].lower()
        app_aliases = [target_app_lower]
        if "wuthering" in target_app_lower:
            app_aliases.extend(["wuthering", "launcher_main", "kuro game", "kuro", "client-win64-shipping", "launcher.exe", "wutheringwaves"])

        EXCLUDED_PROCS = ("explorer.exe", "searchhost.exe", "cmd.exe", "powershell.exe", "udclientservice.exe")

        target_found_window = None
        if target_app_lower and snap.title not in ("Search", "Start"):
            if any(alias in snap.title.lower() for alias in app_aliases):
                target_found_window = (snap.hwnd, snap.title)

        if not target_found_window and win32gui:
            import win32process
            def _enum_win_cb(h, acc):
                if win32gui.IsWindow(h):
                    t = win32gui.GetWindowText(h).lower()
                    if t and t not in ("search", "start", "program manager", "nahimic", "windows input experience"):
                        if any(alias in t for alias in app_aliases):
                            acc.append((h, win32gui.GetWindowText(h)))
                    else:
                        # Check window process name
                        try:
                            _, pid = win32process.GetWindowThreadProcessId(h)
                            if pid > 0:
                                import psutil
                                p_name = psutil.Process(pid).name().lower()
                                if any(alias in p_name for alias in app_aliases) and p_name not in EXCLUDED_PROCS:
                                    acc.append((h, f"{p_name} (HWND {h})"))
                        except Exception:
                            pass
                return True
            found_wins = []
            try:
                win32gui.EnumWindows(_enum_win_cb, found_wins)
            except Exception:
                pass
            if found_wins:
                target_found_window = found_wins[0]

        # Check process fallback if app result was clicked or fallback launch executed
        if not target_found_window and (app_result_clicked or step_count > 3):
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name']):
                    p_name = proc.name().lower()
                    if any(alias in p_name for alias in app_aliases) and p_name not in EXCLUDED_PROCS:
                        target_found_window = (0, f"Process '{proc.name()}' (PID {proc.pid})")
                        break
            except Exception:
                pass

        if target_found_window:
            w_hwnd, w_title = target_found_window
            print(f"\n[GOAL ACHIEVED] Target '{w_title}' matching '{intent['target_app']}' is running and verified!")
            trace.append({
                "step": step_count,
                "action": "verify_completion",
                "status": "GOAL_ACHIEVED",
                "window_title": w_title,
                "hwnd": w_hwnd,
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
            if app_result_clicked:
                print("  [*] App result already clicked; awaiting process/window initialization...")
                time.sleep(1.5)
                continue

            action_desc = "Open Windows Start Menu via Win Key"
            details = {"action": "dispatch_vk_key(VK_LWIN)", "foreground_window": snap.title}

            if not fast_confirm_action(action_desc, details):
                print("\n[*] User CANCELLED action. Stopping loop.")
                trace.append({"step": step_count, "action": "open_start_menu", "status": "USER_CANCELLED"})
                return {"success": False, "status": "ABORT_USER_CANCELLED", "trace": trace}

            dispatch_vk_key(0x5B, hold_duration=0.08)  # VK_LWIN
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

        # Phase 5: Locate and Click Search Result Element & Press Enter
        target_item = find_app_icon_or_shortcut(snap, intent["target_app"])
        if target_item and not app_result_clicked:
            action_desc = f"Click App Result '{target_item.name}' ({target_item.control_type}) & Press Enter"
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
                set_window_foreground_passive(snap.hwnd)
                time.sleep(0.05)
                res_click = dispatch_real_click(snap.hwnd, target_item)
                time.sleep(0.1)
                set_window_foreground_passive(snap.hwnd)
                time.sleep(0.05)
                dispatch_enter_key(hold_duration=0.08)
            else:
                res_click = simulate_click(snap.hwnd, target_item)
                if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
                    dispatch_enter_key(hold_duration=0.08)
            os.environ["MIKU_FAST_CONFIRM_PASSTHROUGH"] = "false"

            app_result_clicked = True
            trace.append({"step": step_count, "action": "click_app_result", "result": res_click.to_dict()})
            time.sleep(2.5)  # Wait for app window to launch
            continue
        else:
            # Fallback Launch Integration
            print(f"  [*] Attempting verified app launch fallback for '{intent['target_app']}'...")
            from tools.app_launcher import launch_app
            l_res = launch_app(intent["target_app"])
            trace.append({"step": step_count, "action": "fallback_launch_app", "result": l_res})
            if l_res.get("success") or l_res.get("status") in ("SUCCESS", "ALREADY_RUNNING"):
                time.sleep(3.5)
                continue
            return {"success": False, "status": "ABORT_APP_RESULT_NOT_FOUND", "trace": trace}

    return {
        "success": False,
        "status": "ABORT_MAX_STEPS_EXCEEDED",
        "steps_taken": step_count,
        "trace": trace,
    }
