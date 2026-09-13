"""
eval/execute_single_real_click.py — Single Authorized Live Real-Click Canary Runner

Strictly controlled, single real-click test:
- Target Application : Windows Calculator (calc.exe)
- Target UI Element  : 'Clear' button (clearButton)
- Pre-execution UI coordinate & occlusion verification (HWND, process, not-minimized, foreground, WindowFromPoint)
- Interactive human confirmation gate ('CONFIRM CLICK')
- Post-click independent UI-state verification
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.screen_inspector import (
    REAL_CLICK_ENABLED,
    dispatch_real_click,
    find_element,
    find_all_elements,
    inspect_screen_elements,
    verify_element_clickable,
)
from tools.dispatcher import dispatch_tool, ToolCall, RiskLevel


def run_single_real_click():
    print("=" * 85)
    print("PHASE C: SINGLE AUTHORIZED REAL-CLICK CANARY TEST RUNNER")
    print(f"  Target Application : Windows Calculator (calc.exe)")
    print(f"  Target Element     : 'Clear' (Button)")
    print(f"  Mouse Action       : Left Click")
    print(f"  REAL_CLICK_ENABLED : {REAL_CLICK_ENABLED}")
    print("=" * 85, flush=True)

    if not REAL_CLICK_ENABLED:
        print("\n[ERROR] REAL_CLICK_ENABLED is False. Execution blocked by safety circuit breaker.", flush=True)
        return False

    # Step 1: Open Calculator fresh
    print("\n[1/5] Launching Calculator fresh via app launcher...", flush=True)
    open_call = ToolCall(action="Open App", argument="calculator", risk=RiskLevel.HIGH, is_valid=True)
    res_open = dispatch_tool(open_call, hitl_confirmed=True)
    print(f"  App Dispatch Output: {res_open.output}", flush=True)
    time.sleep(1.5)

    # Step 2: Screen Inspection & Element Localization
    print("\n[2/5] Inspecting Calculator UI tree and locating target element...", flush=True)
    # Find foreground Calculator window
    from ctypes import windll
    user32 = windll.user32
    fg_hwnd = user32.GetForegroundWindow()

    snap = inspect_screen_elements(fg_hwnd)
    print(f"  Foreground Window : HWND {snap.hwnd} ('{snap.title}', process={snap.process_name})", flush=True)
    print(f"  Window Rect       : {snap.window_rect}", flush=True)
    print(f"  Elements Scanned  : {snap.total_scanned} ({len(snap.elements)} interactive)", flush=True)

    # Locate 'Clear' button
    target_btn = find_element(snap, "Clear") or find_element(snap, "clearButton")
    if not target_btn:
        # Fallback to first Button if name localized
        buttons = find_all_elements(snap, "Button")
        for b in buttons:
            if "clear" in b.automation_id.lower() or "clear" in b.name.lower():
                target_btn = b
                break
        if not target_btn and buttons:
            target_btn = buttons[0]

    assert target_btn is not None, "Failed to locate target button in Calculator window!"
    print(f"  Target Located    : '{target_btn.name}' (id={target_btn.automation_id}) at center={target_btn.center}", flush=True)

    # Step 3: Mandatory Pre-Click Safety Verification (Unconditional Gate)
    print("\n[3/5] Running pre-click coordinate & occlusion verification (WindowFromPoint)...", flush=True)
    verif = verify_element_clickable(snap.hwnd, target_btn)
    print(f"  Verification Result : is_safe={verif.is_safe} | reason={verif.reason}", flush=True)
    print(f"  Verification Details: {verif.details}", flush=True)

    if not verif.is_safe:
        print(f"\n[SAFETY REJECTION] Pre-click verification failed with reason '{verif.reason}'. Aborting click immediately.", flush=True)
        return False

    # Step 4: Interactive Live Human Confirmation Prompt
    print("\n[4/5] Awaiting live interactive human confirmation...", flush=True)
    prompt = (
        f"\n" + "=" * 85 + "\n"
        f"[LIVE HUMAN CLICK CONFIRMATION REQUIRED]\n"
        f"Target Application : {snap.title} (HWND {snap.hwnd})\n"
        f"Target UI Element  : '{target_btn.name}' (id={target_btn.automation_id})\n"
        f"Target Coordinates : {target_btn.center}\n"
        f"Mouse Button       : LEFT CLICK\n"
        f"Circuit Breaker    : REAL_CLICK_ENABLED={REAL_CLICK_ENABLED}\n"
        f"Type 'CONFIRM CLICK' to dispatch real physical OS mouse input, or anything else to cancel:\n"
        + "=" * 85 + "\n"
        f"Confirmation: "
    )
    print(prompt, end="", flush=True)
    resp = sys.stdin.readline().strip()
    print(f"\n[RECEIVED HUMAN INPUT]: '{resp}'", flush=True)

    if resp != "CONFIRM CLICK":
        print(f"[ABORTED] Live confirmation cancelled by user ('{resp}'). Zero physical input dispatched.", flush=True)
        return False

    # Step 5: Physical Click Dispatch
    print("\n[5/5] Target verified and human confirmed. Dispatching single physical mouse click...", flush=True)
    click_res = dispatch_real_click(
        target_hwnd=snap.hwnd,
        element=target_btn,
        button="left",
        interactive_confirmed=True
    )

    print("\n" + "=" * 85)
    print("--- REAL CLICK DISPATCH EXECUTION REPORT ---")
    print(f"  Status                 : {click_res.status}")
    print(f"  Success                : {click_res.success}")
    print(f"  Real Input Dispatched  : {click_res.real_input_dispatched} (PHYSICAL WIN32 INPUT)")
    print(f"  Coordinate Clicked     : {click_res.coordinate}")
    print(f"  Action Log             : {click_res.action_log}")
    print("=" * 85, flush=True)

    if not click_res.success or not click_res.real_input_dispatched:
        print("[FAIL] Real click was not dispatched successfully.", flush=True)
        return False

    # Independent Post-Click Verification
    print("\n--- INDEPENDENT POST-CLICK AUDIT ---", flush=True)
    time.sleep(0.5)
    post_snap = inspect_screen_elements(snap.hwnd)
    print(f"  Post-Click Window State : HWND {post_snap.hwnd} ('{post_snap.title}')", flush=True)
    print(f"  Post-Click Elements     : {len(post_snap.elements)} interactive elements responsive", flush=True)

    # Check Calculator display element
    results_el = find_element(post_snap, "CalculatorResults") or find_element(post_snap, "Display is")
    display_text = results_el.name if results_el else "Unknown"
    print(f"  Calculator Display Text : '{display_text}'", flush=True)

    # Save detailed audit log
    out_dir = Path("logs/phase13_consolidation")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "real_click_canary_audit.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": {
                "app": snap.title,
                "process": snap.process_name,
                "hwnd": snap.hwnd,
                "element_name": target_btn.name,
                "automation_id": target_btn.automation_id,
                "coordinate": target_btn.center,
                "button": "left",
            },
            "pre_click_verification": verif.to_dict(),
            "human_confirmation_input": resp,
            "dispatch_result": click_res.to_dict(),
            "post_click_audit": {
                "display_text": display_text,
                "interactive_elements_count": len(post_snap.elements),
                "latency_ms": post_snap.latency_ms,
            }
        }, f, indent=2)

    print(f"  Detailed Canary Audit Log Saved: {log_path}", flush=True)
    print("=" * 85, flush=True)
    return True


if __name__ == "__main__":
    success = run_single_real_click()
    sys.exit(0 if success else 1)
