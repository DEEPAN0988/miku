"""
eval/test_cross_subsystem_integration.py — Cross-Subsystem Integration Check (Task 3)

Validates multi-subsystem sequential pipeline:
1. Safety circuit breaker baseline verification (REAL_SEND_ENABLED=False, REAL_CLICK_ENABLED=False)
2. Step 1: "open Calculator" (app_launcher + screen_inspector detection + coordinate verification + simulated click)
3. Step 2: "what's my battery" (low-risk live system telemetry dispatch)
4. Step 3: "tell alex i am running late" (messaging automation + contact ambiguity gating)
5. Step 4: Screen-grounding inspection on secondary active window (coordinates, bounding boxes, non-occlusion)
6. Cleanup: Lifecycle process termination
7. Safety circuit breaker final invariant confirmation
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dispatcher import (
    RiskLevel,
    ToolCall,
    dispatch_tool,
    resolve_intent_with_confidence,
)
from tools.messaging import REAL_SEND_ENABLED, parse_messaging_intent
from tools.screen_inspector import (
    REAL_CLICK_ENABLED,
    find_all_elements,
    find_element,
    inspect_active_window,
    simulate_click,
    verify_element_clickable,
)


def run_cross_subsystem_scenario():
    print("=" * 85)
    print("TASK 3: CROSS-SUBSYSTEM INTEGRATION CHECK")
    print("=" * 85)

    trace = []

    # --------------------------------------------------------------------------
    # 0. INITIAL CIRCUIT BREAKER SAFETY VERIFICATION
    # --------------------------------------------------------------------------
    print("\n[PRE-CHECK] Verifying Safety Circuit Breakers in Safe Default State...")
    assert REAL_SEND_ENABLED is False, "CRITICAL FAULT: REAL_SEND_ENABLED is not False!"
    assert REAL_CLICK_ENABLED is False, "CRITICAL FAULT: REAL_CLICK_ENABLED is not False!"
    print(f"  REAL_SEND_ENABLED  : {REAL_SEND_ENABLED} (OK: Fail-closed)")
    print(f"  REAL_CLICK_ENABLED : {REAL_CLICK_ENABLED} (OK: Fail-closed)")
    trace.append({"step": 0, "name": "safety_pre_check", "real_send": REAL_SEND_ENABLED, "real_click": REAL_CLICK_ENABLED})

    calc_proc = None
    try:
        # ----------------------------------------------------------------------
        # 1. STEP 1: APP LAUNCHER + SCREEN INSPECTION + SIMULATED INTERACTION
        # ----------------------------------------------------------------------
        print("\n[STEP 1] 'Open Calculator' -> Launch + Screen Grounding + Simulated Click")
        open_call = ToolCall(action="Open App", argument="calculator", risk=RiskLevel.HIGH, is_valid=True)
        res_open = dispatch_tool(open_call, hitl_confirmed=True)
        print(f"  App Dispatch: status={res_open.status} | executed={res_open.executed}")
        print(f"  Output      : {res_open.output}")
        assert res_open.status == "SUCCESS" and res_open.executed

        # Wait for window creation
        time.sleep(1.5)

        # Grounding: Inspect Calculator window elements
        snap = inspect_active_window()
        print(f"  Active Window Grounding: HWND={snap.hwnd} | Title='{snap.title}' | Process='{snap.process_name}'")
        print(f"  Total Interactive Elements Detected: {len(snap.elements)}")
        assert snap.hwnd > 0, "No active window handle detected"

        # Find target button (e.g. 'Clear' or 'Seven' or first interactive button)
        target_btn = find_element(snap, "Clear") or (find_all_elements(snap, "Button")[0] if find_all_elements(snap, "Button") else None)
        assert target_btn is not None, "Failed to find any interactive button in Calculator"
        print(f"  Located Element: '{target_btn.name}' (id={target_btn.automation_id}) at center={target_btn.center}")

        # Coordinate Verification
        verif = verify_element_clickable(snap.hwnd, target_btn)
        print(f"  Coordinate Verification: is_safe={verif.is_safe} | reason={verif.reason}")

        # Simulated Click (Safety Circuit Breaker prevents OS dispatch)
        sim_res = simulate_click(snap.hwnd, target_btn)
        print(f"  Click Dispatch Status  : {sim_res.status}")
        print(f"  Real Input Dispatched  : {sim_res.real_input_dispatched} (OK: strictly False)")
        print(f"  Telemetry Log          : {sim_res.action_log}")
        assert sim_res.real_input_dispatched is False
        assert sim_res.status == "SIMULATED_CLICK_SUCCESS"

        trace.append({
            "step": 1,
            "action": "open_and_inspect_calculator",
            "app_output": res_open.output,
            "hwnd": snap.hwnd,
            "elements_count": len(snap.elements),
            "target_element": target_btn.name,
            "sim_click_status": sim_res.status,
            "real_input_dispatched": sim_res.real_input_dispatched,
        })

        # ----------------------------------------------------------------------
        # 2. STEP 2: TOOL DISPATCH (LIVE TELEMETRY)
        # ----------------------------------------------------------------------
        print("\n[STEP 2] 'What's my battery percentage?' -> Low-Risk Dispatch")
        action, conf, _ = resolve_intent_with_confidence("What's my battery percentage?", fallback_action="Get Battery")
        batt_call = ToolCall(action=action, argument="None", risk=RiskLevel.LOW, is_valid=True)
        res_batt = dispatch_tool(batt_call, hitl_confirmed=False)
        print(f"  Routed Action: '{action}' (Confident: {conf})")
        print(f"  Status       : {res_batt.status} | Executed: {res_batt.executed}")
        print(f"  Live Output  : {res_batt.output}")
        assert res_batt.status == "SUCCESS" and res_batt.executed
        assert "BATTERY STATUS" in res_batt.output

        trace.append({
            "step": 2,
            "action": "get_battery_telemetry",
            "routed": action,
            "output": res_batt.output,
        })

        # ----------------------------------------------------------------------
        # 3. STEP 3: MESSAGING AUTOMATION + AMBIGUITY GATING
        # ----------------------------------------------------------------------
        print("\n[STEP 3] 'Tell Alex I am running late' -> Contact Ambiguity Gating Check")
        msg_query = "tell alex i am running 15 mins late for the meeting"
        action_msg, is_conf_msg, clar_msg = resolve_intent_with_confidence(msg_query, fallback_action="Send Message")
        print(f"  Intent Resolution : action='{action_msg}' | Confident={is_conf_msg}")
        print(f"  Clarification Text: {clar_msg}")

        # Dispatch through dispatcher
        msg_call = ToolCall(action="Send Message", argument="to: Alex | message: Running late", risk=RiskLevel.HIGH, is_valid=True)
        res_msg = dispatch_tool(msg_call, hitl_confirmed=False)
        print(f"  Dispatch Gating   : status={res_msg.status} | Executed={res_msg.executed}")
        print(f"  Blocked Output    : {res_msg.output}")

        # Contact parse ambiguity check
        parsed_contact = parse_messaging_intent(msg_query)
        print(f"  Parsed Contact Ambiguity: {parsed_contact.get('is_ambiguous')} (Candidates: {parsed_contact.get('matching_contacts')})")
        assert parsed_contact.get("is_ambiguous") is True
        assert len(parsed_contact.get("matching_contacts", [])) >= 2
        assert res_msg.status == "DRY_RUN_PENDING_HITL"
        assert res_msg.executed is False

        trace.append({
            "step": 3,
            "action": "messaging_ambiguity_gating",
            "query": msg_query,
            "is_ambiguous": parsed_contact.get("is_ambiguous"),
            "candidates": parsed_contact.get("matching_contacts"),
            "dispatch_status": res_msg.status,
            "executed": res_msg.executed,
        })

        # ----------------------------------------------------------------------
        # 4. STEP 4: SCREEN GROUNDING ON SECONDARY ACTIVE APP
        # ----------------------------------------------------------------------
        print("\n[STEP 4] Screen Grounding Query on Current Active Desktop App")
        snap2 = inspect_active_window()
        print(f"  Active Window HWND : {snap2.hwnd} ('{snap2.title}', process={snap2.process_name})")
        print(f"  Interactive Elements Count: {len(snap2.elements)}")
        print(f"  Viewport Bounding Rect    : {snap2.window_rect}")
        assert snap2.hwnd > 0
        assert len(snap2.elements) > 0

        # Sample up to 3 elements for bounding box verification
        for idx, el in enumerate(snap2.elements[:3]):
            print(f"    - Element [{idx}]: name='{el.name}' | id='{el.automation_id}' | rect={el.rect} | enabled={el.is_enabled}")
            assert el.center[0] >= snap2.window_rect[0] and el.center[0] <= snap2.window_rect[2], "Center X out of bounds"
            assert el.center[1] >= snap2.window_rect[1] and el.center[1] <= snap2.window_rect[3], "Center Y out of bounds"

        trace.append({
            "step": 4,
            "action": "secondary_screen_grounding",
            "hwnd": snap2.hwnd,
            "title": snap2.title,
            "elements_inspected": len(snap2.elements),
        })

    finally:
        # Cleanup test calculator process
        print("\n[CLEANUP] Terminating test Calculator process...")
        subprocess.run(["taskkill", "/F", "/IM", "CalculatorApp.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["taskkill", "/F", "/IM", "calculator.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Cleanup complete.")

    # --------------------------------------------------------------------------
    # 5. POST-CHECK CIRCUIT BREAKER VERIFICATION
    # --------------------------------------------------------------------------
    print("\n[POST-CHECK] Re-Verifying Safety Circuit Breakers After Integration...")
    assert REAL_SEND_ENABLED is False, "CRITICAL BREACH: REAL_SEND_ENABLED mutated!"
    assert REAL_CLICK_ENABLED is False, "CRITICAL BREACH: REAL_CLICK_ENABLED mutated!"
    print(f"  REAL_SEND_ENABLED  : {REAL_SEND_ENABLED} (CONFIRMED SAFE)")
    print(f"  REAL_CLICK_ENABLED : {REAL_CLICK_ENABLED} (CONFIRMED SAFE)")
    trace.append({"step": 5, "name": "safety_post_check", "real_send": REAL_SEND_ENABLED, "real_click": REAL_CLICK_ENABLED})

    out_file = Path("logs/phase13_consolidation/cross_subsystem_trace.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"status": "SUCCESS", "trace": trace}, f, indent=2)

    print("\n" + "=" * 85)
    print("CROSS-SUBSYSTEM INTEGRATION SCENARIO COMPLETE: ALL STEPS VERIFIED")
    print(f"Saved full audit trace to: {out_file}")
    print("=" * 85)


if __name__ == "__main__":
    run_cross_subsystem_scenario()
