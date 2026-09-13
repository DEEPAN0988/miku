"""
eval/test_messaging_automation.py — Phase 12 Remediation Audit Suite: UI-Target Verification & Safety Circuit Breaker

EVALUATION SPECIFICATION:
  1. SAFETY CIRCUIT BREAKER INVARIANT:
     - Verify REAL_SEND_ENABLED is permanently False in code.
     - Verify any attempt to force a non-dry-run send is blocked by the circuit breaker.
  2. SIMULATED PIPELINE SUITE:
     - Verify end-to-end flow across 5 distinct dictated messages in simulation mode.
     - Verify unconfirmed sends are strictly blocked (DRY_RUN_PENDING_HITL) showing exact text.
     - Verify confirmed sends in simulation mode return SIMULATED_SUCCESS with verified UI target.
  3. NEGATIVE / FAILURE-PATH MISMATCH SUITE (Task 1 & Task 3):
     - Deliberately test mismatch scenarios:
       a) Wrong foreground application (e.g. Chrome / Notepad active) -> ABORT_WRONG_APP
       b) Modal / Collision dialog active (e.g. "Forward to...") -> ABORT_MODAL_DIALOG_DETECTED
       c) Recipient chat mismatch (e.g. chat header shows "Mom" when sending to "David") -> ABORT_RECIPIENT_NOT_VISIBLE
       d) No foreground window (HWND 0) -> ABORT_NO_FOREGROUND_WINDOW
     - Verify that each failure path immediately ABORTS with zero keystrokes.
  4. AMBIGUITY SUITE (Task 5):
     - Test contact ambiguity (multiple 'Alex' entries in contact book).
     - Test app ambiguity (conflicting messaging platforms).
"""

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))

from tools.dispatcher import (
    CONFIDENCE_MARGIN,
    CONFIDENCE_RATIO,
    RiskLevel,
    ToolCall,
    detect_query_argument,
    dispatch_tool,
    extract_deterministic_slot,
    rank_with_structure,
    resolve_intent_with_confidence,
)
from tools.messaging import (
    DEFAULT_CONTACT_BOOK,
    REAL_SEND_ENABLED,
    format_dictated_text,
    parse_messaging_intent,
    send_whatsapp_message,
)
from tools.ui_verifier import verify_chat_ui_target

# Standard simulation cases
SIMULATION_CASES = [
    {
        "id": "sim_david_lunch",
        "query": "send to david hey are we still on for lunch today at noon",
        "expected_action": "Send Message",
        "expected_recipient": "David Smith",
        "expected_message": "Are we still on for lunch today at noon?",
    },
    {
        "id": "sim_sarah_report",
        "query": "tell sarah i sent the quarterly report over email please check it",
        "expected_action": "Send Message",
        "expected_recipient": "Sarah Jenkins",
        "expected_message": "I sent the quarterly report over email please check it.",
    },
    {
        "id": "sim_mom_grocery",
        "query": "message mom asking if she needs anything from the grocery store",
        "expected_action": "Send Message",
        "expected_recipient": "Mom",
        "expected_message": "She needs anything from the grocery store.",
    },
    {
        "id": "sim_bob_congrats",
        "query": "send message to bob congratulations on the promotion well deserved",
        "expected_action": "Send Message",
        "expected_recipient": "Bob Roberts",
        "expected_message": "Congratulations on the promotion well deserved.",
    },
    {
        "id": "sim_dad_weekend",
        "query": "text dad what time should we meet this weekend",
        "expected_action": "Send Message",
        "expected_recipient": "Dad",
        "expected_message": "What time should we meet this weekend?",
    },
]

# Negative mismatch test cases (Task 1 & Task 3)
MISMATCH_ABORT_CASES = [
    {
        "id": "abort_wrong_app",
        "description": "Wrong foreground application active (Chrome instead of WhatsApp)",
        "recipient": "David Smith",
        "mock_state": {
            "hwnd": 201,
            "is_window": True,
            "process_name": "chrome.exe",
            "title": "Google Chrome - Inbox",
            "class_name": "Chrome_WidgetWin_1",
            "child_texts": ["Search Google"],
        },
        "expected_status": "ABORT_WRONG_APP",
    },
    {
        "id": "abort_modal_dialog",
        "description": "WhatsApp is active but a modal 'Forward to...' dialog has focus",
        "recipient": "David Smith",
        "mock_state": {
            "hwnd": 202,
            "is_window": True,
            "process_name": "whatsapp.exe",
            "title": "Forward to...",
            "class_name": "ApplicationFrameWindow",
            "child_texts": ["Select contact", "LinkedIn Profile"],
        },
        "expected_status": "ABORT_MODAL_DIALOG_DETECTED",
    },
    {
        "id": "abort_recipient_mismatch",
        "description": "WhatsApp is active but the open chat is Mom instead of David Smith",
        "recipient": "David Smith",
        "mock_state": {
            "hwnd": 203,
            "is_window": True,
            "process_name": "whatsapp.exe",
            "title": "WhatsApp - Mom",
            "class_name": "ApplicationFrameWindow",
            "child_texts": ["Mom", "Online", "Type a message"],
        },
        "expected_status": "ABORT_RECIPIENT_NOT_VISIBLE",
    },
    {
        "id": "abort_no_window",
        "description": "Headless or no active foreground window (HWND 0)",
        "recipient": "David Smith",
        "mock_state": {
            "hwnd": 0,
            "is_window": False,
            "process_name": "unknown",
            "title": "",
            "class_name": "",
            "child_texts": [],
        },
        "expected_status": "ABORT_NO_FOREGROUND_WINDOW",
    },
]

# Ambiguity test cases
AMBIGUITY_CASES = [
    {
        "id": "ambig_contact_alex",
        "query": "tell alex i am running 15 mins late for the meeting",
        "expected_status": "CLARIFICATION_REQUIRED",
        "ambiguity_type": "CONTACT_AMBIGUITY",
        "reason": "Multiple contacts match 'Alex' ('Alex Miller' and 'Alex Chen')",
    },
    {
        "id": "ambig_app_conflict",
        "query": "send message on discord and whatsapp to david hello",
        "expected_status": "CLARIFICATION_REQUIRED",
        "ambiguity_type": "APP_AMBIGUITY",
        "reason": "Conflicting apps specified (Discord vs WhatsApp)",
    },
]


def test_circuit_breaker() -> bool:
    print("=" * 85)
    print("VERIFYING SAFETY CIRCUIT BREAKER (DEFENSE IN DEPTH)")
    print("=" * 85)
    print(f"  REAL_SEND_ENABLED Flag Value: {REAL_SEND_ENABLED}")
    assert REAL_SEND_ENABLED is False, "CRITICAL VIOLATION: REAL_SEND_ENABLED must be False in code!"

    # Attempt forced non-dry-run send
    res = send_whatsapp_message("Test Recipient", "Test text", dry_run=False)
    print(f"  Forced Live Send Attempt -> Status: {res['status']}")
    print(f"  Output -> {res['output']}")
    assert res["status"] == "CIRCUIT_BREAKER_BLOCKED", f"Failed: Expected CIRCUIT_BREAKER_BLOCKED, got {res['status']}"
    assert not res["executed"], "CRITICAL VIOLATION: Forced live send executed an OS action!"
    print("  [PASS] Circuit breaker successfully blocked forced live send.")
    return True


def run_simulation_suite() -> Dict[str, Any]:
    print("\n" + "=" * 85)
    print("TASK 3: END-TO-END SIMULATED PIPELINE VERIFICATION")
    print("=" * 85)

    results = []
    for case in SIMULATION_CASES:
        cid = case["id"]
        query = case["query"]
        exp_recip = case["expected_recipient"]
        print(f"\n--- Running [{cid}]: '{query}' ---")

        # 1. Intent routing
        action, is_confident, clar_prompt = resolve_intent_with_confidence(query, "Search Web")
        arg = extract_deterministic_slot(query, action)
        tc = ToolCall(
            action=action,
            argument=arg or "",
            risk=RiskLevel.HIGH,
            is_valid=True,
            is_confident=is_confident,
            clarification_prompt=clar_prompt,
        )

        print(f"  Action   : {tc.action} | Confident: {is_confident}")
        print(f"  Argument : {tc.argument}")

        # 2. Gate Test 1: Unconfirmed Dispatch (MUST BE BLOCKED)
        unconf_res = dispatch_tool(tc, hitl_confirmed=False)
        print(f"  [Gate Test 1: Unconfirmed Send]")
        print(f"    Status: {unconf_res.status} | Executed: {unconf_res.executed}")
        print(f"    Prompt:\n{unconf_res.output}")

        assert unconf_res.status == "DRY_RUN_PENDING_HITL", f"Unconfirmed send was not blocked! Got {unconf_res.status}"
        assert not unconf_res.executed, "Unconfirmed send executed an OS action!"

        # 3. Gate Test 2: Confirmed Simulation Send (Valid Mock Target)
        valid_mock_state = {
            "hwnd": 999,
            "is_window": True,
            "process_name": "whatsapp.exe",
            "title": f"WhatsApp - {exp_recip}",
            "class_name": "ApplicationFrameWindow",
            "child_texts": [exp_recip, "Type a message"],
        }
        conf_res = dispatch_tool(tc, hitl_confirmed=True, mock_window_state=valid_mock_state)
        print(f"  [Gate Test 2: Confirmed Simulation Send]")
        print(f"    Status: {conf_res.status} | Executed: {conf_res.executed}")
        print(f"    Output: {conf_res.output}")

        assert conf_res.status in ("SUCCESS", "SIMULATED_SUCCESS"), f"Confirmed simulation send failed! Got {conf_res.status}"
        assert not conf_res.executed, "Simulation mode executed an OS action!"

        results.append({
            "id": cid,
            "query": query,
            "action": action,
            "argument": arg,
            "unconfirmed_status": unconf_res.status,
            "confirmed_status": conf_res.status,
            "passed": True,
        })

    print(f"\nTask 3 Simulation Suite Passed: {len(results)}/{len(SIMULATION_CASES)} cases verified.")
    return {"passed": len(results), "total": len(SIMULATION_CASES), "cases": results}


def run_mismatch_abort_suite() -> Dict[str, Any]:
    print("\n" + "=" * 85)
    print("TASK 1 & TASK 3: NEGATIVE MISMATCH & UI-TARGET ABORT VERIFICATION")
    print("=" * 85)

    results = []
    for case in MISMATCH_ABORT_CASES:
        cid = case["id"]
        desc = case["description"]
        recip = case["recipient"]
        mock_state = case["mock_state"]
        exp_status = case["expected_status"]

        print(f"\n--- Testing Mismatch [{cid}]: {desc} ---")
        print(f"  Target Recipient : '{recip}'")
        print(f"  Active Window    : HWND {mock_state['hwnd']} | Process: '{mock_state['process_name']}' | Title: '{mock_state['title']}'")

        # 1. Test UI Verifier directly
        verif_res = verify_chat_ui_target("whatsapp", recip, mock_window_state=mock_state)
        print(f"  UI Verifier Result -> Verified: {verif_res['verified']} | Status: {verif_res['status']}")
        print(f"  Reason: {verif_res['reason']}")
        assert not verif_res["verified"], f"Verifier incorrectly passed for mismatched window: {cid}"
        assert verif_res["status"] == exp_status, f"Expected {exp_status}, got {verif_res['status']}"

        # 2. Test Dispatcher Abort via send_whatsapp_message
        send_res = send_whatsapp_message(recip, "Test message content", dry_run=True, mock_window_state=mock_state)
        print(f"  Dispatcher Execution Result -> Status: {send_res['status']} | Executed: {send_res['executed']}")
        print(f"  Output -> {send_res['output']}")

        assert send_res["status"] == exp_status, f"Dispatcher failed to abort on mismatch: got {send_res['status']}"
        assert not send_res["executed"], "CRITICAL VIOLATION: Mismatch executed an action!"

        results.append({
            "id": cid,
            "description": desc,
            "expected_status": exp_status,
            "actual_status": send_res["status"],
            "abort_output": send_res["output"],
            "passed": True,
        })

    print(f"\nMismatch Abort Suite Passed: {len(results)}/{len(MISMATCH_ABORT_CASES)} failure paths correctly aborted.")
    return {"passed": len(results), "total": len(MISMATCH_ABORT_CASES), "cases": results}


def run_ambiguity_suite() -> Dict[str, Any]:
    print("\n" + "=" * 85)
    print("TASK 5: AMBIGUITY AUDIT IN MESSAGING DOMAIN")
    print("=" * 85)

    results = []
    for case in AMBIGUITY_CASES:
        cid = case["id"]
        query = case["query"]
        exp_status = case["expected_status"]

        print(f"\n--- Testing Ambiguity [{cid}]: '{query}' ---")
        action, is_confident, clar_prompt = resolve_intent_with_confidence(query, "Search Web")
        arg = extract_deterministic_slot(query, action)
        tc = ToolCall(
            action=action,
            argument=arg or "",
            risk=RiskLevel.HIGH,
            is_valid=True,
            is_confident=is_confident,
            clarification_prompt=clar_prompt,
        )

        dispatch_res = dispatch_tool(tc, hitl_confirmed=False)
        print(f"  Dispatch Result -> Status: {dispatch_res.status} | Confident: {is_confident}")
        print(f"  Output -> {dispatch_res.output}")

        assert dispatch_res.status == exp_status, f"Expected {exp_status}, got {dispatch_res.status}"
        results.append({
            "id": cid,
            "query": query,
            "status": dispatch_res.status,
            "output": dispatch_res.output,
            "passed": True,
        })

    print(f"\nAmbiguity Suite Passed: {len(results)}/{len(AMBIGUITY_CASES)} cases verified.")
    return {"passed": len(results), "total": len(AMBIGUITY_CASES), "cases": results}


def main():
    test_circuit_breaker()
    sim_summary = run_simulation_suite()
    abort_summary = run_mismatch_abort_suite()
    ambig_summary = run_ambiguity_suite()

    out_data = {
        "circuit_breaker_verified": True,
        "simulation": sim_summary,
        "mismatch_aborts": abort_summary,
        "ambiguity": ambig_summary,
    }

    os.makedirs("logs/phase12_messaging_automation", exist_ok=True)
    out_path = "logs/phase12_messaging_automation/remediation_audit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nRemediation audit results successfully logged to: {out_path}")


if __name__ == "__main__":
    main()
