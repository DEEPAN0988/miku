"""
eval/live_window_verifier_test.py — Live Foreground Window Verification & Detection Audit

Tests verify_chat_ui_target against REAL live Windows state (Task 1 & Task 2).
STRICT INVARIANT: No real keystrokes or sends occur (REAL_SEND_ENABLED = False).
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.ui_verifier import inspect_live_foreground_window, verify_chat_ui_target
from tools.messaging import REAL_SEND_ENABLED

def run_live_test(scenario_id: str, target_app: str, target_recipient: str, delay_sec: int = 0):
    print("=" * 80)
    print(f"LIVE TEST SCENARIO: [{scenario_id}]")
    print(f"  Target Application : '{target_app}'")
    print(f"  Target Recipient   : '{target_recipient}'")
    print(f"  REAL_SEND_ENABLED  : {REAL_SEND_ENABLED} (Hardcoded safety lock)")
    print("=" * 80)

    if delay_sec > 0:
        print(f"Waiting {delay_sec} seconds for user to set foreground window focus...")
        for i in range(delay_sec, 0, -1):
            print(f"  T-{i}s...", flush=True)
            time.sleep(1)
        print("Capturing live foreground window now!\n")

    # 1. Raw OS Inspection
    raw_window = inspect_live_foreground_window()
    print("--- RAW OS TELEMETRY (Live Win32 / UIAutomation) ---")
    print(f"  HWND         : {raw_window.get('hwnd')} (Hex: {hex(raw_window.get('hwnd', 0))})")
    print(f"  Process Name : '{raw_window.get('process_name')}' (PID: {raw_window.get('process_id')})")
    print(f"  Window Class : '{raw_window.get('class_name')}'")
    print(f"  Window Title : '{raw_window.get('title')}'")
    print(f"  Is Window    : {raw_window.get('is_window')}")
    print(f"  Is Visible   : {raw_window.get('is_visible')}")
    child_texts = raw_window.get('child_texts', [])
    print(f"  Extracted Elements / Child Texts ({len(child_texts)} items):")
    for idx, txt in enumerate(child_texts[:15]):
        # Clean unicode characters for Windows console safety
        safe_txt = txt.encode('ascii', errors='replace').decode('ascii')
        print(f"    [{idx}] '{safe_txt}'")
    if len(child_texts) > 15:
        print(f"    ... and {len(child_texts) - 15} more items")

    # 2. Verification Verdict
    verdict = verify_chat_ui_target(target_app=target_app, target_recipient=target_recipient)
    print("\n--- VERIFICATION VERDICT ---")
    print(f"  Verified : {verdict.get('verified')}")
    print(f"  Status   : {verdict.get('status')}")
    print(f"  Reason   : {verdict.get('reason')}")
    print("=" * 80 + "\n")
    return raw_window, verdict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live UI Target Verifier")
    parser.add_argument("--scenario", default="scenario_1_wrong_app")
    parser.add_argument("--app", default="whatsapp")
    parser.add_argument("--recipient", default="Alice")
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()

    run_live_test(args.scenario, args.app, args.recipient, args.delay)
