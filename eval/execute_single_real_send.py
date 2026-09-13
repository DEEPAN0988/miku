"""
eval/execute_single_real_send.py — Single Authorized Live Real-Send Test Runner

Strictly controlled, single real-send test:
- Recipient: Aravind
- Exact text: "Testing Miku - please ignore!"
- Pre-execution UI target verification (HWND, process, in-app modal rejection, contact in chat header)
- Interactive human confirmation gate ('CONFIRM SEND')
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.messaging import REAL_SEND_ENABLED, send_whatsapp_message
from tools.ui_verifier import inspect_live_foreground_window, verify_chat_ui_target
from tools.app_launcher import focus_app

def run_single_real_send():
    recipient = "Aravind"
    message = "Testing Miku - please ignore!"

    print("=" * 80)
    print("SINGLE AUTHORIZED REAL TEST SEND RUNNER")
    print(f"  Target Recipient : '{recipient}'")
    print(f"  Exact Message    : \"{message}\"")
    print(f"  REAL_SEND_ENABLED: {REAL_SEND_ENABLED}")
    print("=" * 80, flush=True)

    if not REAL_SEND_ENABLED:
        print("\n[ERROR] REAL_SEND_ENABLED is False. Execution blocked by safety circuit breaker.")
        return False

    # 1. Interactive Live Human Confirmation Prompt
    prompt = (
        f"\n" + "=" * 80 + "\n"
        f"[LIVE HUMAN CONFIRMATION REQUIRED]\n"
        f"Target Recipient: {recipient}\n"
        f"Exact Final Text: \"{message}\"\n"
        f"Type 'CONFIRM SEND' to transmit this message to a real person, or anything else to cancel:\n"
        + "=" * 80 + "\n"
        f"Confirmation: "
    )
    print(prompt, end="", flush=True)
    resp = sys.stdin.readline().strip()
    print(f"\n[RECEIVED HUMAN INPUT]: '{resp}'", flush=True)

    if resp != "CONFIRM SEND":
        print("[ABORTED] Live confirmation cancelled by user. Zero messages sent.", flush=True)
        return False

    # 2. Focus and Target UI Verification
    print("\nEnsuring WhatsApp is in foreground and verifying target chat...", flush=True)
    focus_app("whatsapp")
    time.sleep(1.0)

    raw_window = inspect_live_foreground_window()
    print("--- LIVE FOREGROUND WINDOW TELEMETRY ---")
    print(f"  HWND         : {raw_window.get('hwnd')}")
    print(f"  Process Name : '{raw_window.get('process_name')}'")
    print(f"  Window Title : '{raw_window.get('title')}'")
    print(f"  Class Name   : '{raw_window.get('class_name')}'")
    print(f"  Elements ({len(raw_window.get('child_texts', []))} items)")

    ui_verif = verify_chat_ui_target("whatsapp", recipient)
    print("\n--- UI TARGET VERIFICATION VERDICT ---")
    print(f"  Verified : {ui_verif.get('verified')}")
    print(f"  Status   : {ui_verif.get('status')}")
    print(f"  Reason   : {ui_verif.get('reason')}")

    if not ui_verif.get("verified"):
        print(f"\n[SAFETY ABORT] Verification failed: {ui_verif.get('status')}. Aborting real send.", flush=True)
        return False

    # 3. Real Dispatch
    print("\nTarget verified. Dispatching single live message...", flush=True)
    res = send_whatsapp_message(
        recipient=recipient,
        text=message,
        dry_run=False,
        interactive_confirmed=True
    )

    print("\n" + "=" * 80)
    print("--- DISPATCH EXECUTION REPORT ---")
    print(f"  Status   : {res.get('status')}")
    print(f"  Executed : {res.get('executed')}")
    print(f"  Dry Run  : {res.get('dry_run')}")
    print(f"  Output   : {res.get('output')}")
    if res.get("error"):
        print(f"  Error    : {res.get('error')}")
    print("=" * 80, flush=True)
    return res.get("executed", False)

if __name__ == "__main__":
    run_single_real_send()
