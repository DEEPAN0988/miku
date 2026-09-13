"""
eval/poll_live_verifier.py — Interactive Live Foreground Poller

Polls the interactive foreground window for up to timeout_sec.
As soon as the target application or a window change occurs, it captures
the live OS state and executes verify_chat_ui_target.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.ui_verifier import inspect_live_foreground_window, verify_chat_ui_target
from tools.messaging import REAL_SEND_ENABLED

def poll_and_verify(target_app: str, target_recipient: str, timeout_sec: int = 15):
    print("=" * 80)
    print(f"POLLING FOR LIVE FOREGROUND FOCUS: Target App='{target_app}', Recipient='{target_recipient}'")
    print(f"Timeout: {timeout_sec} seconds. Please click / focus the target window now!")
    print(f"REAL_SEND_ENABLED: {REAL_SEND_ENABLED} (Hardcoded safety lock - NO SEND CAN OCCUR)")
    print("=" * 80, flush=True)

    start_time = time.time()
    last_hwnd = None

    while (time.time() - start_time) < timeout_sec:
        raw = inspect_live_foreground_window()
        proc = raw.get("process_name", "").lower()
        hwnd = raw.get("hwnd", 0)

        # If target app is focused, capture immediately!
        if any(app in proc for app in [target_app, "whatsapp"]):
            print(f"\n>>> TARGET WINDOW DETECTED IN FOREGROUND! (Elapsed: {time.time() - start_time:.1f}s) <<<")
            _print_verdict(raw, target_app, target_recipient)
            return True

        if hwnd != last_hwnd:
            print(f"  [Focus shifted] HWND: {hwnd} | Process: '{proc}' | Title: '{raw.get('title')}'", flush=True)
            last_hwnd = hwnd

        time.sleep(0.5)

    print(f"\n[TIMEOUT after {timeout_sec}s] No target window detected. Capturing current foreground state:")
    raw = inspect_live_foreground_window()
    _print_verdict(raw, target_app, target_recipient)
    return False

def _print_verdict(raw_window, target_app, target_recipient):
    print("\n--- RAW OS TELEMETRY (Live Win32 / UIAutomation) ---")
    print(f"  HWND         : {raw_window.get('hwnd')} (Hex: {hex(raw_window.get('hwnd', 0))})")
    print(f"  Process Name : '{raw_window.get('process_name')}' (PID: {raw_window.get('process_id')})")
    print(f"  Window Class : '{raw_window.get('class_name')}'")
    print(f"  Window Title : '{raw_window.get('title')}'")
    print(f"  Is Window    : {raw_window.get('is_window')}")
    print(f"  Is Visible   : {raw_window.get('is_visible')}")
    child_texts = raw_window.get("child_texts", [])
    print(f"  Extracted Elements / Child Texts ({len(child_texts)} items):")
    for idx, txt in enumerate(child_texts[:20]):
        safe_txt = txt.encode("ascii", errors="replace").decode("ascii")
        print(f"    [{idx}] '{safe_txt}'")
    if len(child_texts) > 20:
        print(f"    ... and {len(child_texts) - 20} more items")

    verdict = verify_chat_ui_target(target_app=target_app, target_recipient=target_recipient, mock_window_state=raw_window)
    print("\n--- VERIFICATION VERDICT ---")
    print(f"  Verified : {verdict.get('verified')}")
    print(f"  Status   : {verdict.get('status')}")
    print(f"  Reason   : {verdict.get('reason')}")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="whatsapp")
    parser.add_argument("--recipient", default="Aravind")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    poll_and_verify(args.app, args.recipient, args.timeout)
