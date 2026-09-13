"""
eval/test_interactive_confirmation.py — Real Interactive Confirmation Test

Runs a real terminal prompt waiting on stdin for human confirmation,
verifies the input, and passes it to send_whatsapp_message to prove
the circuit breaker (REAL_SEND_ENABLED = False) blocks execution even after confirmation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.messaging import request_live_human_confirmation, send_whatsapp_message, REAL_SEND_ENABLED

def run_interactive_gate(recipient: str, message: str, allow_piped: bool = False):
    print("=" * 80)
    print("STARTING LIVE INTERACTIVE HUMAN CONFIRMATION HARNESS")
    print(f"  Target Recipient : '{recipient}'")
    print(f"  Exact Final Text : \"{message}\"")
    print(f"  REAL_SEND_ENABLED: {REAL_SEND_ENABLED} (Safety Circuit Breaker)")
    print("=" * 80, flush=True)

    # 1. Check isatty unless allow_piped is enabled for terminal pipe testing
    is_tty = bool(sys.stdin and sys.stdin.isatty())
    print(f"Terminal isatty check: {is_tty}", flush=True)

    if not is_tty and not allow_piped:
        print("[FAIL CLOSED] Non-interactive environment detected (isatty=False). Confirmation rejected.", flush=True)
        return False

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

    confirmed = (resp == "CONFIRM SEND")
    print(f"Human Confirmation Verdict: {confirmed}", flush=True)

    # 2. Even if confirmed, invoke send_whatsapp_message with dry_run=False
    print("\nAttempting non-dry-run dispatch to verify circuit breaker enforcement...", flush=True)
    res = send_whatsapp_message(
        recipient=recipient,
        text=message,
        dry_run=False,
        interactive_confirmed=confirmed
    )

    print("\n--- DISPATCH EXECUTION RESULT ---")
    print(f"  Status   : {res.get('status')}")
    print(f"  Executed : {res.get('executed')}")
    print(f"  Dry Run  : {res.get('dry_run')}")
    print(f"  Error    : {res.get('error')}")
    print(f"  Output   : {res.get('output')}")
    print("=" * 80, flush=True)
    return res

if __name__ == "__main__":
    allow_piped_flag = ("--allow-piped" in sys.argv)
    run_interactive_gate("Aravind", "Hi.", allow_piped=allow_piped_flag)
