"""
run_miku_interactive.py

Dedicated CLI launcher for MIKU Autonomous Desktop Interactive Loop Controller.

Usage:
  python run_miku_interactive.py "open Notepad"
  python run_miku_interactive.py "search for miku in Antigravity IDE"
"""

import sys
import os

import ctypes

def check_and_elevate() -> bool:
    """Checks if running as Admin; if not, requests elevation via ShellExecuteW 'runas'."""
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("[*] Requesting Administrator Elevation (runas) for full UAC mouse control...", flush=True)
            script_path = os.path.abspath(sys.argv[0])
            args = " ".join([f'"{a}"' for a in sys.argv[1:]])
            params = f'"{script_path}" {args}'.strip()
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, params, None, 1
            )
            if ret > 32:
                print("[+] Self-elevation request issued successfully.", flush=True)
                sys.exit(0)
            else:
                print(f"[!] User declined elevation prompt (exit code {ret}). Continuing non-elevated...", flush=True)
                return False
        else:
            print("[+] Process is running with Administrator privileges!", flush=True)
            return True
    except Exception as exc:
        print(f"[!] Auto-elevation check failed: {exc}", flush=True)
        return False

# Set environment flags for 100% continuous autonomous execution (zero Y/Enter prompting)
os.environ["MIKU_LIVE_EXECUTION"] = "true"
os.environ["MIKU_AUTONOMOUS_MODE"] = "true"
os.environ["MIKU_AUTO_APPROVE"] = "true"

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tools.screen_inspector as si
import tools.typing_automation as ta
from tools.interactive_loop import run_autonomous_task_loop

# Enable real input dispatch circuit breakers
si.REAL_CLICK_ENABLED = True
ta.REAL_TYPE_ENABLED = True

def main():
    check_and_elevate()

    if len(sys.argv) > 1:
        task_goal = " ".join(sys.argv[1:])
    else:
        task_goal = "open Notepad"

    print("=" * 80)
    print("MIKU AUTONOMOUS DESKTOP INTERACTIVE LOOP CONTROLLER (FULL AUTONOMY)")
    print(f"Goal            : '{task_goal}'")
    print(f"Circuit Breakers: REAL_CLICK_ENABLED={si.REAL_CLICK_ENABLED}, REAL_TYPE_ENABLED={ta.REAL_TYPE_ENABLED}")
    print("=" * 80)

    result = run_autonomous_task_loop(task_goal, max_steps=10, real_execution=True)

    print("\n" + "=" * 80)
    print("EXECUTION OUTCOME SUMMARY:")
    print(f"Status      : {result.get('status')}")
    print(f"Success     : {result.get('success')}")
    print(f"Steps Taken : {result.get('steps_taken')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
