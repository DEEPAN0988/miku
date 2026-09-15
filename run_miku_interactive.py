"""
run_miku_interactive.py

Dedicated CLI launcher for MIKU Autonomous Desktop Interactive Loop Controller.

Usage:
  python run_miku_interactive.py "open Notepad"
  python run_miku_interactive.py "search for miku in Antigravity IDE"
"""

import sys
import os

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
