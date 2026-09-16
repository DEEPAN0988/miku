"""
run_miku_master.py — Unified Master Autonomous Controller for MIKU

Provides a single master entrypoint for:
1. Subsystem Diagnostics & Hardware Telemetry (--diag)
2. Headless Automated Batch Scripting (--batch <file|cmds>)
3. Real-Time Autonomous Interactive Task Loop (--task "<goal>")
4. System & Process Kernel Direct Manipulation

100% Local, Offline, Zero External API Dependencies.
"""

import argparse
import asyncio
import ctypes
import os
import sys
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def setup_environment():
    """Environment flags for continuous autonomous execution."""
    os.environ["MIKU_LIVE_EXECUTION"] = "true"
    os.environ["MIKU_AUTONOMOUS_MODE"] = "true"
    os.environ["MIKU_AUTO_APPROVE"] = "true"

from batch_macro import run_batch_macro
from cli_macro_engine import DeterministicCLI
from os_controller import OSController
from run_system_diagnostics import run_diagnostics
from tools.interactive_loop import run_autonomous_task_loop


def check_and_elevate() -> bool:
    """Checks for Administrator privileges and requests elevation if necessary."""
    try:
        if os.environ.get("MIKU_NO_ELEVATE") == "true":
            print("[*] MIKU_NO_ELEVATE active. Continuing non-elevated.", flush=True)
            return False

        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("[*] Requesting Administrator Elevation (runas)...", flush=True)
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
                print("[!] Continued non-elevated.", flush=True)
                return False
        else:
            print("[+] Administrator privileges confirmed.", flush=True)
            return True
    except Exception as exc:
        print(f"[!] Auto-elevation check failed: {exc}", flush=True)
        return False


def run_master_controller(
    diag_mode: bool = False,
    task_goal: Optional[str] = None,
    batch_cmds: Optional[List[str]] = None,
):
    setup_environment()
    print("=" * 80)
    print(" MIKU MASTER AUTONOMOUS CONTROLLER v0.2")
    print(" Zero-Cloud API | Direct Win32 Kernel & UIA Automation | Fully Local")
    print("=" * 80)

    # 1. Run Diagnostics if requested
    if diag_mode:
        print("\n[MASTER] Running System Diagnostics...")
        success = run_diagnostics()
        return 0 if success else 1

    # 2. Run Single Task Loop if specified
    if task_goal:
        print(f"\n[MASTER] Executing Autonomous Task: '{task_goal}'")
        result = run_autonomous_task_loop(task_goal, max_steps=10, real_execution=True)
        print("\n" + "=" * 80)
        print("TASK OUTCOME SUMMARY:")
        print(f"Status  : {result.get('status')}")
        print(f"Success : {result.get('success')}")
        print(f"Steps   : {result.get('steps_taken')}")
        print("=" * 80)
        return 0 if result.get("success") else 1

    # 3. Run Batch Script if provided
    if batch_cmds:
        print("\n[MASTER] Running Batch Macro Execution...")
        asyncio.run(run_batch_macro(batch_cmds))
        return 0

    # 4. Interactive CLI Fallback
    print("\n[MASTER] Entering Interactive Command Shell...")
    print("Type 'sys info', 'sys ps', 'open notepad', or 'exit' to quit.\n")
    cli = DeterministicCLI()
    os_ctrl = OSController()

    while True:
        try:
            cmd = input("MIKU-MASTER> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                print("[MASTER] Shutting down controller.")
                break

            sys_match = cli.sys_pattern.match(cmd)
            if sys_match:
                op = sys_match.group(1).lower()
                arg = sys_match.group(2)
                res = cli.execute_sys_command(op, arg)
                print(f"[RESULT] {res}")
            else:
                task = cli.parse_ui_command(cmd)
                if task:
                    print(f"[PARSED UI TASK] Action: {task.get('action')} | Target: {task.get('target')}")
                    asyncio.run(run_batch_macro([cmd]))
                else:
                    print(f"[!] Unrecognized command format: '{cmd}'")
        except (KeyboardInterrupt, EOFError):
            print("\n[MASTER] Exiting shell.")
            break
    return 0


def main():
    parser = argparse.ArgumentParser(description="MIKU Master Autonomous Controller")
    parser.add_argument("--diag", action="store_true", help="Run system diagnostics")
    parser.add_argument("--task", type=str, help="Autonomous task goal string")
    parser.add_argument("--batch", type=str, nargs="+", help="Batch macro commands or script file path")
    args = parser.parse_args()

    check_and_elevate()

    batch_cmds = None
    if args.batch:
        if len(args.batch) == 1 and os.path.exists(args.batch[0]):
            with open(args.batch[0], "r", encoding="utf-8") as f:
                batch_cmds = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        else:
            batch_cmds = args.batch

    sys.exit(run_master_controller(diag_mode=args.diag, task_goal=args.task, batch_cmds=batch_cmds))


if __name__ == "__main__":
    main()
