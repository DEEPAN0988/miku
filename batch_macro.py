"""
batch_macro.py — Automated Sequential Batch Execution Pipeline

Executes a sequence of command strings sequentially through DeterministicCLI and AgentLoop.
Supports reading macro files (--file <path>), inline CLI args, or default workspace setup commands.
Includes failsafe step validation.
"""

import asyncio
import os
import sys
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent_loop import AgentLoop
from cli_macro_engine import DeterministicCLI
from config import default_config
from human_gate import FastConfirm
from safe_executor import ActionDispatch
from ui_inspector import TreeInspector


async def run_batch_macro(commands: Optional[List[str]] = None):
    os.environ["MIKU_AUTO_APPROVE"] = "true"
    os.environ["MIKU_AUTONOMOUS_MODE"] = "true"
    print("[INFO] Starting automated batch script...")

    # 1. Initialize the deterministic pipeline
    cli = DeterministicCLI()
    agent = AgentLoop(
        inspector=TreeInspector(),
        gate=FastConfirm(config=default_config),
        executor=ActionDispatch(),
        config=default_config,
    )

    # 2. Define or load command sequence
    if not commands:
        commands = [
            "sys boot",
            "click 'Terminal'",
            "type 'npm install framer-motion\\n' into 'Terminal'",
            "click 'Explorer'",
            "type 'tailwind.config.ts\\n' into 'Search'",
            "click 'Text Editor'",
            "type '// Update Tailwind config\\n' into 'Text Editor'",
            "sys lock",
        ]

    # 3. Execute the batch safely
    for cmd in commands:
        cmd_clean = cmd.strip()
        if not cmd_clean or cmd_clean.startswith("#"):
            continue

        print(f"\n[INFO] Parsing command: {cmd_clean}")

        # Route to OS Controller
        sys_match = cli.sys_pattern.match(cmd_clean)
        if sys_match:
            cli.execute_sys_command(sys_match.group(1).lower(), sys_match.group(2))
            continue

        # Route to UI Accessibility Tree
        task = cli.parse_ui_command(cmd_clean)
        if task:
            result = await agent.run_task(task)
            # Failsafe: Halt the entire script if any single step fails
            if result.get("status") != "success":
                print("[ERROR] Pipeline step failed. Halting batch execution.")
                break
        else:
            print(f"[ERROR] Invalid syntax detected: {cmd_clean}")
            break


# Backward compatibility alias
run_setup_macro = run_batch_macro

if __name__ == "__main__":
    cmds = []
    args = sys.argv[1:]
    if "--file" in args:
        idx = args.index("--file")
        if idx + 1 < len(args):
            filepath = args[idx + 1]
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    cmds = [line.strip() for line in f if line.strip()]
            else:
                print(f"[ERROR] Macro script file not found: {filepath}")
                sys.exit(1)
    elif args:
        cmds = args

    asyncio.run(run_batch_macro(cmds if cmds else None))
