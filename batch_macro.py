"""
batch_macro.py — Automated Sequential Batch Execution Pipeline

Bypasses the interactive terminal loop to execute a hardcoded list of command strings
sequentially through DeterministicCLI and AgentLoop. Includes failsafe step validation.
"""

import asyncio
import os
import sys

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


async def run_setup_macro():
    print("[INFO] Starting automated batch script...")

    # 1. Initialize the deterministic pipeline
    cli = DeterministicCLI()
    agent = AgentLoop(
        inspector=TreeInspector(),
        gate=FastConfirm(config=default_config),
        executor=ActionDispatch(),
        config=default_config,
    )

    # 2. Define the exact command sequence
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
        print(f"\n[INFO] Parsing command: {cmd}")

        # Route to OS Controller
        sys_match = cli.sys_pattern.match(cmd)
        if sys_match:
            cli.execute_sys_command(sys_match.group(1).lower(), sys_match.group(2))
            continue

        # Route to UI Accessibility Tree
        task = cli.parse_ui_command(cmd)
        if task:
            result = await agent.run_task(task)
            # Failsafe: Halt the entire script if any single step fails
            if result.get("status") != "success":
                print("[ERROR] Pipeline step failed. Halting batch execution.")
                break
        else:
            print(f"[ERROR] Invalid syntax detected: {cmd}")
            break


if __name__ == "__main__":
    asyncio.run(run_setup_macro())
