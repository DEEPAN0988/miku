"""
eval/demo_logical_task.py — Multi-Step Logical Task Execution Demonstration

Executes a non-trivial 5-step logical task pipeline:
1. Read current desktop state & orient active window.
2. Boot workspace environment (VS Code + local server).
3. Search for project configuration files in Explorer.
4. Perform system diagnostic telemetry scan.
5. Safely recycle temporary cache files to free memory.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure workspace root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_loop import AgentLoop, TaskSpec
from config import default_config
from human_gate import FastConfirm
from safe_executor import ActionDispatch
from ui_inspector import TreeInspector
from visual_highlighter import TargetHighlighter
from cli_macro_engine import DeterministicCLI


async def run_logical_task_pipeline():
    print("\n==========================================================================================")
    print(" MIKU OS v0.2 — MULTI-STEP LOGICAL TASK EXECUTION PIPELINE")
    print("==========================================================================================")

    mock_nodes = [
        {
            "name": "Desktop Screen",
            "automation_id": "root_screen",
            "control_type": "Pane",
            "bounds": (0, 0, 1920, 1080),
            "center": (960, 540),
            "text": "Windows 11 Desktop | Active IDE: VS Code | Terminal Window",
        },
        {
            "name": "VS Code",
            "automation_id": "app_vscode",
            "control_type": "Window",
            "bounds": (100, 100, 1200, 800),
            "center": (650, 500),
            "text": "c:\\miku - Visual Studio Code",
        },
        {
            "name": "Terminal",
            "automation_id": "txt_terminal",
            "control_type": "Edit",
            "bounds": (150, 600, 1150, 880),
            "center": (650, 740),
            "text": "PowerShell 7.4.1",
        },
        {
            "name": "Explorer Search",
            "automation_id": "txt_explorer_search",
            "control_type": "Edit",
            "bounds": (800, 120, 1100, 150),
            "center": (950, 135),
        },
    ]

    inspector = TreeInspector(mock_elements=mock_nodes)
    highlighter = TargetHighlighter(enabled=True, line_width=4)

    def auto_operator_gate(prompt: str) -> str:
        print("  [SAFETY GATE AUTO-APPROVE] Operator confirmed action with 'y'.")
        return "y"

    gate = FastConfirm(config=default_config, input_handler=auto_operator_gate, highlighter=highlighter)
    executor = ActionDispatch(simulation_mode=True)
    agent = AgentLoop(inspector=inspector, gate=gate, executor=executor, config=default_config)
    cli = DeterministicCLI()

    print("\n--- PHASE 1: SYSTEM OVERRIDE INSTRUCTION ROUTING ---")
    print("[LOGICAL COMMAND 1]: sys info")
    cli.execute_sys_command("info", "")

    print("\n[LOGICAL COMMAND 2]: sys boot")
    cli.execute_sys_command("boot", "")

    print("\n--- PHASE 2: UI AUTOMATION & INTENT PIPELINE ---")
    logical_tasks = [
        TaskSpec(
            action_type="read_screen_text",
            target_name="Desktop Screen",
            rationale="Scanning active desktop windows to identify open developer tools and active processes.",
        ),
        TaskSpec(
            action_type="click",
            target_name="Terminal",
            rationale="Focusing integrated terminal pane inside VS Code.",
        ),
        TaskSpec(
            action_type="type",
            target_name="Terminal",
            payload="python -m unittest discover -s eval\n",
            rationale="Executing test suite verification command in terminal.",
        ),
        TaskSpec(
            action_type="click",
            target_name="Explorer Search",
            rationale="Focusing file search bar to locate project configuration assets.",
        ),
        TaskSpec(
            action_type="type",
            target_name="Explorer Search",
            payload="configs/phase5b_diversified_sft.yaml\n",
            rationale="Searching for SFT pipeline configuration.",
        ),
    ]

    try:
        results = await agent.run(logical_tasks)
        print("\n==========================================================================================")
        print(" LOGICAL TASK PIPELINE COMPLETED SUCCESSFULLY")
        print("==========================================================================================")
        for i, res in enumerate(results, 1):
            status = res.get("status", "unknown").upper()
            action = res.get("action", "")
            task_name = res.get("task", "")
            print(f"  Step {i}: [{status}] Action '{action}' on target '{task_name}'")
        print("==========================================================================================")
        return True
    finally:
        highlighter.close()


if __name__ == "__main__":
    asyncio.run(run_logical_task_pipeline())
