"""
run_notepad_automation.py — Desktop Task Execution Demonstration

Fulfills user request:
1. Read current screen to orient (read_screen_text).
2. Click target control (FastConfirm + Win32 Red Overlay + AI Intent Rationale).
3. Type 'Notepad' into search.
4. Type 'Miku orchestration layer online.' into document.
"""

import asyncio
import sys
import time

from agent_loop import AgentLoop, TaskSpec
from config import default_config
from human_gate import FastConfirm
from safe_executor import ActionDispatch
from ui_inspector import TreeInspector
from visual_highlighter import TargetHighlighter
from cli_macro_engine import RegexCommandParser


async def main():
    print("\n=======================================================")
    print("STARTING MIKU NOTEPAD AUTOMATION PIPELINE")
    print("=======================================================")

    mock_nodes = [
        {"name": "Desktop Screen", "automation_id": "root_screen", "control_type": "Pane", "bounds": (0, 0, 1920, 1080), "center": (960, 540), "text": "Windows 11 Desktop | Taskbar | Start Button | Active Applications"},
        {"name": "Start", "automation_id": "btn_start", "control_type": "Button", "bounds": (10, 1040, 50, 1075), "center": (30, 1055)},
        {"name": "Search", "automation_id": "txt_search", "control_type": "Edit", "bounds": (60, 1040, 300, 1075), "center": (180, 1055)},
        {"name": "Text Editor", "automation_id": "txt_notepad_doc", "control_type": "Document", "bounds": (200, 200, 1200, 800), "center": (700, 500), "text": "Untitled - Notepad"},
    ]

    inspector = TreeInspector(mock_elements=mock_nodes)
    highlighter = TargetHighlighter(enabled=True, line_width=5)
    
    # Custom input handler to simulate operator approving steps for headless test run
    def simulated_operator_input(prompt):
        print("\n[OPERATOR INTERFACE SIMULATOR] Auto-approving step with 'y'...")
        return "y"

    gate = FastConfirm(config=default_config, input_handler=simulated_operator_input, highlighter=highlighter)
    executor = ActionDispatch(simulation_mode=True)  # Hardware execution simulated for automated verification
    agent = AgentLoop(
        inspector=inspector,
        gate=gate,
        executor=executor,
        config=default_config,
    )

    # Sequence of tasks requested by user
    tasks = [
        TaskSpec(
            action_type="read_screen_text",
            target_name="Desktop Screen",
            rationale="I am reading the screen text to orient myself and locate the Windows Start button.",
        ),
        TaskSpec(
            action_type="click",
            target_name="Start",
            rationale="I am clicking the Windows Start button to open the search bar.",
        ),
        TaskSpec(
            action_type="type",
            target_name="Search",
            payload="Notepad\n",
            rationale="I am typing 'Notepad' into the search bar to launch the Notepad application.",
        ),
        TaskSpec(
            action_type="type",
            target_name="Text Editor",
            payload="Miku orchestration layer online.",
            rationale="I am typing 'Miku orchestration layer online.' into the document.",
        ),
    ]

    try:
        results = await agent.run(tasks)
        print("\n=======================================================")
        print("AUTOMATION TASK PIPELINE COMPLETED")
        for i, res in enumerate(results, 1):
            print(f"  Step {i}: {res.get('action')} on '{res.get('task')}' -> Status: {res.get('status').upper()}")
        print("=======================================================")
    finally:
        highlighter.close()


if __name__ == "__main__":
    asyncio.run(main())
