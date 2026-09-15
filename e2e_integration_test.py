"""
e2e_integration_test.py — End-to-End Orchestration Layer Test

Simulates an LLM agent (Miku) generating a sequence of tasks to verify:
1. CQRS Routing (Read actions bypass FastConfirm).
2. UI Inspector (Finds real Windows elements via accessibility tree).
3. Visual Highlighter (Draws Win32 red box on target).
4. FastConfirm Gate (Halts execution and displays AI rationale).
5. Safe Executor (Consumes AuthToken and dispatches OS hardware event).
"""

import asyncio
import sys

# Import the architecture components
from config import default_config
from ui_inspector import TreeInspector
from visual_highlighter import TargetHighlighter
from human_gate import FastConfirm
from safe_executor import ActionDispatch
from agent_loop import AgentLoop, TaskSpec


async def run_e2e_test():
    print("\n" + "=" * 50)
    print("INITIALIZING MIKU ORCHESTRATION E2E TEST")
    print("=" * 50)

    # 1. Initialize the Nervous System
    print("[SYSTEM] Booting UI Inspector...")
    inspector = TreeInspector()

    print("[SYSTEM] Booting Visual Highlighter Daemon...")
    highlighter = TargetHighlighter(enabled=True, line_width=5)

    print("[SYSTEM] Booting FastConfirm Gate...")
    gate = FastConfirm(config=default_config, highlighter=highlighter)

    print("[SYSTEM] Booting Hardware Executor...")
    executor = ActionDispatch()

    print("[SYSTEM] Initializing Agent Loop...")
    agent = AgentLoop(
        inspector=inspector,
        gate=gate,
        executor=executor,
        config=default_config,
    )

    # 2. Simulate Miku's JSON tool calls based on the schemas
    # We use safe, ubiquitous Windows elements for testing.
    mock_llm_tasks = [
        {
            "action": "read_screen_text",
            "target": "Taskbar",
            "rationale": "I am reading the Taskbar to verify the system state and locate the Start button."
        },
        {
            "action": "click",
            "target": "Start", 
            "rationale": "I am clicking the Start button to open the Windows system menu and confirm hardware execution is working."
        }
    ]

    print("\n[SYSTEM] Starting Task Pipeline...\n")

    # 3. Process the simulated LLM queue
    for step_num, task_dict in enumerate(mock_llm_tasks, 1):
        print(f"\n--- EXECUTING STEP {step_num} ---")
        try:
            # Parse the dictionary just like LangChain/OpenAI function calling output
            task_spec = TaskSpec.from_dict(task_dict)
            
            # Run the task through the orchestrator
            result = await agent.run_task(task_spec)
            
            print(f"\n[STEP {step_num} RESULT]: {result['status'].upper()}")
            
            # If it's a read action, show a snippet of what Miku "saw"
            if task_spec.action_type == "read_screen_text" and "data" in result:
                snippet = str(result['data'].get('text', ''))[:100].replace('\n', ' | ')
                print(f"[EXTRACTED DATA]: {snippet}...")
                
        except Exception as e:
            print(f"\n[!] E2E TEST FAILED ON STEP {step_num}: {e}")
            break
            
        # Brief pause to simulate LLM thinking between steps
        await asyncio.sleep(1.0)

    print("\n" + "=" * 50)
    print("E2E TEST COMPLETE")
    print("=" * 50)

    # 4. Clean shutdown of the Win32 daemon thread
    highlighter.close()


if __name__ == "__main__":
    if sys.platform != "win32":
        print("This E2E test requires Windows for UIAutomation and Win32GUI overlay.")
        sys.exit(1)
        
    try:
        asyncio.run(run_e2e_test())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Test aborted by user.")
