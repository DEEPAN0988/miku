"""
cli_macro_engine.py — Deterministic CLI Engine for Miku (NO AI, NO API)

This engine replaces the LLM. It uses strict Regex pattern matching to parse
commands you type in the terminal and maps them directly to the AgentLoop.
You must use exact syntax. Natural language is not supported.
"""

import asyncio
import re
import sys
from typing import Optional

from agent_loop import AgentLoop, TaskSpec
from config import default_config
from ui_inspector import TreeInspector
from visual_highlighter import TargetHighlighter
from human_gate import FastConfirm
from safe_executor import ActionDispatch

class RegexCommandParser:
    def __init__(self):
        # 1. READ Command: read <target>
        self.read_pattern = re.compile(r"^read\s+['\"]?(.+?)['\"]?$", re.IGNORECASE)
        
        # 2. CLICK Command: click <target>
        self.click_pattern = re.compile(r"^click\s+['\"]?(.+?)['\"]?$", re.IGNORECASE)
        
        # 3. TYPE Command: type <payload> into <target>
        self.type_pattern = re.compile(r"^type\s+['\"](.+?)['\"]\s+into\s+['\"]?(.+?)['\"]?$", re.IGNORECASE)

    def parse(self, user_input: str) -> Optional[TaskSpec]:
        user_input = user_input.strip()

        # Check for TYPE
        match = self.type_pattern.match(user_input)
        if match:
            payload, target = match.groups()
            return TaskSpec(
                action_type="type",
                target_name=target,
                payload=payload,
                rationale=f"[HARDCODED MACRO] Operator commanded TYPE into '{target}'."
            )

        # Check for CLICK
        match = self.click_pattern.match(user_input)
        if match:
            target = match.groups()[0]
            return TaskSpec(
                action_type="click",
                target_name=target,
                rationale=f"[HARDCODED MACRO] Operator commanded CLICK on '{target}'."
            )

        # Check for READ
        match = self.read_pattern.match(user_input)
        if match:
            target = match.groups()[0]
            return TaskSpec(
                action_type="read_screen_text",
                target_name=target,
                rationale=f"[HARDCODED MACRO] Operator commanded READ on '{target}'."
            )

        return None

async def interactive_macro_terminal():
    print("\n" + "="*50)
    print("MIKU AUTOMATION: REGEX MACRO ENGINE (NO AI)")
    print("="*50)
    print("Syntax Rules:")
    print("  1. read <target>")
    print("  2. click <target>")
    print("  3. type \"<text>\" into <target>")
    print("="*50)

    # Boot the execution framework
    inspector = TreeInspector()
    highlighter = TargetHighlighter(enabled=True, line_width=5)
    gate = FastConfirm(config=default_config, highlighter=highlighter)
    executor = ActionDispatch()
    
    agent = AgentLoop(
        inspector=inspector, gate=gate, executor=executor, config=default_config
    )
    parser = RegexCommandParser()

    while True:
        try:
            user_input = input("\nMiku-Macro> ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            task = parser.parse(user_input)
            
            if task:
                result = await agent.run_task(task)
                if task.action_type == "read_screen_text" and result.get("status") == "success":
                    print(f"\n[EXTRACTED TEXT]:\n{result.get('data', {}).get('text', '')}")
            else:
                print("[!] Syntax Error. You must use exact format (e.g., click Start)")
                
        except KeyboardInterrupt:
            print("\n[SYSTEM] Terminated by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")

    highlighter.close()

if __name__ == "__main__":
    if sys.platform != "win32":
        print("This requires Windows UIAutomation.")
        sys.exit(1)
    asyncio.run(interactive_macro_terminal())
