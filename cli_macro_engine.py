"""
cli_macro_engine.py — Deterministic CLI Engine for Miku (NO AI, NO API)

This engine replaces the LLM. It uses strict Regular Expressions to parse
terminal commands, routing system commands to OSController and UI commands to AgentLoop.
Clean, professional logging using standard system tags ([INFO], [ERROR]).
"""

import asyncio
import re
import sys
from typing import Optional

from agent_loop import AgentLoop, TaskSpec
from config import default_config
from human_gate import FastConfirm
from os_controller import OSController
from safe_executor import ActionDispatch
from ui_inspector import TreeInspector


class DeterministicCLI:
    def __init__(self):
        self.os_ctrl = OSController()
        # Strict syntax patterns
        self.sys_pattern = re.compile(r"^sys\s+(kill|lock|boot|ps|info|screenshot)\s*(.*)$", re.IGNORECASE)
        self.open_pattern = re.compile(r"^open\s+(?:['\"](.+?)['\"]|([a-zA-Z0-9_\-\.\:\\]+))$", re.IGNORECASE)
        self.click_pattern = re.compile(r"^click\s+['\"]?(.+?)['\"]?$", re.IGNORECASE)
        self.type_pattern = re.compile(r"^type\s+['\"](.+?)['\"]\s+into\s+['\"]?(.+?)['\"]?$", re.IGNORECASE)
        self.read_pattern = re.compile(r"^read\s+['\"]?(.+?)['\"]?$", re.IGNORECASE)

    def execute_sys_command(self, cmd_type: str, args: str):
        """Routes direct OS commands."""
        args_clean = args.strip()
        if cmd_type == "kill":
            result = self.os_ctrl.kill_process(args_clean)
        elif cmd_type == "lock":
            result = self.os_ctrl.lock_workstation()
        elif cmd_type == "boot":
            result = self.os_ctrl.boot_workspace()
        elif cmd_type == "ps":
            result = self.os_ctrl.list_processes(args_clean)
        elif cmd_type == "info":
            result = self.os_ctrl.get_system_info()
        elif cmd_type == "screenshot":
            result = self.os_ctrl.take_screenshot(args_clean or "screenshot.png")
        else:
            result = {"status": "error", "msg": f"[ERROR] Unknown sys command: {cmd_type}"}
        print(result["msg"])
        return result

    def parse_ui_command(self, command: str) -> Optional[TaskSpec]:
        """Parses standard UI automation tasks."""
        command = command.strip()

        match = self.open_pattern.match(command)
        if match:
            target = match.group(1) or match.group(2)
            return TaskSpec(
                action_type="open",
                target_name=target,
                rationale=f"[INFO] Launching application '{target}' via OS shell.",
            )

        match = self.click_pattern.match(command)
        if match:
            target = match.group(1)
            return TaskSpec(
                action_type="click",
                target_name=target,
                rationale=f"[INFO] Executing UI click event on '{target}'.",
            )

        match = self.type_pattern.match(command)
        if match:
            payload, target = match.groups()
            return TaskSpec(
                action_type="type",
                target_name=target,
                payload=payload,
                rationale=f"[INFO] Executing UI keystroke injection into '{target}'.",
            )

        match = self.read_pattern.match(command)
        if match:
            target = match.group(1)
            return TaskSpec(
                action_type="read_screen_text",
                target_name=target,
                rationale=f"[INFO] Inspecting screen text on '{target}'.",
            )

        return None


async def run_terminal():
    print("==================================================")
    print("SYSTEM AUTOMATION TERMINAL [DETERMINISTIC MODE]")
    print("==================================================")
    cli = DeterministicCLI()
    agent = AgentLoop(
        inspector=TreeInspector(),
        gate=FastConfirm(config=default_config),
        executor=ActionDispatch(),
        config=default_config,
    )

    while True:
        try:
            cmd = input("\nTerminal> ").strip()
            if cmd.lower() in ["exit", "quit"]:
                print("[INFO] Terminating session.")
                break
            if not cmd:
                continue

            # 1. Check for OS System Override
            sys_match = cli.sys_pattern.match(cmd)
            if sys_match:
                cli.execute_sys_command(sys_match.group(1).lower(), sys_match.group(2))
                continue

            # 2. Process UI Automation Task
            task = cli.parse_ui_command(cmd)
            if task:
                result = await agent.run_task(task)
                if task.action_type == "read_screen_text" and result.get("status") == "success":
                    print(f"\n[EXTRACTED TEXT]:\n{result.get('data', {}).get('text', '')}")
            else:
                print(
                    "[ERROR] Invalid syntax. Expected: sys <cmd>, open <app>, click <target>, type <txt> into <target>, or read <target>."
                )

        except KeyboardInterrupt:
            print("\n[INFO] Session aborted by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Command failed: {e}")


# Alias for backward compatibility
interactive_macro_terminal = run_terminal
RegexCommandParser = DeterministicCLI

if __name__ == "__main__":
    if sys.platform != "win32":
        print("[ERROR] This engine requires Windows UIAutomation.")
        sys.exit(1)
    asyncio.run(run_terminal())
