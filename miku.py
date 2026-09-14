"""
miku.py — Interactive OS Agent Command-Line Interface (REPL)

Main interactive entry point for the Miku OS Agent:
  - Uninhibited full autonomous live control: Bézier cursor glides and humanized typing.
  - Interactive REPL loop prompting with 'Miku > '.
  - Routes arbitrary user commands directly into run_computer_use_task().
  - Live Astra multimodal reasoning when OPENAI_API_KEY is available, with dynamic
    heuristic intent translation when operating offline.
"""

from __future__ import annotations

import os

# ==============================================================================
# 1. Complete Autonomous Unlock (Must be set before any Miku tool imports)
# ==============================================================================
os.environ["MIKU_LIVE_EXECUTION"] = "true"
os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

import ctypes
import json
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Circuit breaker unlocks
import tools.screen_inspector as screen_inspector
import tools.typing_automation as typing_automation
import tools.orchestrator as orchestrator
from tools.computer_use_agent import run_computer_use_task, ComputerUseTaskResult
from tools.orchestrator import AstraVisionClient

screen_inspector.REAL_CLICK_ENABLED = True
typing_automation.REAL_TYPE_ENABLED = True
orchestrator.REAL_CLICK_ENABLED = True
orchestrator.REAL_TYPE_ENABLED = True


BANNER = r"""
  __  __ _____ _  ___   _ 
 |  \/  |_   _| |/ / | | |
 | |\/| | | | | ' /| | | |   MIKU OS AGENT — v0.3.5
 | |  | |_| |_| . \| |_| |   Autonomous Computer Use & Vision Engine
 |_|  |_|_____|_|\_\\___/    Powered by gpt-6-astra
===============================================================================
 [*] Win32 Humanizer     : ACTIVE (Cubic Bézier Glides & Micro-Delayed Typing)
 [*] Autonomous Mode     : UNLOCKED (Live Non-Blocking OS Control)
 [*] Vision Architecture : Multimodal GDI BitBlt + Dynamic UI Grounding
===============================================================================
"""


def print_banner() -> None:
    print(BANNER, flush=True)
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
    if has_api_key:
        print(f"[+] Model Endpoint     : {model} (Live API Key Detected)", flush=True)
    else:
        print(f"[*] Model Endpoint     : {model} (Offline / Dynamic Heuristic Active)", flush=True)
        print("    Tip: Use '/key <YOUR_API_KEY>' to connect directly to live Astra API.", flush=True)
    print("\nType your command below, '/help' for commands, or 'exit' / 'quit' to quit.\n", flush=True)


def print_help() -> None:
    print("\n" + "-" * 60, flush=True)
    print("Miku Interactive REPL Commands:", flush=True)
    print("  <any objective>     Dispatches task to computer use agent", flush=True)
    print("                      e.g. 'open notepad and type hello world'", flush=True)
    print("                      e.g. 'open calculator and do 77 multiplied by 3'", flush=True)
    print("  /key <api_key>      Set or inspect OPENAI_API_KEY in real time", flush=True)
    print("  /model <name>       Set target vision model (default: openai/gpt-6-astra)", flush=True)
    print("  /status             Show system status & Win32 humanizer configuration", flush=True)
    print("  /clear              Clear the console screen", flush=True)
    print("  exit / quit         Exit the Miku REPL session", flush=True)
    print("-" * 60, flush=True)


def print_status() -> None:
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
    cur_pos = screen_inspector.get_current_cursor_pos()
    print("\n" + "=" * 60, flush=True)
    print("MIKU SYSTEM DIAGNOSTICS & STATUS:", flush=True)
    print(f"  Physical Cursor Pos : {cur_pos}", flush=True)
    print(f"  Real Click Enabled  : {screen_inspector.REAL_CLICK_ENABLED}", flush=True)
    print(f"  Real Type Enabled   : {typing_automation.REAL_TYPE_ENABLED}", flush=True)
    print(f"  Autonomous Mode     : {os.environ.get('MIKU_AUTONOMOUS_MODE')}", flush=True)
    print(f"  Live Execution      : {os.environ.get('MIKU_LIVE_EXECUTION')}", flush=True)
    print(f"  Active Vision Model : {model}", flush=True)
    print(f"  API Key Configured  : {'YES' if has_key else 'NO (Offline heuristic active)'}", flush=True)
    print("=" * 60, flush=True)


def create_dynamic_intent_runner(prompt: str) -> Callable[[Dict[str, Any]], str]:
    """
    Decomposes natural language desktop instructions into structured Win32 actions
    when running in offline mode without an active OPENAI_API_KEY.
    """
    clean_p = prompt.strip()
    lower_p = clean_p.lower()
    step_idx = 0

    # 1. Opening application (e.g. "open notepad", "launch calculator", "search for paint")
    app_target = None
    open_match = re.search(
        r"(?:open|launch|start|run|search for)\s+(?:the\s+)?([a-zA-Z0-9_\-\.\s]+?)(?:\s+(?:and|,|then|to)\s+|$)",
        lower_p,
    )
    if open_match:
        app_target = open_match.group(1).strip()

    # 2. Typing text (e.g. "type hello world", "type 'hello'")
    type_text = None
    type_match = re.search(
        r"(?:type|write|enter|input)\s+['\"]?([^'\"]+?)['\"]?(?:\s+(?:and|,|then)\s+|$)",
        clean_p,
        re.IGNORECASE,
    )
    if type_match:
        type_text = type_match.group(1).strip()

    # 3. Arithmetic calculation in Calculator (e.g. "do 77 multiplied by 3", "calculate 72 * 4")
    calc_expr = None
    calc_match = re.search(
        r"(\d+)\s*(?:multiplied by|\*|times|x)\s*(\d+)",
        lower_p,
    )
    if calc_match:
        n1, n2 = calc_match.group(1), calc_match.group(2)
        calc_expr = f"{n1}*{n2}="

    actions_queue: List[Dict[str, Any]] = []

    if app_target:
        actions_queue.append({"action": "press_key", "key": "win"})
        actions_queue.append({"action": "type", "text": app_target})
        actions_queue.append({"action": "press_key", "key": "enter"})
        actions_queue.append({"action": "wait", "seconds": 1.5})

    if calc_expr:
        actions_queue.append({"action": "type", "text": calc_expr})
        actions_queue.append({"action": "wait", "seconds": 0.5})
    elif type_text:
        actions_queue.append({"action": "type", "text": type_text})
        actions_queue.append({"action": "wait", "seconds": 0.5})

    if not actions_queue:
        actions_queue.append({"action": "type", "text": clean_p})

    actions_queue.append({
        "action": "terminate",
        "reason": f"Executed sequence for: '{clean_p}'",
    })

    def runner(payload: Dict[str, Any]) -> str:
        nonlocal step_idx
        if step_idx < len(actions_queue):
            act = actions_queue[step_idx]
            step_idx += 1
            return json.dumps(act)
        return json.dumps({"action": "terminate", "reason": "All planned actions executed."})

    return runner


def repl() -> None:
    """Main interactive REPL loop."""
    print_banner()

    while True:
        try:
            prompt_input = input("Miku > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[*] Session closed by user. Goodbye!")
            break

        if not prompt_input:
            continue

        # Handle termination commands
        if prompt_input.lower() in ("exit", "quit", "q", ":q"):
            print("[*] Session closed. Goodbye!")
            break

        # Handle built-in REPL commands
        if prompt_input.startswith("/key"):
            parts = prompt_input.split(maxsplit=1)
            if len(parts) > 1:
                new_key = parts[1].strip()
                os.environ["OPENAI_API_KEY"] = new_key
                mask = f"{new_key[:7]}...{new_key[-4:]}" if len(new_key) > 11 else "***"
                print(f"[+] OPENAI_API_KEY updated successfully ({mask}).")
            else:
                curr = os.environ.get("OPENAI_API_KEY", "")
                if curr:
                    mask = f"{curr[:7]}...{curr[-4:]}" if len(curr) > 11 else "***"
                    print(f"[*] Current OPENAI_API_KEY: {mask}")
                else:
                    print("[!] No OPENAI_API_KEY set. Usage: /key <your_api_key>")
            continue

        if prompt_input.startswith("/model"):
            parts = prompt_input.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1].strip()
                os.environ["ASTRA_MODEL"] = new_model
                print(f"[+] Active model set to: '{new_model}'")
            else:
                curr_model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
                print(f"[*] Current model: '{curr_model}'. Usage: /model <model_name>")
            continue

        if prompt_input.startswith("/status"):
            print_status()
            continue

        if prompt_input.startswith("/clear"):
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        if prompt_input.startswith("/help"):
            print_help()
            continue

        # ----------------------------------------------------------------------
        # Dispatch Dynamic Objective to Miku's Computer Use Agent Loop
        # ----------------------------------------------------------------------
        print(f"\n[*] [TASK DISPATCHED] Objective: \"{prompt_input}\"", flush=True)
        t_start = time.perf_counter()

        has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
        if has_api_key:
            model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
            client = AstraVisionClient(model=model)
        else:
            client = AstraVisionClient(
                model="local/dynamic-heuristic",
                api_runner=create_dynamic_intent_runner(prompt_input),
            )

        try:
            res: ComputerUseTaskResult = run_computer_use_task(
                objective=prompt_input,
                client=client,
                real_execution=True,
                delay_between_steps=0.25,
            )
            elapsed = time.perf_counter() - t_start
            print(f"\n[+] [TASK COMPLETED] Duration: {elapsed:.2f}s | Steps: {res.total_steps} | Status: {res.final_status}", flush=True)
            if res.output:
                print(f"    Summary: {res.output}", flush=True)
            if res.error:
                print(f"    [!] Details: {res.error}", flush=True)
        except KeyboardInterrupt:
            print("\n[!] [TASK ABORTED] Execution stopped by user via CTRL+C. Control returned to console.", flush=True)
        except Exception as exc:
            print(f"\n[!] [TASK ERROR] {exc}", flush=True)

        print("-" * 75, flush=True)


if __name__ == "__main__":
    repl()
