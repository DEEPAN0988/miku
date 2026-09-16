"""
miku.py — Interactive OS Agent Command-Line Interface (REPL)

Main interactive entry point for the Miku OS Agent:
  - Uninhibited full autonomous live control: Bézier cursor glides and humanized typing.
  - Interactive REPL loop prompting with 'Miku > '.
  - Native Inference: Miku v0.2 loaded directly via PyTorch (no pre-trained models)
  - Vision Encoder: [NOT YET TRAINED] using Win32 structured text parsing
"""

from __future__ import annotations

import os

# ==============================================================================
# 1. Complete Autonomous Unlock (Must be set before any Miku tool imports)
# ==============================================================================
os.environ["MIKU_LIVE_EXECUTION"] = "true"
os.environ["MIKU_AUTONOMOUS_MODE"] = "true"
os.environ["MIKU_AUTO_APPROVE"] = "true"

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
from tools.local_vision import (
    get_action_coordinates,
    capture_desktop_pil,
    ground_and_click_target,
    HAS_VISION_DEPS,
    get_acceleration_device_and_dtype,
)

screen_inspector.REAL_CLICK_ENABLED = True
typing_automation.REAL_TYPE_ENABLED = True
orchestrator.REAL_CLICK_ENABLED = True
orchestrator.REAL_TYPE_ENABLED = True


BANNER = r"""
  __  __ _____ _  ___   _ 
 |  \/  |_   _| |/ / | | |
 | |\/| | | | | ' /| | | |   MIKU OS AGENT — v0.2
 | |  | |_| |_| . \| |_| |   Autonomous Computer Use & Vision Engine
 |_|  |_|_____|_|\_\\___/    Native In-Process VLM + Multimodal Bridge
===============================================================================
 [*] Win32 Humanizer     : ACTIVE (Cubic Bézier Glides & Micro-Delayed Typing)
 [*] Native Inference    : Miku v0.2 loaded directly via PyTorch (no pre-trained models)
 [*] Vision Encoder      : [NOT YET TRAINED] using Win32 structured text parsing
 [*] Autonomous Mode     : UNLOCKED (Live Non-Blocking OS Control)
===============================================================================
"""


def print_banner() -> None:
    print(BANNER, flush=True)
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
    dev, dtype = get_acceleration_device_and_dtype()

    print(" [*] Local Inference     : [ACTIVE] Miku v0.2 (own checkpoint)", flush=True)
    print(" [*] Vision Encoder      : [NOT YET TRAINED] Win32 structured text fallback", flush=True)

    # Eagerly load & cache Miku model at startup to avoid cold-start on first command
    from tools.miku_inference import load_miku_model, get_model_info
    info = get_model_info()
    if info.get("checkpoint_available"):
        print(" [*] Loading Miku model (one-time startup)...", flush=True)
        try:
            _, _, dev = load_miku_model()
            print(f" [+] Model ready on {dev}.", flush=True)
        except Exception as e:
            print(f" [!] Model load failed: {e}", flush=True)
    else:
        print("[!] [WARNING] Miku checkpoint not found! Run train/train_sft.py first.", flush=True)

    print("\nType your command below, '/help' for commands, or 'exit' / 'quit' to quit.\n", flush=True)


def print_help() -> None:
    print("\n" + "-" * 60, flush=True)
    print("Miku Interactive REPL Commands:", flush=True)
    print("  <any objective>     Dispatches task to computer use agent loop", flush=True)
    print("                      e.g. 'open notepad and type hello world'", flush=True)
    print("                      e.g. 'open calculator and do 77 multiplied by 3'", flush=True)
    print("  click <element>     Grounds element with in-process VLM and clicks", flush=True)
    print("                      e.g. 'click Calculator icon' or 'click Dark mode'", flush=True)
    print("  /vision <element>   Explicit in-process VLM visual grounding & click", flush=True)
    print("  /key <api_key>      Set or inspect OPENAI_API_KEY in real time", flush=True)
    print("  /model <name>       Set target vision model (default: openai/gpt-6-astra)", flush=True)
    print("", flush=True)
    print("  /check              Test Miku Inference health", flush=True)
    print("  /status             Show system status, VLM hardware acceleration, & humanizer", flush=True)
    print("  /clear              Clear the console screen", flush=True)
    print("  exit / quit         Exit the Miku REPL session", flush=True)
    print("-" * 60, flush=True)


def print_status() -> None:
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
    cur_pos = screen_inspector.get_current_cursor_pos()
    dev, dtype = get_acceleration_device_and_dtype()

    print("\n" + "=" * 60, flush=True)
    print("MIKU SYSTEM DIAGNOSTICS & STATUS:", flush=True)
    print(f"  Physical Cursor Pos : {cur_pos}", flush=True)
    print(f"  Real Click Enabled  : {screen_inspector.REAL_CLICK_ENABLED}", flush=True)
    print(f"  Real Type Enabled   : {typing_automation.REAL_TYPE_ENABLED}", flush=True)
    print(f"  Autonomous Mode     : {os.environ.get('MIKU_AUTONOMOUS_MODE')}", flush=True)
    print(f"  Live Execution      : {os.environ.get('MIKU_LIVE_EXECUTION')}", flush=True)
    print(f"  In-Process VLM Deps : {'AVAILABLE' if HAS_VISION_DEPS else 'MISSING'}", flush=True)
    print(f"  VLM Acceleration    : Device={dev}, DType={dtype}", flush=True)
    print(f"  Cloud Vision Model  : {model}", flush=True)
    print(f"  API Key Configured  : {'YES' if has_key else 'NO (Miku inference active)'}", flush=True)
    print("=" * 60, flush=True)


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



        if prompt_input.startswith("/vision"):
            parts = prompt_input.split(maxsplit=1)
            if len(parts) > 1:
                target_elem = parts[1].strip()
                print(f"\n[*] [LOCAL VISION] Grounding '{target_elem}' with native in-process VLM...", flush=True)
                res = ground_and_click_target(target_elem, duration=0.20, real_execution=True)
                if res.get("success"):
                    coords = res.get("coordinates")
                    print(f"[+] [LOCAL VISION SUCCESS] Grounded and clicked '{target_elem}' at screen coordinate {coords}!", flush=True)
                else:
                    print(f"[!] [LOCAL VISION FAILED] {res.get('message', res.get('error'))}", flush=True)
            else:
                print("[!] Usage: /vision <target element to click> (e.g. /vision Calculator icon)")
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

        if prompt_input.startswith("/check"):
            from tools.miku_inference import get_model_info
            info = get_model_info()
            print("[MIKU SYSTEM HEALTH]")
            print(f"- Checkpoint Available: {info.get('checkpoint_available')}")
            print(f"- Checkpoint Path:      {info.get('checkpoint_path')}")
            print(f"- Weights Loaded:       {info.get('loaded')}")
            if info.get("loaded"):
                print(f"- Device:               {info.get('device')}")
                print(f"- Parameter Count:      {info.get('params')}")
                print(f"- Config:               {info.get('config')}")
            continue

        # ----------------------------------------------------------------------
        # Direct Natural Language Ground & Click Dispatch via In-Process VLM
        # ----------------------------------------------------------------------
        click_match = re.match(r"^(?:click|tap|press\s+on)\s+(?:the\s+)?(.+)$", prompt_input, re.IGNORECASE)
        if click_match and not any(kw in prompt_input.lower() for kw in ("and", "then", "open", "type", "search")):
            target_elem = click_match.group(1).strip()
            print(f"\n[*] [LOCAL VISION] Grounding '{target_elem}' with native in-process VLM...", flush=True)
            res = ground_and_click_target(target_elem, duration=0.20, real_execution=True)
            if res.get("success"):
                coords = res.get("coordinates")
                print(f"[+] [LOCAL VISION SUCCESS] Grounded and clicked '{target_elem}' at screen coordinate {coords}!", flush=True)
            else:
                print(f"[!] [LOCAL VISION FAILED] {res.get('message', res.get('error'))}", flush=True)
            print("-" * 75, flush=True)
            continue

        # ----------------------------------------------------------------------
        # Multi-Step Computer Use Agent Loop Dispatch (Cloud Astra or Miku Inference)
        # ----------------------------------------------------------------------
        print(f"\n[*] [TASK DISPATCHED] Objective: \"{prompt_input}\"", flush=True)
        t_start = time.perf_counter()

        has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
        if has_api_key:
            model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
            client = AstraVisionClient(model=model)
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
        else:
            # 100% Native Real-Time Autonomous Interactive Loop Dispatch
            # -------------------------------------------------------------------
            from tools.interactive_loop import run_autonomous_task_loop
            try:
                loop_res = run_autonomous_task_loop(prompt_input, max_steps=8, real_execution=True)
                print(f"[*] [MIKU AUTONOMOUS LOOP FINISHED] Status: {loop_res.get('status')}", flush=True)
            except KeyboardInterrupt:
                print("\n[!] [TASK ABORTED] Execution stopped by user via CTRL+C. Control returned to console.", flush=True)
            except Exception as exc:
                print(f"\n[!] [TASK ERROR] {exc}", flush=True)

        print("-" * 75, flush=True)



if __name__ == "__main__":
    repl()
