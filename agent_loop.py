"""
agent_loop.py — Async Task Runner for Desktop Automation Orchestration Layer

STRICT SAFETY CONSTRAINTS:
1. Every action step queries TreeInspector for exact OS accessibility coordinates or text.
2. Every physical action (click, type) pauses for FastConfirm authorization while displaying a visual red target overlay.
3. Read-only actions (read_screen_text) BYPASS the FastConfirm safety gate, returning text directly to Miku.
4. If an element is missing or action is denied, the pipeline HALTS gracefully.
   No unsupervised auto-retries. Interactive human recovery instructions are requested.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from config import OrchestratorConfig, default_config
from human_gate import ActionDeniedError, AuthToken, ConfirmationTimeoutError, FastConfirm
from safe_executor import ActionDispatch, UnauthorizedActionError
from ui_inspector import ElementNotFoundError, TreeInspector


class PipelineHaltedError(Exception):
    """Raised when the automation pipeline halts due to an unrecoverable failure or user abort."""
    pass


@dataclass
class TaskSpec:
    """Represents a single automation step specification."""
    action_type: str  # "click", "type", "read_screen_text"
    target_name: str  # Logical name or AutomationId in UI tree
    control_type: Optional[str] = None
    payload: Optional[str] = None  # Text content for "type" actions
    rationale: Optional[str] = None  # AI reasoning intent presented to human operator
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        """Constructs a TaskSpec from a dictionary representation."""
        return cls(
            action_type=data.get("action_type") or data.get("action", "click"),
            target_name=data.get("target_name") or data.get("target", ""),
            control_type=data.get("control_type"),
            payload=data.get("payload") or data.get("text"),
            rationale=data.get("rationale") or data.get("intent") or data.get("reasoning"),
            description=data.get("description"),
        )


class AgentLoop:
    """
    Core asyncio task runner orchestrating tree inspection, human gate confirmation,
    visual target highlighting, read-only text extraction, and safe physical execution.
    """

    def __init__(
        self,
        inspector: Optional[TreeInspector] = None,
        gate: Optional[FastConfirm] = None,
        executor: Optional[ActionDispatch] = None,
        config: Optional[OrchestratorConfig] = None,
        input_handler: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or default_config
        self.inspector = inspector or TreeInspector()
        self.gate = gate or FastConfirm(config=self.config, input_handler=input_handler)
        self.executor = executor or ActionDispatch()
        self.input_handler = input_handler
        self.step_count = 0

    async def _read_recovery_input(self, prompt: str) -> str:
        """Reads user recovery instructions asynchronously."""
        if self.input_handler is not None:
            return self.input_handler(prompt)
        print(prompt, end="", flush=True)
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        return line.strip() if line else ""

    async def _handle_manual_recovery(self, reason: str, task: TaskSpec) -> str:
        """
        Presents manual recovery instructions to the human operator upon step failure.
        
        Options:
          [r] Retry current step (after manual window adjustment)
          [s] Skip this step and proceed
          [q] Quit pipeline (default)
        """
        print(f"\n=======================================================")
        print(f"[PIPELINE HALTED] Step execution halted for target '{task.target_name}'.")
        print(f"Reason: {reason}")
        print(f"=======================================================")
        prompt = "Manual Recovery Options: [r] Retry / [s] Skip / [q] Quit pipeline (default): "

        try:
            choice = await asyncio.wait_for(
                self._read_recovery_input(prompt),
                timeout=self.config.CONFIRMATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            print("[PIPELINE HALTED] Recovery prompt timed out. Quitting pipeline.")
            return "q"

        choice_clean = choice.strip().lower()
        if choice_clean in ("r", "retry"):
            print("[MANUAL RECOVERY] User requested explicit retry of step.")
            return "r"
        elif choice_clean in ("s", "skip"):
            print("[MANUAL RECOVERY] User requested skipping step.")
            return "s"
        else:
            print("[MANUAL RECOVERY] Quitting pipeline.")
            return "q"

    async def run_task(self, task_input: Union[TaskSpec, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes a single TaskSpec or task dictionary through the inspection, confirmation,
        and dispatch pipeline. Enforces step budget limits and human recovery hooks.
        """
        task = task_input if isinstance(task_input, TaskSpec) else TaskSpec.from_dict(task_input)

        self.step_count += 1
        if self.step_count > self.config.MAX_EXECUTION_STEPS_PER_RUN:
            raise PipelineHaltedError(
                f"Maximum execution step budget ({self.config.MAX_EXECUTION_STEPS_PER_RUN}) exceeded."
            )

        print(f"\n--- [STEP {self.step_count}] Processing Task: {task.action_type.upper()} '{task.target_name}' ---")

        # READ-ONLY ACTION BRANCH: Bypass FastConfirm gate, zero physical hardware input
        if task.action_type in ("read_screen_text", "read", "get_text", "read_text"):
            print(f"[MIKU IS READING]: '{task.target_name}'")
            if task.rationale:
                print(f"  Rationale: {task.rationale}")

            try:
                text_data = self.inspector.read_element_text(task.target_name)
                print(f"[UI INSPECTOR] Text extracted ({len(text_data.get('text', ''))} chars).")
                return {
                    "status": "success",
                    "task": task.target_name,
                    "action": task.action_type,
                    "data": text_data,
                }
            except ElementNotFoundError as e:
                recovery = await self._handle_manual_recovery(str(e), task)
                if recovery == "r":
                    return await self.run_task(task_input)
                elif recovery == "s":
                    return {"status": "skipped", "target": task.target_name, "reason": str(e)}
                else:
                    raise PipelineHaltedError(f"Pipeline aborted by user: {e}")

        # HARDWARE ACTIONS BRANCH: Strict FastConfirm gate authorization required
        while True:
            # Step a: Ask TreeInspector for element coordinates & bounding box
            try:
                elem_info = self.inspector.find_element_by_name(
                    name=task.target_name,
                    control_type=task.control_type,
                )
                coords: Tuple[int, int] = elem_info["center"]
                bounds: Optional[Tuple[int, int, int, int]] = elem_info.get("bounds")
                print(f"[UI INSPECTOR] Found node '{elem_info['name']}' at X:{coords[0]}, Y:{coords[1]}.")
            except ElementNotFoundError as e:
                recovery = await self._handle_manual_recovery(str(e), task)
                if recovery == "r":
                    continue  # Explicit human-approved retry
                elif recovery == "s":
                    return {"status": "skipped", "target": task.target_name, "reason": str(e)}
                else:
                    raise PipelineHaltedError(f"Pipeline aborted by user: {e}")

            # Step b: Pause and trigger FastConfirm safety gate (with visual overlay & AI rationale)
            try:
                token: AuthToken = await self.gate.request_confirmation(
                    action_type=task.action_type,
                    target=task.target_name,
                    coords=coords,
                    payload=task.payload,
                    bounds=bounds,
                    rationale=task.rationale,
                )
            except (ActionDeniedError, ConfirmationTimeoutError) as e:
                recovery = await self._handle_manual_recovery(str(e), task)
                if recovery == "r":
                    continue
                elif recovery == "s":
                    return {"status": "skipped", "target": task.target_name, "reason": str(e)}
                else:
                    raise PipelineHaltedError(f"Pipeline aborted by user: {e}")

            # Step c: Pass token to safe_executor.py
            try:
                if task.action_type == "click":
                    exec_result = self.executor.click(token)
                elif task.action_type == "type":
                    exec_result = self.executor.type(token)
                else:
                    raise ValueError(f"Unsupported action type: {task.action_type}")

                return {
                    "status": "success",
                    "task": task.target_name,
                    "action": task.action_type,
                    "exec_result": exec_result,
                }
            except UnauthorizedActionError as e:
                recovery = await self._handle_manual_recovery(f"Security Error: {e}", task)
                if recovery == "r":
                    continue
                elif recovery == "s":
                    return {"status": "skipped", "target": task.target_name, "reason": str(e)}
                else:
                    raise PipelineHaltedError(f"Pipeline aborted due to authorization failure: {e}")

    async def run(self, task_list: List[Union[TaskSpec, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Executes a deterministic sequence of TaskSpecs or task dictionaries.
        """
        results = []
        print(f"\n=======================================================")
        print(f"STARTING DESKTOP AUTOMATION PIPELINE ({len(task_list)} tasks)")
        print(f"Safety Gate: {'ACTIVE' if self.config.CONFIRMATION_GATE_ENABLED else 'DISABLED'}")
        print(f"Max Step Budget: {self.config.MAX_EXECUTION_STEPS_PER_RUN}")
        print(f"=======================================================")

        try:
            for task_item in task_list:
                res = await self.run_task(task_item)
                results.append(res)
                print(f"[STEP {self.step_count} COMPLETE] Result: {res['status']}")
        except PipelineHaltedError as e:
            print(f"\n[PIPELINE TERMINATED] {e}")
            results.append({"status": "pipeline_halted", "reason": str(e)})

        print(f"\n=======================================================")
        print(f"PIPELINE SUMMARY: Completed {len([r for r in results if r.get('status') == 'success'])}/{len(task_list)} tasks.")
        print(f"=======================================================")
        return results


async def demo_runner():
    """Demonstration runner for interactive testing with read_screen_text, visual target highlighter, and rationale."""
    print("Running AgentLoop Demonstration Mode (Includes Read-Only Text Extraction)...")
    
    mock_elements = [
        {"name": "File", "automation_id": "MenuItemFile", "control_type": "MenuItem", "bounds": (10, 10, 50, 30), "center": (30, 20)},
        {"name": "DocumentContainer", "automation_id": "DocPane", "control_type": "Pane", "bounds": (50, 50, 600, 400), "center": (325, 225), "text": "Miku Confidential Project Report 2026."},
        {"name": "Search Box", "automation_id": "EditSearch", "control_type": "Edit", "bounds": (100, 50, 400, 80), "center": (250, 65)},
        {"name": "Submit", "automation_id": "BtnSubmit", "control_type": "Button", "bounds": (450, 750, 550, 790), "center": (500, 770)},
    ]
    
    inspector = TreeInspector(mock_elements=mock_elements)
    executor = ActionDispatch(simulation_mode=True)
    loop_runner = AgentLoop(inspector=inspector, executor=executor)
    
    tasks = [
        TaskSpec(
            action_type="read_screen_text",
            target_name="DocumentContainer",
            rationale="Inspect the contents of DocumentContainer to verify active report text.",
        ),
        TaskSpec(
            action_type="click",
            target_name="File",
            control_type="MenuItem",
            rationale="Open File menu.",
        ),
    ]
    
    await loop_runner.run(tasks)


if __name__ == "__main__":
    if "--demo" in sys.argv or len(sys.argv) > 1:
        asyncio.run(demo_runner())
    else:
        print("Usage: python agent_loop.py --demo")
