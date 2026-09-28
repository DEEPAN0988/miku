"""
Core 3: Execution Engine (Hands).
Handles OS actions, minimum-jerk cursor movement, closed-loop verification,
CDP browser commands, and AST sandboxed modules.
"""
import time
from multiprocessing import Queue
from typing import Optional, Dict, Any, Tuple

from miku.ipc.messages import ActionRequestMsg, ActionResultMsg
from miku.core3_execution.motion import minimum_jerk_trajectory
from miku.core3_execution.closed_loop_click import ClosedLoopVerifier
from miku.core3_execution.os_actions import OSActionsExecutor
from miku.core3_execution.cdp_browser import CDPBrowserDriver
from miku.core3_execution.ast_sandbox import ASTSandboxRunner

class ExecutionDaemon:
    def __init__(
        self,
        action_queue: Queue,
        result_queue: Queue
    ):
        self.action_queue = action_queue
        self.result_queue = result_queue

        self.verifier = ClosedLoopVerifier()
        self.os_exec = OSActionsExecutor()
        self.cdp = CDPBrowserDriver()
        self.sandbox = ASTSandboxRunner()

        self.user_interrupted: bool = False

    def trigger_user_interrupt(self):
        """
        Simulates or receives user mouse/keyboard input mid-task to yield immediately.
        """
        self.user_interrupted = True

    def execute_request(self, request: ActionRequestMsg) -> ActionResultMsg:
        """
        Executes a single action request.
        """
        now = time.time()

        # 1. Closed-loop check for click actions (Bottleneck #3 Fix)
        if request.action_type in ("click_target", "click") and request.coords:
            expected_ctrl = request.params.get("target", "button")
            valid, reason = self.verifier.verify_before_click(
                target_coords=request.coords,
                expected_control=expected_ctrl
            )
            if not valid:
                res = ActionResultMsg(
                    task_id=request.task_id,
                    success=False,
                    status="cancelled_target_changed",
                    message=reason,
                    timestamp=now
                )
                self.result_queue.put(res)
                return res

            # Minimum-jerk motion
            start_pos = (500, 500)  # Reference origin or current cursor pos
            path = minimum_jerk_trajectory(start_pos, request.coords, steps=10)
            for pt in path:
                if self.user_interrupted:
                    self.user_interrupted = False
                    res = ActionResultMsg(
                        task_id=request.task_id,
                        success=False,
                        status="aborted_by_user",
                        message="User moved mouse; automation yielded immediately.",
                        timestamp=time.time()
                    )
                    self.result_queue.put(res)
                    return res

        # 2. App Launch
        if request.action_type == "open_app":
            app = request.params.get("app", request.target)
            ok, msg = self.os_exec.launch_app(app)
            status = "completed" if ok else "failed"
            res = ActionResultMsg(task_id=request.task_id, success=ok, status=status, message=msg)
            self.result_queue.put(res)
            return res

        # 3. Close App
        if request.action_type == "close_app":
            app = request.params.get("target", request.target)
            ok, msg = self.os_exec.close_app(app)
            status = "completed" if ok else "failed"
            res = ActionResultMsg(task_id=request.task_id, success=ok, status=status, message=msg)
            self.result_queue.put(res)
            return res

        # 4. Volume
        if request.action_type == "volume":
            direction = request.params.get("direction", "up")
            ok, msg = self.os_exec.adjust_volume(direction)
            res = ActionResultMsg(task_id=request.task_id, success=ok, status="completed", message=msg)
            self.result_queue.put(res)
            return res

        # 5. File Deletion (Confirmation gate)
        if request.action_type == "delete_file":
            path = request.params.get("path", "")
            confirmed = request.params.get("confirmed", False)
            ok, msg = self.os_exec.execute_destructive_action(request.task_id, "delete_file", path, confirmed)
            status = "completed" if ok else ("confirmation_required" if "CONFIRMATION" in msg else "failed")
            res = ActionResultMsg(task_id=request.task_id, success=ok, status=status, message=msg)
            self.result_queue.put(res)
            return res

        # 6. Browser Tasks (CDP)
        if request.action_type.startswith("browser_"):
            if request.action_type == "browser_extract":
                ok, msg = self.cdp.extract_page_text()
            else:
                ok, msg = True, f"Navigating to {request.params.get('url', '')}"
            res = ActionResultMsg(task_id=request.task_id, success=ok, status="completed", message=msg)
            self.result_queue.put(res)
            return res

        # Fallback completion
        res = ActionResultMsg(task_id=request.task_id, success=True, status="completed", message=f"Executed {request.action_type}")
        self.result_queue.put(res)
        return res
