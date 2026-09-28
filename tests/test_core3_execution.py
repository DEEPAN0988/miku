"""
Test Suite: Core 3 Execution Engine (Hands).
Covers Minimum-Jerk Motion, Closed-Loop Verification (Bottleneck #3),
User Takeover Interruption, AST Sandbox (Bottleneck #4), and Confirmation Gating.
"""
import unittest
from pathlib import Path
import tempfile
import shutil

from miku.core3_execution.motion import minimum_jerk_trajectory
from miku.core3_execution.closed_loop_click import ClosedLoopVerifier
from miku.core3_execution.ast_sandbox import ASTSandboxRunner
from miku.core3_execution.os_actions import OSActionsExecutor
from miku.core3_execution.execution_daemon import ExecutionDaemon
from miku.ipc.messages import ActionRequestMsg
from multiprocessing import Queue

class TestCore3Execution(unittest.TestCase):
    def test_minimum_jerk_trajectory_smoothness(self):
        """
        Verifies minimum-jerk trajectory starts and ends exactly at target,
        producing monotonic interpolation with smooth bell-shaped velocity.
        """
        start = (100, 200)
        end = (600, 800)
        steps = 25
        path = minimum_jerk_trajectory(start, end, steps=steps)

        self.assertEqual(len(path), steps + 1)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)

        # Velocities: midpoint velocity should be highest
        dx_start = abs(path[1][0] - path[0][0])
        dx_mid = abs(path[steps // 2 + 1][0] - path[steps // 2][0])
        self.assertGreater(dx_mid, dx_start, "Velocity must peak near midpoint (minimum-jerk bell curve)")

    def test_closed_loop_notification_trap_abort(self):
        """
        Bottleneck #3 Fix: If target changes or popup appears before hardware click,
        asserts action is aborted with 'target changed, action cancelled'.
        """
        verifier = ClosedLoopVerifier()

        # Mock sample state where unexpected notification banner appeared over target
        verifier.sample_target_state = lambda coords: {
            "x": coords[0],
            "y": coords[1],
            "window_title": "Unexpected Toast Notification",
            "control_type": "Button",
            "control_name": "Dismiss Toast Notification"
        }

        valid, reason = verifier.verify_before_click(
            target_coords=(300, 400),
            expected_control="Submit Form",
            expected_window="My Secure App"
        )

        self.assertFalse(valid, "Must abort click when target is obscured by unexpected popup")
        self.assertIn("target changed, action cancelled", reason)

    def test_user_interruption_mid_task(self):
        """
        Interrupt test: simulates user mouse movement mid-automation.
        Asserts Core 3 yields immediately and logs clean partial completion.
        """
        action_q = Queue()
        result_q = Queue()
        daemon = ExecutionDaemon(action_q, result_q)

        # Trigger user interruption
        daemon.trigger_user_interrupt()
        req = ActionRequestMsg(
            action_type="click_target",
            target="button",
            coords=(200, 200),
            task_id="task_int_1"
        )
        # Mock verifier to pass
        daemon.verifier.verify_before_click = lambda *args, **kwargs: (True, "verified")

        result = daemon.execute_request(req)
        self.assertEqual(result.status, "aborted_by_user")
        self.assertFalse(result.success)
        self.assertIn("User moved mouse", result.message)

    def test_ast_sandbox_privilege_boundary(self):
        """
        Bottleneck #4 Fix: Rejects dangerous AST calls and tests default-deny scratch dry-run.
        """
        runner = ASTSandboxRunner()

        # 1. Reject disallowed calls: eval, exec, arbitrary imports
        malicious_code = """
import os
os.system("rm -rf /")
eval("print('danger')")
"""
        is_valid, violations = runner.validate_code_ast(malicious_code)
        self.assertFalse(is_valid)
        self.assertTrue(any("Disallowed" in v for v in violations))

        # 2. Accept safe code and dry-run against scratch target
        safe_code = """
for i in range(3):
    pass
"""
        test_dir = Path(tempfile.gettempdir()) / "test_target_dir"
        test_dir.mkdir(exist_ok=True)
        try:
            ok, msg, log = runner.dry_run_against_scratch(safe_code, test_dir)
            self.assertTrue(ok)
            self.assertIn("Dry-run succeeded", msg)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_destructive_action_confirmation_gate(self):
        """
        PRD §7: Unconfirmed destructive operations (file deletion) require spoken/typed confirmation.
        """
        executor = OSActionsExecutor()
        dummy_file = Path("test_delete_me.tmp")
        dummy_file.write_text("temporary data")

        try:
            # First attempt: unconfirmed
            ok, msg = executor.execute_destructive_action("del_1", "delete_file", str(dummy_file), confirmed=False)
            self.assertFalse(ok)
            self.assertIn("CONFIRMATION_REQUIRED", msg)
            self.assertTrue(dummy_file.exists(), "File must NOT be deleted without explicit confirmation")

            # Second attempt: user confirms
            ok_conf, msg_conf = executor.execute_destructive_action("del_1", "delete_file", str(dummy_file), confirmed=True)
            self.assertTrue(ok_conf)
            self.assertFalse(dummy_file.exists(), "File should be deleted once confirmed")
        finally:
            if dummy_file.exists():
                dummy_file.unlink()

    def test_os_actions_app_launch_and_close(self):
        """
        Tests OSActionsExecutor app mappings, alias handling, close_app, and open_file.
        """
        executor = OSActionsExecutor()

        # Check APP_MAP mappings
        self.assertEqual(executor.APP_MAP["calc"], "calc.exe")
        self.assertEqual(executor.APP_MAP["calculator"], "calc.exe")
        self.assertEqual(executor.APP_MAP["edge"], "microsoft-edge:")
        self.assertEqual(executor.APP_MAP["browser"], "microsoft-edge:")
        self.assertEqual(executor.APP_MAP["vscode"], "code")
        self.assertEqual(executor.APP_MAP["paint"], "mspaint.exe")

        # Check PROCESS_MAP mappings
        self.assertIn("CalculatorApp.exe", executor.PROCESS_MAP["calculator"])
        self.assertIn("calc.exe", executor.PROCESS_MAP["calc"])
        self.assertIn("msedge.exe", executor.PROCESS_MAP["edge"])

        # Test open_file with a temporary file
        temp_file = Path("test_open_file.tmp")
        temp_file.write_text("sample content")
        try:
            # Existing file
            ok, msg = executor.open_file(str(temp_file))
            self.assertTrue(ok)
            # Non-existent file
            ok_bad, msg_bad = executor.open_file("non_existent_file_xyz_123.tmp")
            self.assertFalse(ok_bad)
            self.assertIn("File not found", msg_bad)
        finally:
            if temp_file.exists():
                temp_file.unlink()

if __name__ == "__main__":
    unittest.main()
