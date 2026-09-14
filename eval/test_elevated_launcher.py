"""
eval/test_elevated_launcher.py — Comprehensive Unit Tests for Elevated Task Scheduler Launcher
"""

import os
import sys
import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.system_dispatcher import (
    _sanitize_app_key,
    _validate_path_security,
    check_elevated_task_exists,
    launch_elevated_app,
    register_elevated_task,
)


class DummyCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestElevatedLauncher(unittest.TestCase):

    def test_sanitize_app_key(self):
        """Verify normalization of app names into safe Task Scheduler identifiers."""
        self.assertEqual(_sanitize_app_key("Wuthering Waves"), "wuthering_waves")
        self.assertEqual(_sanitize_app_key("Visual Studio Code!"), "visual_studio_code")
        self.assertEqual(_sanitize_app_key("App--Name__Test"), "app_name_test")
        self.assertEqual(_sanitize_app_key(""), "app")
        self.assertEqual(_sanitize_app_key("  "), "app")
        # Length constraint
        long_name = "a" * 100
        self.assertLessEqual(len(_sanitize_app_key(long_name)), 48)

    def test_validate_path_security(self):
        """Verify rejection of command injection characters in file paths."""
        valid, clean = _validate_path_security(r"C:\Program Files\App\game.exe")
        self.assertTrue(valid)
        self.assertEqual(clean, r"C:\Program Files\App\game.exe")

        # Rejects injection characters
        for bad_char in [";", "&", "|", "`", "$", "\x00", "\n"]:
            v, _ = _validate_path_security(f"C:\\test{bad_char}inject.exe")
            self.assertFalse(v, f"Failed to reject character: {repr(bad_char)}")

        # Rejects empty
        v, _ = _validate_path_security("")
        self.assertFalse(v)

    def test_check_elevated_task_exists(self):
        """Verify task existence query parser with mocked runner."""
        def mock_runner_found(cmd):
            self.assertEqual(cmd, ["schtasks", "/query", "/tn", "Miku_Elevated_test_app"])
            return DummyCompletedProcess(0, stdout="Folder: \\ \nTaskName: Miku_Elevated_test_app")

        def mock_runner_missing(cmd):
            return DummyCompletedProcess(1, stdout="", stderr="ERROR: The system cannot find the file specified.")

        self.assertTrue(check_elevated_task_exists("Miku_Elevated_test_app", runner=mock_runner_found))
        self.assertFalse(check_elevated_task_exists("Miku_Elevated_test_app", runner=mock_runner_missing))
        self.assertFalse(check_elevated_task_exists("", runner=mock_runner_found))

    def test_register_elevated_task_not_found(self):
        """Rejects non-existent executable files before calling shell."""
        res = register_elevated_task("Miku_Elevated_bogus", r"C:\fake\non_existent_exe_123.exe")
        self.assertEqual(res["status"], "NOT_FOUND")
        self.assertFalse(res["registered"])

    def test_register_elevated_task_success(self):
        """Verify ShellExecute invocation with 'runas' and proper PowerShell arguments."""
        recorded_calls = []

        def mock_shell(hwnd, verb, file, params, directory, show):
            recorded_calls.append({
                "verb": verb,
                "file": file,
                "params": params,
                "directory": directory,
            })
            return 42  # Success code > 32

        # Use an existing system executable for validation
        real_exe = os.path.abspath(sys.executable)
        real_dir = os.path.dirname(real_exe)

        def mock_check(cmd):
            return DummyCompletedProcess(0, stdout="Miku_Elevated_python")

        res = register_elevated_task(
            task_name="Miku_Elevated_python",
            executable_path=real_exe,
            working_dir=real_dir,
            shell_executor=mock_shell,
            wait_timeout_sec=1.0,
            check_runner=mock_check,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["registered"])
        self.assertEqual(len(recorded_calls), 1)
        call = recorded_calls[0]
        self.assertEqual(call["verb"], "runas")
        self.assertEqual(call["file"], "powershell.exe")
        self.assertIn("-RunLevel Highest", call["params"])
        self.assertIn("-LogonType Interactive", call["params"])
        self.assertIn(real_exe, call["params"])

    def test_register_elevated_task_uac_denied(self):
        """Verify handling when user clicks 'No' on the UAC prompt (code <= 32)."""
        def mock_shell_denied(*args):
            return 5  # ERROR_ACCESS_DENIED

        real_exe = os.path.abspath(sys.executable)
        res = register_elevated_task(
            task_name="Miku_Elevated_denied",
            executable_path=real_exe,
            shell_executor=mock_shell_denied,
            wait_timeout_sec=0.1,
        )
        self.assertEqual(res["status"], "UAC_DENIED")
        self.assertFalse(res["registered"])

    def test_launch_elevated_app_when_task_exists(self):
        """When Task Scheduler task exists, runs silently with zero UAC prompts."""
        run_cmds = []

        def mock_task_runner(cmd):
            run_cmds.append(cmd)
            if cmd[1] == "/query":
                return DummyCompletedProcess(0, stdout="Miku_Elevated_wuthering_waves")
            elif cmd[1] == "/run":
                return DummyCompletedProcess(0, stdout="SUCCESS: Attempted to run the scheduled task.")
            return DummyCompletedProcess(1)

        real_exe = os.path.abspath(sys.executable)

        with patch("tools.app_launcher.resolve_app_path", return_value=("wuthering waves", real_exe)):
            res = launch_elevated_app(
                "wuthering waves",
                task_runner=mock_task_runner,
            )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "TASK_SCHEDULER")
        self.assertTrue(res["executed"])
        self.assertEqual(res["task_name"], "Miku_Elevated_wuthering_waves")
        self.assertEqual(run_cmds[-1], ["schtasks", "/run", "/tn", "Miku_Elevated_wuthering_waves"])

    def test_launch_elevated_app_auto_registration_flow(self):
        """When task is missing, auto-registers and then executes."""
        task_state = {"exists": False}

        def mock_task_runner(cmd):
            if cmd[1] == "/query":
                if task_state["exists"]:
                    return DummyCompletedProcess(0, stdout="Miku_Elevated_test_app")
                return DummyCompletedProcess(1, stderr="Not found")
            elif cmd[1] == "/run":
                return DummyCompletedProcess(0, stdout="SUCCESS")
            return DummyCompletedProcess(1)

        def mock_reg_fn(**kwargs):
            task_state["exists"] = True
            return {"status": "SUCCESS", "registered": True}

        real_exe = os.path.abspath(sys.executable)

        with patch("tools.app_launcher.resolve_app_path", return_value=("test app", real_exe)):
            res = launch_elevated_app(
                "test app",
                auto_register=True,
                task_runner=mock_task_runner,
                register_fn=mock_reg_fn,
            )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["mode"], "TASK_SCHEDULER_REGISTERED_AND_RUN")
        self.assertTrue(res["executed"])

    def test_launch_elevated_app_fallback_to_direct_runas(self):
        """When Task Scheduler registration fails or is skipped, falls back to direct ShellExecuteW runas."""
        shell_calls = []

        def mock_shell(hwnd, verb, file, params, directory, show):
            shell_calls.append({"verb": verb, "file": file})
            return 42

        def mock_runner_missing(cmd):
            return DummyCompletedProcess(1, stderr="Not found")

        real_exe = os.path.abspath(sys.executable)

        with patch("tools.app_launcher.resolve_app_path", return_value=("custom app", real_exe)):
            res = launch_elevated_app(
                "custom app",
                auto_register=False,  # Skip auto registration to test fallback
                task_runner=mock_runner_missing,
                shell_executor=mock_shell,
            )

        self.assertEqual(res["status"], "FALLBACK_DIRECT_ELEVATION")
        self.assertEqual(res["mode"], "DIRECT_RUNAS")
        self.assertTrue(res["executed"])
        self.assertEqual(len(shell_calls), 1)
        self.assertEqual(shell_calls[0]["verb"], "runas")
        self.assertEqual(shell_calls[0]["file"], real_exe)


if __name__ == "__main__":
    unittest.main()
