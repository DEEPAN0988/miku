"""
Unit tests for tools.app_launcher (Multi-tier Application Launch Resilience & Fallback Loop).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.app_launcher import (
    find_direct_game_binary,
    _verify_process_alive,
    launch_app,
    resolve_app_path,
)


class TestAppLauncherResilience(unittest.TestCase):
    def test_find_direct_game_binary(self):
        # Create a mock game directory tree
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher = os.path.join(temp_dir, "launcher.exe")
            with open(launcher, "w") as f:
                f.write("mock launcher")

            game_dir = os.path.join(temp_dir, "My Game")
            os.makedirs(game_dir, exist_ok=True)
            game_exe = os.path.join(game_dir, "My Game.exe")
            with open(game_exe, "w") as f:
                f.write("mock game")

            detected = find_direct_game_binary(launcher)
            self.assertIsNotNone(detected)
            self.assertEqual(os.path.abspath(detected), os.path.abspath(game_exe))

    def test_verify_process_alive_returns_false_for_nonexistent_process(self):
        self.assertFalse(_verify_process_alive("fake_nonexistent_proc_99999.exe", min_duration=0.1))

    def test_launch_app_success_direct_startfile(self):
        with patch("tools.app_launcher.resolve_app_path", return_value=("notepad", "notepad.exe")):
            with patch("os.startfile", return_value=None):
                with patch("tools.app_launcher._verify_process_alive", return_value=True):
                    res = launch_app("notepad")
                    self.assertEqual(res["status"], "SUCCESS")
                    self.assertEqual(res["mode"], "DIRECT_STARTFILE")

    def test_launch_app_tier2_runasinvoker_fallback(self):
        real_exe = os.path.abspath(sys.executable)

        # Simulate Tier 1 startfile failing with WinError 740 (Elevation Required)
        err_740 = OSError()
        err_740.winerror = 740

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None

        with patch("tools.app_launcher.resolve_app_path", return_value=("elevated_tool", real_exe)):
            with patch("os.startfile", side_effect=err_740):
                with patch("subprocess.Popen", return_value=mock_proc):
                    with patch("tools.app_launcher._verify_process_alive", return_value=True):
                        res = launch_app("elevated_tool")
                        self.assertEqual(res["status"], "SUCCESS")
                        self.assertEqual(res["mode"], "COMPAT_RUNASINVOKER")
                        self.assertEqual(res["pid"], 99999)

    def test_launch_app_tier3_elevated_bridge_fallback(self):
        real_exe = os.path.abspath(sys.executable)

        err_740 = OSError()
        err_740.winerror = 740

        # Simulate Tier 1 and Tier 2 both failing
        with patch("tools.app_launcher.resolve_app_path", return_value=("strict_admin_app", real_exe)):
            with patch("os.startfile", side_effect=err_740):
                with patch("tools.app_launcher._verify_process_alive", return_value=False):
                    with patch("tools.system_dispatcher.launch_elevated_app", return_value={
                        "status": "SUCCESS",
                        "mode": "TASK_SCHEDULER",
                        "executed": True,
                        "task_name": "Miku_Elevated_strict_admin_app",
                    }) as mock_elevated:
                        res = launch_app("strict_admin_app")
                        self.assertEqual(res["status"], "SUCCESS")
                        self.assertEqual(res["mode"], "TASK_SCHEDULER")
                        mock_elevated.assert_called_once()


if __name__ == "__main__":
    unittest.main()
