"""
eval/test_master_launcher.py — Test suite for MIKU Master Autonomous Controller
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import run_miku_master as rmm


class TestMasterLauncher(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    @patch("run_miku_master.run_diagnostics", return_value=True)
    def test_diag_mode(self, mock_diag):
        ret = rmm.run_master_controller(diag_mode=True)
        self.assertEqual(ret, 0)
        mock_diag.assert_called_once()

    @patch("run_miku_master.run_autonomous_task_loop")
    def test_task_mode_success(self, mock_loop):
        mock_loop.return_value = {"status": "success", "success": True, "steps_taken": 3}
        ret = rmm.run_master_controller(task_goal="open Notepad")
        self.assertEqual(ret, 0)
        mock_loop.assert_called_once_with("open Notepad", max_steps=10, real_execution=True)

    @patch("run_miku_master.run_batch_macro")
    def test_batch_mode(self, mock_batch):
        ret = rmm.run_master_controller(batch_cmds=["sys info", "sys ps"])
        self.assertEqual(ret, 0)
        mock_batch.assert_called_once_with(["sys info", "sys ps"])


if __name__ == "__main__":
    unittest.main()
