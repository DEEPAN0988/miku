"""
Unit tests for tools.file_lifecycle (Safe File Lifecycle Management).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from tools.file_lifecycle import (
    format_double_null_terminated_path,
    request_live_human_delete_confirmation,
    safe_recycle_file,
)


class TestFileLifecycle(unittest.TestCase):
    def setUp(self):
        self.orig_live = os.environ.get("MIKU_LIVE_EXECUTION")
        self.orig_auto = os.environ.get("MIKU_AUTONOMOUS_MODE")

    def tearDown(self):
        if self.orig_live is not None:
            os.environ["MIKU_LIVE_EXECUTION"] = self.orig_live
        else:
            os.environ.pop("MIKU_LIVE_EXECUTION", None)

        if self.orig_auto is not None:
            os.environ["MIKU_AUTONOMOUS_MODE"] = self.orig_auto
        else:
            os.environ.pop("MIKU_AUTONOMOUS_MODE", None)

    def test_format_double_null_terminated_path(self):
        path = "relative/test.txt"
        res = format_double_null_terminated_path(path)
        self.assertTrue(res.endswith("\0\0"))
        self.assertTrue(os.path.isabs(res.rstrip("\0")))

    def test_file_not_found(self):
        fake_path = r"C:\path\to\nonexistent\random_file_987654.tmp"
        result = safe_recycle_file(fake_path)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "FILE_NOT_FOUND")

    def test_dry_run_simulation_preserves_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe simulation test")
            temp_path = f.name

        try:
            os.environ["MIKU_LIVE_EXECUTION"] = "false"
            os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

            result = safe_recycle_file(temp_path, force_dry_run=True)
            self.assertTrue(result["success"])
            self.assertEqual(result["status"], "SIMULATED_RECYCLE")
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_safety_gate_unauthorized_in_non_interactive_env(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"unauthorized test")
            temp_path = f.name

        try:
            # Enable live execution but disable autonomous mode
            os.environ["MIKU_LIVE_EXECUTION"] = "true"
            os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

            with patch("sys.stdin.isatty", return_value=False):
                result = safe_recycle_file(temp_path)
                self.assertFalse(result["success"])
                self.assertEqual(result["status"], "ABORT_UNAUTHORIZED")
                self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_live_recycle_with_autonomous_authorization(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"recycle bin live test")
            temp_path = f.name

        try:
            # Enable both flags to allow autonomous execution
            os.environ["MIKU_LIVE_EXECUTION"] = "true"
            os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

            result = safe_recycle_file(temp_path)
            self.assertTrue(result["success"])
            self.assertEqual(result["status"], "RECYCLED_SUCCESS")
            # Verify file was moved out of original location to Recycle Bin
            self.assertFalse(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
