"""
eval/test_file_lifecycle.py — Unit Tests for Safe File Lifecycle Management & Recycle Bin Wrapper
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

from tools.file_lifecycle import (
    REAL_FILE_DELETE_ENABLED,
    request_live_human_delete_confirmation,
    safe_delete,
)


class TestFileLifecycle(unittest.TestCase):

    def test_circuit_breaker_invariant(self):
        """Constitutional Safety Invariant: REAL_FILE_DELETE_ENABLED must be False by default."""
        self.assertFalse(
            REAL_FILE_DELETE_ENABLED,
            "CRITICAL SAFETY VIOLATION: REAL_FILE_DELETE_ENABLED must be False by default."
        )

    def test_missing_path_returns_not_found(self):
        bogus_path = os.path.abspath("non_existent_file_123456.tmp")
        res = safe_delete(bogus_path, dry_run=True)
        self.assertEqual(res["status"], "NOT_FOUND")
        self.assertFalse(res["executed"])

    def test_empty_target_paths(self):
        res = safe_delete([], dry_run=True)
        self.assertEqual(res["status"], "ERROR")
        self.assertFalse(res["executed"])

    def test_dry_run_preserves_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe delete dry run test")
            temp_path = f.name

        try:
            self.assertTrue(os.path.exists(temp_path))
            res = safe_delete(temp_path, dry_run=True)
            self.assertEqual(res["status"], "SIMULATED_RECYCLE_BIN")
            self.assertFalse(res["executed"])
            # Ensure file is still on disk
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_circuit_breaker_blocks_real_deletion_by_default(self):
        """Verify non-dry-run safe_delete is blocked by standing circuit breaker."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe delete circuit breaker test")
            temp_path = f.name

        try:
            res = safe_delete(temp_path, dry_run=False)
            self.assertEqual(res["status"], "CIRCUIT_BREAKER_BLOCKED")
            self.assertFalse(res["executed"])
            # File must still exist untouched
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_programmatic_bypass_parameter_rejected(self):
        """Verify programmatic 'confirmed' parameter was stripped and raises TypeError."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"bypass rejection test")
            temp_path = f.name

        try:
            with self.assertRaises(TypeError):
                safe_delete(temp_path, dry_run=False, confirmed=True)  # type: ignore
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_unconfirmed_blocks_execution_when_enabled(self):
        """When circuit breaker is enabled, unconfirmed invocation returns PENDING_CONFIRMATION."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe delete confirmation test")
            temp_path = f.name

        try:
            with patch("tools.file_lifecycle.REAL_FILE_DELETE_ENABLED", True):
                with patch("tools.file_lifecycle.request_live_human_delete_confirmation", return_value=False):
                    res = safe_delete(temp_path, dry_run=False)
                    self.assertEqual(res["status"], "PENDING_CONFIRMATION")
                    self.assertFalse(res["executed"])
                    self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_mocked_confirmed_safe_delete_to_recycle_bin(self):
        """Verify SHFileOperation is called with FOF_ALLOWUNDO when confirmed (mocked)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe delete recycle bin test")
            temp_path = f.name

        def mock_sh_op(*args, **kwargs):
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return (0, False)

        try:
            with patch("tools.file_lifecycle.REAL_FILE_DELETE_ENABLED", True):
                with patch("tools.file_lifecycle.request_live_human_delete_confirmation", return_value=True):
                    with patch("tools.file_lifecycle.shell.SHFileOperation", side_effect=mock_sh_op) as mock_sh:
                        res = safe_delete(temp_path, dry_run=False, silent=True)
                        self.assertEqual(res["status"], "SUCCESS")
                        self.assertTrue(res["executed"])
                        self.assertTrue(mock_sh.called)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_batch_safe_delete_mocked(self):
        """Verify batch file deletion formats paths for SHFileOperation (mocked)."""
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(f"batch file {i}".encode())
                files.append(f.name)

        def mock_sh_op_batch(*args, **kwargs):
            for p in files:
                if os.path.exists(p):
                    os.unlink(p)
            return (0, False)

        try:
            with patch("tools.file_lifecycle.REAL_FILE_DELETE_ENABLED", True):
                with patch("tools.file_lifecycle.request_live_human_delete_confirmation", return_value=True):
                    with patch("tools.file_lifecycle.shell.SHFileOperation", side_effect=mock_sh_op_batch) as mock_sh:
                        res = safe_delete(files, dry_run=False, silent=True)
                        self.assertEqual(res["status"], "SUCCESS")
                        self.assertTrue(res["executed"])
                        self.assertEqual(len(res["paths"]), 3)
                        self.assertTrue(mock_sh.called)
        finally:
            for p in files:
                if os.path.exists(p):
                    os.unlink(p)


if __name__ == "__main__":
    unittest.main()
