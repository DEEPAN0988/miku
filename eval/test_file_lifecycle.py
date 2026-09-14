"""
eval/test_file_lifecycle.py — Unit Tests for Safe File Lifecycle Management & Recycle Bin Wrapper
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath("."))

from tools.file_lifecycle import safe_delete


class TestFileLifecycle(unittest.TestCase):

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

    def test_unconfirmed_blocks_execution(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe delete confirmation test")
            temp_path = f.name

        try:
            res = safe_delete(temp_path, dry_run=False, confirmed=False)
            self.assertEqual(res["status"], "PENDING_CONFIRMATION")
            self.assertFalse(res["executed"])
            # Ensure file was not deleted
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_confirmed_safe_delete_to_recycle_bin(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"safe delete recycle bin test")
            temp_path = f.name

        self.assertTrue(os.path.exists(temp_path))
        res = safe_delete(temp_path, dry_run=False, confirmed=True, silent=True)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["executed"])
        # File must no longer exist at original path (it was moved to the Recycle Bin)
        self.assertFalse(os.path.exists(temp_path))

    def test_batch_safe_delete(self):
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(f"batch file {i}".encode())
                files.append(f.name)

        for p in files:
            self.assertTrue(os.path.exists(p))

        res = safe_delete(files, dry_run=False, confirmed=True, silent=True)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["executed"])
        for p in files:
            self.assertFalse(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()
