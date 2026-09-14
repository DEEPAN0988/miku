"""
Unit tests for tools.resource_fetcher (WinGet wrapper and BrowserDownloadMonitor).
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from tools.resource_fetcher import (
    BrowserDownloadMonitor,
    parse_winget_search_table,
    sanitize_package_id,
    winget_install_package,
)


class TestResourceFetcher(unittest.TestCase):
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

    def test_sanitize_package_id(self):
        valid_ids = ["Git.Git", "Python.Python.3.11", "Microsoft.PowerToys", "7zip.7zip", "app-123_456"]
        for pkg in valid_ids:
            ok, val = sanitize_package_id(pkg)
            self.assertTrue(ok, f"Expected {pkg} to be valid.")
            self.assertEqual(val, pkg)

        invalid_ids = [
            "",
            "   ",
            "Git.Git; calc.exe",
            "foo & bar",
            "pkg|evil",
            "id with spaces",
            "id>output",
            "a" * 150,
        ]
        for pkg in invalid_ids:
            ok, _ = sanitize_package_id(pkg)
            self.assertFalse(ok, f"Expected {pkg} to be rejected.")

    def test_parse_winget_search_table(self):
        sample_output = """
Name                                                               Id                                             Version                Match                                     Source
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Git                                                                Git.Git                                        2.55.0.3                                                         winget
Microsoft Git                                                      Microsoft.Git                                  2.55.0.0.8                                                       winget
"""
        records = parse_winget_search_table(sample_output)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], "Git.Git")
        self.assertEqual(records[0]["version"], "2.55.0.3")
        self.assertEqual(records[1]["id"], "Microsoft.Git")

    def test_winget_install_dry_run_simulation(self):
        os.environ["MIKU_LIVE_EXECUTION"] = "false"
        result = winget_install_package("Git.Git", force_dry_run=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "SIMULATED_INSTALL")
        self.assertEqual(result["package_id"], "Git.Git")

    def test_winget_install_invalid_id_rejected(self):
        result = winget_install_package("Git.Git; calc.exe")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "INVALID_PACKAGE_ID")

    def test_winget_install_unauthorized(self):
        os.environ["MIKU_LIVE_EXECUTION"] = "true"
        os.environ["MIKU_AUTONOMOUS_MODE"] = "false"

        with patch("sys.stdin.isatty", return_value=False):
            result = winget_install_package("Git.Git")
            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "ABORT_UNAUTHORIZED")

    def test_browser_download_monitor_resolution(self):
        temp_dir = tempfile.mkdtemp()
        completed_events = []

        def callback(filepath: str, size: int):
            completed_events.append((filepath, size))

        try:
            monitor = BrowserDownloadMonitor(
                download_dir=temp_dir,
                poll_interval=0.1,
                on_download_complete=callback,
            )

            # Test target filename resolution helper
            p_cr = Path(temp_dir) / "document.pdf.crdownload"
            p_part = Path(temp_dir) / "image.png.part"
            self.assertEqual(BrowserDownloadMonitor.resolve_target_filename(p_cr).name, "document.pdf")
            self.assertEqual(BrowserDownloadMonitor.resolve_target_filename(p_part).name, "image.png")

            # 1. Step 1: Active download starts (.crdownload file created)
            p_cr.write_bytes(b"partial byte data")

            # Poll once to register active download
            finished = monitor.poll_once()
            self.assertEqual(len(finished), 0)
            self.assertIn(str(p_cr.resolve()), monitor._active_downloads)

            # 2. Step 2: Download completes (.crdownload renamed to target file)
            final_target = Path(temp_dir) / "document.pdf"
            p_cr.unlink()
            final_target.write_bytes(b"complete final pdf content 1234567890")

            # Poll again to resolve completed download
            finished = monitor.poll_once()
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["filename"], "document.pdf")
            self.assertEqual(finished[0]["size_bytes"], len(b"complete final pdf content 1234567890"))
            self.assertEqual(len(completed_events), 1)
            self.assertEqual(completed_events[0][0], str(final_target))

            # Active downloads tracking should be cleared
            self.assertNotIn(str(p_cr.resolve()), monitor._active_downloads)

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
