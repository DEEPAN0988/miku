"""
eval/test_resource_fetcher.py — Unit Tests for Smart Resource Fetcher & Entity Classifier
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath("."))

from tools.resource_fetcher import (
    ResourceType,
    classify_resource_request,
    dispatch_resource_fetch,
    format_search_url_for_resource,
    monitor_download_completion,
    search_winget,
)


class TestResourceFetcher(unittest.TestCase):

    def test_entity_classifier(self):
        """Verify classification across applications, datasets, media, and documents."""
        # Application
        app_res = classify_resource_request("install vscode")
        self.assertEqual(app_res["category"], ResourceType.APPLICATION)
        self.assertIn("vscode", app_res["target_entity"].lower())

        app_res2 = classify_resource_request("download git installer")
        self.assertEqual(app_res2["category"], ResourceType.APPLICATION)

        # Dataset
        ds_res = classify_resource_request("download kaggle titanic dataset")
        self.assertEqual(ds_res["category"], ResourceType.DATASET)

        ds_res2 = classify_resource_request("fetch parquet corpus for llama")
        self.assertEqual(ds_res2["category"], ResourceType.DATASET)

        # Media
        media_res = classify_resource_request("download lofi beats video")
        self.assertEqual(media_res["category"], ResourceType.MEDIA)

        # Document
        doc_res = classify_resource_request("download attention is all you need paper pdf")
        self.assertEqual(doc_res["category"], ResourceType.DOCUMENT)

    def test_dispatch_application_dry_run(self):
        """Verify application query produces winget search in dry_run mode."""
        res = dispatch_resource_fetch("install vlc", dry_run=True)
        self.assertEqual(res["status"], "SIMULATED_APP_FETCH")
        self.assertEqual(res["action"], "winget_search")
        self.assertEqual(res["category"], "APPLICATION")
        self.assertIn("winget search", res["command"])

    def test_dispatch_application_with_mock_winget(self):
        """Verify application query parses winget matches properly when runner is mocked."""
        from unittest.mock import patch

        def mock_winget_runner(cmd):
            return {
                "status": "SUCCESS",
                "found": True,
                "package_name": "vlc",
                "packages": [
                    {"name": "VLC media player", "id": "VideoLAN.VLC", "version": "3.0.23", "source": "winget"}
                ],
                "raw_output": "VLC media player  VideoLAN.VLC  3.0.23  winget",
                "error": None,
            }

        with patch("tools.resource_fetcher.REAL_RESOURCE_FETCH_ENABLED", True):
            res = dispatch_resource_fetch("install vlc", dry_run=False, winget_runner=mock_winget_runner)
            self.assertEqual(res["status"], "WINGET_MATCH_FOUND")
            self.assertEqual(res["recommended_id"], "VideoLAN.VLC")
            self.assertIn("winget install --id VideoLAN.VLC", res["recommended_command"])

    def test_dispatch_application_circuit_breaker_fails_closed(self):
        """Verify that real non-dry-run dispatch is blocked by circuit breaker by default."""
        res = dispatch_resource_fetch("install vlc", dry_run=False)
        self.assertEqual(res["status"], "CIRCUIT_BREAKER_BLOCKED")
        self.assertFalse(res["executed"])

    def test_dispatch_dataset_formats_browser_query(self):
        """Verify dataset query formats browser search URL."""
        res = dispatch_resource_fetch("download common crawl dataset", dry_run=True)
        self.assertEqual(res["status"], "SIMULATED_BROWSER_SEARCH")
        self.assertEqual(res["category"], "DATASET")
        self.assertIn("common%20crawl", res["url"])
        self.assertIn("dataset download", res["search_query"])

    def test_dispatch_document_formats_browser_query(self):
        """Verify document query formats PDF-targeted search URL."""
        res = dispatch_resource_fetch("download transformer paper pdf", dry_run=True)
        self.assertEqual(res["status"], "SIMULATED_BROWSER_SEARCH")
        self.assertEqual(res["category"], "DOCUMENT")
        self.assertIn("filetype%3Apdf", res["url"])

    def test_download_monitor_from_partial(self):
        """Verify monitor detects .crdownload transitioning to completed file."""
        temp_dir = tempfile.mkdtemp()
        try:
            partial_path = os.path.join(temp_dir, "test_archive.zip.crdownload")
            final_path = os.path.join(temp_dir, "test_archive.zip")

            # Create partial file
            with open(partial_path, "wb") as f:
                f.write(b"in progress bytes...")

            start_t = time.time()

            # Schedule rename in 0.8s
            def finish_download():
                time.sleep(0.8)
                if os.path.exists(partial_path):
                    with open(final_path, "wb") as f:
                        f.write(b"complete download bytes 1234567890")
                    os.unlink(partial_path)

            th = threading.Thread(target=finish_download)
            th.start()

            res = monitor_download_completion(
                downloads_dir=temp_dir,
                timeout=5.0,
                poll_interval=0.2,
                start_time=start_t,
            )
            th.join()

            self.assertEqual(res["status"], "DOWNLOAD_COMPLETED")
            self.assertEqual(res["file_name"], "test_archive.zip")
            self.assertTrue(res["size_bytes"] > 0)
            self.assertTrue(res["from_partial"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_download_monitor_timeout(self):
        """Verify monitor times out gracefully when no download occurs."""
        temp_dir = tempfile.mkdtemp()
        try:
            res = monitor_download_completion(
                downloads_dir=temp_dir,
                timeout=1.0,
                poll_interval=0.2,
            )
            self.assertEqual(res["status"], "DOWNLOAD_TIMEOUT")
            self.assertTrue(res["timed_out"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
