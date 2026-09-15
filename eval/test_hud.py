"""
eval/test_hud.py — Unit test suite for MIKU Desktop HUD Overlay
"""

import os
import unittest
from unittest.mock import patch

import miku_hud as hud


class TestMikuHUD(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_hud_update_status(self):
        h = hud.MikuHUD(enabled=False)
        h.update_status("Testing HUD", "Step 1/1")
        self.assertEqual(h.status_text, "Testing HUD")
        self.assertEqual(h.step_info, "Step 1/1")
        h.close()

    def test_global_hud_helper(self):
        hud.update_miku_hud("Global Status Test", "Step 2")
        gh = hud.get_global_hud()
        self.assertEqual(gh.status_text, "Global Status Test")
        self.assertEqual(gh.step_info, "Step 2")


if __name__ == "__main__":
    unittest.main()
