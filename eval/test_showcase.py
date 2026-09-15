"""
eval/test_showcase.py — Test suite for MIKU Capability Showcase
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import run_miku_showcase as rms


import tools.screen_inspector as si
import tools.typing_automation as ta


class TestShowcase(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        self._old_click = si.REAL_CLICK_ENABLED
        self._old_type = ta.REAL_TYPE_ENABLED

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)
        si.REAL_CLICK_ENABLED = self._old_click
        ta.REAL_TYPE_ENABLED = self._old_type

    @patch("run_miku_showcase.run_diagnostics", return_value=True)
    @patch("run_miku_showcase.generate_response", return_value="Sample poem text")
    @patch("run_miku_showcase.asyncio.run")
    def test_showcase_execution(self, mock_asyncio, mock_gen, mock_diag):
        mock_asyncio.side_effect = lambda coro: coro.close()
        rms.showcase()
        mock_diag.assert_called_once()
        mock_gen.assert_called_once()
        mock_asyncio.assert_called_once()


if __name__ == "__main__":
    unittest.main()
