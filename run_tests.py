"""
Custom test runner for Miku Assistant using Python standard library unittest.
"""

import unittest
import sys
import os

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(os.path.dirname(__file__), "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
