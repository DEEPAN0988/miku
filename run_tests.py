"""
Miku Test Runner & Metrics Dashboard.
Executes all unit, accuracy, regression, and adversarial test suites.
Reports against the metrics specified in Architecture v1.2 / Test & Debug Loop.
"""
import sys
import time
import unittest
import psutil
import os

def run_all_tests():
    print("=" * 70)
    print(" MIKU: SOVEREIGN LOCAL AGENT — TEST & DEBUG LOOP RUNNER")
    print("=" * 70)
    start_time = time.time()
    process = psutil.Process(os.getpid())

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed_s = time.time() - start_time
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" METRICS DASHBOARD (Target Tracking)")
    print("=" * 70)
    print(f" • Total Tests Run:             {result.testsRun}")
    print(f" • Failures:                    {len(result.failures)}")
    print(f" • Errors:                      {len(result.errors)}")
    print(f" • Peak Process RAM:            {ram_mb:.1f} MB (Target: < 500 MB)")
    print(f" • Total Test Elapsed:          {elapsed_s:.2f} s")
    print(f" • Memory Budget Cleared:       {'PASS' if ram_mb < 500 else 'FAIL'}")
    print(f" • Deterministic Test State:    {'PASS' if result.wasSuccessful() else 'FAIL'}")
    print("=" * 70)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
