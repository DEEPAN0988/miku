"""
eval/test_various_apps_suite.py — Diverse Cross-Category Application Lifecycle Test Suite

Tests a representative spectrum of applications across:
1. Games (Wuthering Waves / Legion Arena)
2. Web Browsers (Brave / Chrome / Edge)
3. Developer Tools (VS Code / Terminal)
4. Office & Productivity (Word / Excel / Notepad)
5. Modern Store & Media (Microsoft Store / Media Player)
6. Built-in Windows Utilities (Calculator / Paint)
"""

import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.app_launcher import launch_app, close_app

SAMPLE_DIVERSE_APPS = [
    {"category": "Game Launcher", "name": "wuthering waves"},
    {"category": "Gaming Hub", "name": "legion arena"},
    {"category": "Web Browser", "name": "chrome"},
    {"category": "Web Browser", "name": "msedge"},
    {"category": "Modern Store App", "name": "store"},
    {"category": "Developer Tool", "name": "code"},
    {"category": "Terminal", "name": "terminal"},
    {"category": "Productivity", "name": "notepad"},
    {"category": "Productivity", "name": "excel"},
    {"category": "Media Player", "name": "wmplayer"},
    {"category": "Windows Utility", "name": "calc"},
    {"category": "Windows Utility", "name": "paint"},
]

def run_suite():
    print("=" * 75)
    print("MIKU DIVERSE MULTI-CATEGORY APPLICATION LIFECYCLE TEST SUITE")
    print("=" * 75)
    print(f"{'Category':<20} | {'App Name':<16} | {'Launch Status':<12} | {'Mode':<20} | {'Close Status'}")
    print("-" * 75)

    results = []
    for item in SAMPLE_DIVERSE_APPS:
        cat = item["category"]
        name = item["name"]

        # 1. Launch
        t0 = time.time()
        launch_res = launch_app(name)
        latency = round((time.time() - t0) * 1000, 1)

        l_status = launch_res.get("status", "ERROR")
        mode = launch_res.get("mode", "UNKNOWN")
        pid = launch_res.get("pid")

        # 2. Stabilize so window/process renders
        time.sleep(2.0)

        # 3. Graceful Close
        close_res = close_app(name, pid=pid)
        c_status = close_res.get("status", "ERROR")

        print(f"{cat:<20} | {name:<16} | {l_status:<12} | {mode:<20} | {c_status}")
        results.append({
            "category": cat,
            "name": name,
            "launch": launch_res,
            "close": close_res,
            "latency": latency,
        })
        time.sleep(1.0)

    print("=" * 75)
    passed = sum(1 for r in results if r["launch"]["status"] in ("SUCCESS", "SHELL_PROTOCOL", "FALLBACK_DIRECT_ELEVATION") and r["close"]["status"] in ("SUCCESS", "NOT_FOUND"))
    print(f"[*] Total Categories Tested: {len(results)} | Passed: {passed}/{len(results)} ({round(passed/len(results)*100, 1)}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_suite()
