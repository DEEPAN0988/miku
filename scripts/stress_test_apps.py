"""
scripts/stress_test_apps.py — Continuous Multi-App Opening & Resilience Testing Harness

Iterates through all discoverable installed Windows applications on the machine,
tests opening each application via Miku's multi-tier launch loop, validates process
persistence, safely cleans up test-spawned instances, and logs results to a markdown report.

Can run for a specified duration (e.g. 1 hour) in a continuous loop.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Set

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.app_launcher import (

    BUILTIN_APPS,
    launch_app,
    resolve_app_path,
    scan_registry_app_paths,
    scan_start_menu_shortcuts,
)

PROTECTED_PROCESSES = {
    "antigravity",
    "antigravity ide",
    "antigravity ide.exe",
    "explorer.exe",
    "explorer",
    "python.exe",
    "python",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "taskmgr.exe",
    "conhost.exe",
    "system",
    "svchost.exe",
    "services.exe",
    "lsass.exe",
    "csrss.exe",
    "smss.exe",
    "winlogon.exe",
    "dwm.exe",
}

EXCLUDED_APP_KEYWORDS = [
    "uninstall",
    "remove",
    "help",
    "manual",
    "website",
    "documentation",
    "readme",
    "setup",
    "diagnostics",
    "repair",
]


def get_all_testable_apps() -> List[str]:
    """Collects and deduplicates all valid application names on the host."""
    sm = scan_start_menu_shortcuts()
    rp = scan_registry_app_paths()

    unique: Set[str] = set()

    for name in BUILTIN_APPS.keys():
        unique.add(name)

    for name in sm.keys():
        clean = name.strip().lower()
        if not any(k in clean for k in EXCLUDED_APP_KEYWORDS):
            unique.add(name)

    for name in rp.keys():
        clean = name.strip().lower()
        if not any(k in clean for k in EXCLUDED_APP_KEYWORDS):
            unique.add(name)

    # Exclude protected IDE and critical processes from direct launch spam
    filtered = [
        app for app in sorted(unique)
        if app.lower() not in ("antigravity", "antigravity ide")
    ]
    return filtered


def get_running_process_map() -> Dict[int, str]:
    """Queries tasklist for all running PIDs and their image names."""
    import csv
    import io
    proc_map: Dict[int, str] = {}
    try:
        res = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=3)
        reader = csv.reader(io.StringIO(res.stdout))
        for row in reader:
            if len(row) >= 2 and row[1].isdigit():
                proc_map[int(row[1])] = row[0].strip().lower()
    except Exception:
        pass
    return proc_map


def terminate_spawned_process(pid: int, image_name: str = ""):
    """Safely terminates a test-spawned process if not protected."""
    if pid <= 4:
        return
    clean_img = image_name.lower()
    if any(p in clean_img for p in PROTECTED_PROCESSES):
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=3)
    except Exception:
        pass


def write_markdown_report(report_path: str, stats: Dict[str, Any], history: List[Dict[str, Any]]):
    """Generates an audit report detailing launch outcomes and success metrics."""
    duration_min = round(stats.get("elapsed_sec", 0) / 60.0, 1)
    success_count = stats.get("success", 0)
    total_count = stats.get("total", 0)
    rate = round((success_count / total_count * 100.0), 1) if total_count > 0 else 0.0

    lines = [
        "# Miku Application Launch Resilience & Stress Test Report",
        "",
        f"- **Start Time**: {stats.get('start_time')}",
        f"- **Elapsed Time**: {duration_min} minutes ({stats.get('elapsed_sec', 0)} seconds)",
        f"- **Rounds Completed**: {stats.get('rounds', 1)}",
        f"- **Total Tests Executed**: {total_count}",
        f"- **Successful Launches**: {success_count}",
        f"- **Failed Launches**: {stats.get('failed', 0)}",
        f"- **Success Rate**: `{rate}%`",
        "",
        "## Recent Test Log (Last 40 Invocations)",
        "",
        "| Time | Application Name | Mode | Status | PID | Details |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in history[-40:]:
        t_str = item.get("time", "")
        name = item.get("app", "")
        mode = item.get("mode", "N/A")
        status = "**SUCCESS**" if item.get("status") == "SUCCESS" else "<span style='color:red;'>FAILED</span>"
        pid = str(item.get("pid", "-"))
        output = item.get("output", "").replace("|", "-")
        if len(output) > 60:
            output = output[:57] + "..."
        lines.append(f"| {t_str} | `{name}` | `{mode}` | {status} | {pid} | {output} |")

    lines.append("")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"[!] Failed to write markdown report: {e}", flush=True)


def run_stress_test(
    duration_seconds: float = 3600.0,
    interval_seconds: float = 2.0,
    report_path: str = "eval/app_launch_stress_report.md",
    max_apps: int = 0,
):
    apps = get_all_testable_apps()
    if max_apps > 0:
        apps = apps[:max_apps]

    print("=" * 70, flush=True)
    print(f"[*] MIKU APP LAUNCH RESILIENCE & STRESS TEST HARNESS", flush=True)
    print(f"[*] Target Duration: {duration_seconds}s ({round(duration_seconds / 3600.0, 2)} hours)", flush=True)
    print(f"[*] Total Discovered Applications: {len(apps)}", flush=True)
    print(f"[*] Report Destination: {os.path.abspath(report_path)}", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    stats = {
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": 0,
        "success": 0,
        "failed": 0,
        "rounds": 0,
        "elapsed_sec": 0,
    }
    history: List[Dict[str, Any]] = []

    round_idx = 0
    try:
        while (time.time() - start_time) < duration_seconds:
            round_idx += 1
            stats["rounds"] = round_idx
            print(f"\n[*] --- STARTING ROUND {round_idx} (Elapsed: {int(time.time() - start_time)}s) ---", flush=True)

            for idx, app_name in enumerate(apps, 1):
                if (time.time() - start_time) >= duration_seconds:
                    print("[*] Target test duration reached. Concluding test run.", flush=True)
                    break

                # Snapshot processes before launch
                pre_map = get_running_process_map()

                # Attempt launch via resilient multi-tier loop
                t0 = time.time()
                launch_res = launch_app(app_name)
                latency = round((time.time() - t0) * 1000, 1)

                is_success = launch_res.get("status") == "SUCCESS"
                spawned_pid = launch_res.get("pid")

                # Snapshot processes after launch
                post_map = get_running_process_map()
                diff_pids = set(post_map.keys()) - set(pre_map.keys())
                if spawned_pid and spawned_pid in post_map:
                    diff_pids.add(spawned_pid)

                # Extract primary PID for logging
                log_pid = spawned_pid
                if not log_pid and diff_pids:
                    log_pid = list(diff_pids)[0]

                stats["total"] += 1
                if is_success:
                    stats["success"] += 1
                    status_tag = "[+]"
                else:
                    stats["failed"] += 1
                    status_tag = "[!]"

                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                mode_str = launch_res.get("mode", "UNKNOWN")
                print(
                    f"{status_tag} [{idx}/{len(apps)}] '{app_name}' -> {launch_res.get('status')} "
                    f"({mode_str}, {latency}ms, PID: {log_pid})",
                    flush=True,
                )

                history.append({
                    "time": now_str,
                    "app": app_name,
                    "mode": mode_str,
                    "status": launch_res.get("status"),
                    "pid": log_pid,
                    "output": launch_res.get("output", "") or launch_res.get("error", ""),
                })

                # Safe cleanup of newly spawned non-protected processes
                for npid in diff_pids:
                    nimg = post_map.get(npid, "")
                    terminate_spawned_process(npid, nimg)

                stats["elapsed_sec"] = int(time.time() - start_time)
                # Periodically update report file
                if stats["total"] % 5 == 0:
                    write_markdown_report(report_path, stats, history)

                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n[*] Test interrupted by user.", flush=True)

    stats["elapsed_sec"] = int(time.time() - start_time)
    write_markdown_report(report_path, stats, history)

    print("\n" + "=" * 70, flush=True)
    print("[*] TEST HARNESS RUN COMPLETE", flush=True)
    print(f"[*] Total Tested: {stats['total']} | Passed: {stats['success']} | Failed: {stats['failed']}", flush=True)
    success_rate = round(stats['success'] / stats['total'] * 100.0, 1) if stats['total'] > 0 else 0.0
    print(f"[*] Overall Success Rate: {success_rate}%", flush=True)
    print(f"[*] Full Audit Report: {os.path.abspath(report_path)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miku Continuous Application Launch Stress Test")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Total hours to run the test loop (default: 1.0)")
    parser.add_argument("--interval", type=float, default=1.5, help="Cooldown between launches in seconds (default: 1.5)")
    parser.add_argument("--max-apps", type=int, default=0, help="Limit number of apps to test (0 = all)")
    parser.add_argument("--report", type=str, default="eval/app_launch_stress_report.md", help="Output report markdown path")

    args = parser.parse_args()
    run_stress_test(
        duration_seconds=args.duration_hours * 3600.0,
        interval_seconds=args.interval,
        report_path=args.report,
        max_apps=args.max_apps,
    )
