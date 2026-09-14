"""
eval/live_realtime_inspector.py — Continuous Real-Time Live Desktop Window & Screen Inspector

Polls the interactive Windows desktop in real time:
- Automatically detects whenever the user switches focus to any window (IDE, browser, game, explorer, etc.)
- Immediately inspects the foreground window using UIAutomationCore COM via tools.screen_inspector
- Reports HWND, process, window title, interactive element count, inspection latency, and sample UI controls.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.ui_verifier import inspect_live_foreground_window
from tools.screen_inspector import inspect_screen_elements


def start_realtime_monitor(duration_sec: int = 25, poll_interval: float = 0.5):
    print("=" * 80)
    print("LIVE REAL-TIME WINDOW & SCREEN INSPECTOR ACTIVATED")
    print(f"Monitoring foreground focus for {duration_sec}s (sampling every {poll_interval*1000:.0f}ms)")
    print("Switch windows or click on any application to see live real-time inspection!")
    print("=" * 80 + "\n", flush=True)

    start_time = time.time()
    last_hwnd = None
    shift_count = 0

    while (time.time() - start_time) < duration_sec:
        elapsed = time.time() - start_time
        remaining = duration_sec - elapsed

        raw = inspect_live_foreground_window()
        hwnd = raw.get("hwnd", 0)
        proc = raw.get("process_name", "Unknown")
        title = raw.get("title", "")

        if hwnd != last_hwnd:
            shift_count += 1
            last_hwnd = hwnd

            print(f"\n[FOCUS DETECTED #{shift_count} at T+{elapsed:.1f}s | {remaining:.0f}s left]")
            print(f"  Window Title : \"{title}\"")
            print(f"  Process Name : {proc} (PID: {raw.get('process_id')})")
            print(f"  Window Class : {raw.get('class_name')}")
            print(f"  HWND         : {hwnd} ({hex(hwnd)})")

            # Perform live UIA screen inspection
            t0 = time.perf_counter()
            snap = inspect_screen_elements(hwnd)
            insp_ms = (time.perf_counter() - t0) * 1000

            print(f"  --- Live Screen Inspection ({insp_ms:.1f}ms) ---")
            print(f"  Interactive Elements Discovered: {snap.interactive_count}")

            if snap.elements:
                sample = snap.elements[:8]
                for i, el in enumerate(sample, 1):
                    # Sanitize name string for console printing
                    safe_name = el.name.encode("ascii", errors="replace").decode("ascii")
                    print(f"    [{i}] {el.control_type:12}: \"{safe_name}\" at center {el.center}")
                if len(snap.elements) > 8:
                    print(f"    ... and {len(snap.elements) - 8} more interactive controls")
            else:
                print("    (No standard UIA controls exposed or container window)")

            print("-" * 80, flush=True)

        time.sleep(poll_interval)

    print(f"\n[REAL-TIME MONITOR COMPLETE] Monitored for {duration_sec}s. Total focus events captured: {shift_count}.")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Real-Time Desktop Window & Screen Inspector")
    parser.add_argument("--duration", type=int, default=25, help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Poll interval in seconds")
    args = parser.parse_args()

    start_realtime_monitor(duration_sec=args.duration, poll_interval=args.interval)
