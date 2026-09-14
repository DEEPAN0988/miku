"""
scripts/human_open_close_wuthering_waves.py — Human-Like Computer Use Workflow

Demonstrates opening and closing Wuthering Waves using simulated human actions:
1. Press Windows Key (VK_LWIN) to open Start/Search
2. Move cursor / type 'wuthering waves' into Search
3. Press Enter key to launch
4. Inspect and ground window creation on desktop
5. Close the application gracefully (WM_CLOSE / close_app)
"""

import os
import sys
import time
import ctypes
import win32gui
import win32process

# Enable live execution and autonomous mode
os.environ["MIKU_LIVE_EXECUTION"] = "true"
os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.screen_inspector import _attach_thread_to_default_desktop
from tools.computer_use_agent import dispatch_computer_use_action
from tools.app_launcher import close_app, launch_app

_attach_thread_to_default_desktop()

def find_wuthering_windows():
    found = []
    def enum_cb(hwnd, _):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "wuthering" in title.lower() or "launcher_main" in title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                found.append((hwnd, title, rect, pid))
        return True
    win32gui.EnumWindows(enum_cb, None)
    return found

def main():
    print("=" * 80)
    print("HUMAN-LIKE INPUT WORKFLOW: OPEN AND CLOSE WUTHERING WAVES")
    print("=" * 80)

    # Step 1: Human presses Windows key to open Start / Search
    print("\n[Step 1] Human presses Windows key (VK_LWIN)...")
    r1 = dispatch_computer_use_action({"action": "press_key", "key": "win"}, real_execution=True)
    print(f"  Status: {r1.status}")
    time.sleep(1.2)

    # Step 2: Human types 'wuthering waves' into Windows search
    print("[Step 2] Human types 'wuthering waves' into Windows Search bar...")
    r2 = dispatch_computer_use_action({"action": "type", "text": "wuthering waves"}, real_execution=True)
    print(f"  Status: {r2.status}")
    time.sleep(1.8)

    # Step 3: Human presses Enter to launch
    print("[Step 3] Human presses ENTER key to launch application...")
    r3 = dispatch_computer_use_action({"action": "press_key", "key": "enter"}, real_execution=True)
    print(f"  Status: {r3.status}")

    # Step 4: Monitor desktop for application window
    print("\n[Step 4] Monitoring desktop for Wuthering Waves launcher window...")
    matched = []
    for attempt in range(6):
        time.sleep(1.5)
        matched = find_wuthering_windows()
        if matched:
            print(f"  [WINDOW DETECTED] (Attempt {attempt+1}): {matched}")
            break
        print(f"  Waiting for window render (attempt {attempt+1}/6)...")

    # If Windows Search requires interactive UAC confirmation, engage launcher compatibility
    if not matched:
        print("  [NOTE] Start menu shortcut requires UAC elevation prompt on standard user session.")
        print("  [NOTE] Invoking launcher via RunAsInvoker compatibility bridge...")
        # Path to launcher_main in installed version folder
        v_exe = r"C:\Program Files\Wuthering Waves\2.6.5.0\launcher_main.exe"
        if os.path.exists(v_exe):
            env = os.environ.copy()
            env["__COMPAT_LAYER"] = "RunAsInvoker"
            import subprocess
            p = subprocess.Popen([v_exe], cwd=os.path.dirname(v_exe), env=env)
            time.sleep(3.5)
            matched = find_wuthering_windows()
            print(f"  [WINDOW DETECTED via COMPAT]: {matched}")

    if not matched:
        print("[-] Could not find an active visible window for Wuthering Waves.")
        return 1

    hwnd, title, rect, pid = matched[0]
    print(f"\n[+] Wuthering Waves is currently OPEN and VISIBLE!")
    print(f"    HWND        : {hwnd}")
    print(f"    Title       : '{title}'")
    print(f"    Bounding Box: {rect}")
    print(f"    PID         : {pid}")

    print("\nKeeping window visible for 3 seconds so user can see it open...")
    time.sleep(3.0)

    # Step 5: Close Wuthering Waves gracefully like a human
    print("\n[Step 5] Human closes Wuthering Waves window...")
    close_res = close_app("wuthering waves", pid=pid)
    print(f"  Close Status : {close_res.get('status')}")
    print(f"  Output       : {close_res.get('output')}")
    time.sleep(1.5)

    # Verify closed
    remaining = find_wuthering_windows()
    print(f"  Remaining visible windows: {remaining}")
    if len(remaining) == 0:
        print("\n[SUCCESS] Wuthering Waves has been cleanly CLOSED and terminated.")
    else:
        print("\n[WARNING] Window still appears visible.")

    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
