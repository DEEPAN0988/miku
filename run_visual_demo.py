"""
run_visual_demo.py — Standalone Visual Demonstration of Miku's Win32 Humanizer

Architectural Role:
  Bypasses the Astra LLM orchestrator and directly exercises Miku's Win32
  input dispatchers:
    1. Cubic Bézier / minimum-jerk cursor glide with zero-jerk acceleration.
    2. Individual Unicode keystrokes with randomized human typing cadences.

Execution Sequence:
  - 3-second countdown ("Taking over cursor in 3... 2... 1...")
  - Sweep 1: Smoothly glide cursor from current position to screen center (1.0s).
  - Sweep 2: Smoothly glide cursor from center to bottom-left Start menu area (1.5s).
  - Press Windows Key (VK_LWIN), wait 1.0s for Start menu animation.
  - Type "Notepad" using human-delayed typing dispatcher.
  - Press Enter (VK_RETURN) to launch Notepad, wait 2.0s for window render.
  - Sweep 3: Smoothly glide cursor into center of Notepad window.
  - Type: "Hello! Miku's human-like execution curves are now working perfectly."
  - Graceful KeyboardInterrupt (CTRL+C) emergency stop.
"""

from __future__ import annotations

import os
import sys

# Authorize live autonomous execution for the visual demonstration
os.environ["MIKU_LIVE_EXECUTION"] = "true"
os.environ["MIKU_AUTONOMOUS_MODE"] = "true"

import ctypes
import time
from typing import Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import win32gui
except ImportError:
    win32gui = None

from tools.screen_inspector import (
    human_mouse_move,
    get_current_cursor_pos,
    _attach_thread_to_default_desktop,
)
from tools.typing_automation import (
    dispatch_typing_payload,
    dispatch_human_keystrokes,
)

user32 = ctypes.windll.user32

# Virtual key constants
VK_LWIN = 0x5B
VK_RETURN = 0x0D
KEYEVENTF_KEYUP = 0x0002


def find_notepad_window(max_attempts: int = 15, delay_sec: float = 0.2) -> int:
    """Finds top-level Notepad window handle."""
    if win32gui is None:
        return 0

    for _ in range(max_attempts):
        # Method 1: direct class lookup
        hwnd = win32gui.FindWindow("Notepad", None)
        if hwnd and win32gui.IsWindowVisible(hwnd):
            return hwnd

        # Method 2: enumeration fallback (Win11 modern Notepad packaged window)
        matched_hwnd = 0

        def _enum_cb(h, extra):
            nonlocal matched_hwnd
            if win32gui.IsWindowVisible(h):
                title = win32gui.GetWindowText(h).strip().lower()
                cls = win32gui.GetClassName(h).strip().lower()
                if "notepad" in title or "notepad" in cls:
                    matched_hwnd = h
                    return False
            return True

        try:
            win32gui.EnumWindows(_enum_cb, None)
        except Exception:
            pass

        if matched_hwnd:
            return matched_hwnd

        time.sleep(delay_sec)

    return 0


def run_visual_demo() -> None:
    _attach_thread_to_default_desktop()

    print("=" * 75, flush=True)
    print("[*] MIKU OS AGENT — VISUAL HUMANIZER DEMONSTRATION", flush=True)
    print("=" * 75, flush=True)
    print("[*] Pure Win32 Bézier cursor glide and realistic human typing.", flush=True)
    print("[*] Press CTRL+C at ANY time to abort and reclaim cursor control.", flush=True)
    print("=" * 75, flush=True)

    # 1. Countdown
    print("\n[*] Taking over cursor in 3...", flush=True)
    time.sleep(1.0)
    print("[*] Taking over cursor in 2...", flush=True)
    time.sleep(1.0)
    print("[*] Taking over cursor in 1...", flush=True)
    time.sleep(1.0)
    print("[+] Taking over cursor NOW. Let go of your mouse!\n", flush=True)

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    center_x = screen_w // 2
    center_y = screen_h // 2
    start_btn_x = 30
    start_btn_y = screen_h - 30

    print(f"[*] Screen Resolution Detected: {screen_w} x {screen_h}", flush=True)
    print(f"[*] Screen Center Coordinates: ({center_x}, {center_y})", flush=True)
    print(f"[*] Start Area Coordinates   : ({start_btn_x}, {start_btn_y})", flush=True)

    # 2. Sweep 1: Current position -> screen center over 1.0s
    cur_x, cur_y = get_current_cursor_pos()
    print(f"\n[*] [SWEEP 1] Gliding cursor from ({cur_x}, {cur_y}) to center ({center_x}, {center_y}) over 1.0s...", flush=True)
    human_mouse_move(cur_x, cur_y, center_x, center_y, duration=1.0)
    time.sleep(0.3)

    # 3. Sweep 2: Center -> bottom-left corner over 1.5s
    print(f"[*] [SWEEP 2] Gliding cursor from center to Start area ({start_btn_x}, {start_btn_y}) over 1.5s...", flush=True)
    human_mouse_move(center_x, center_y, start_btn_x, start_btn_y, duration=1.5)
    time.sleep(0.3)

    # 4. Press Windows Key (VK_LWIN) & wait 1.0s
    print("[*] [KEYBOARD] Pressing Windows Key (VK_LWIN)...", flush=True)
    user32.keybd_event(VK_LWIN, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
    print("[*] Waiting 1.0s for Start menu animation to finish...", flush=True)
    time.sleep(1.0)

    # 5. Type "Notepad" with human delay
    search_query = "Notepad"
    print(f"[*] [TYPING] Typing '{search_query}' character-by-character with human micro-delays...", flush=True)
    typed_chars = dispatch_typing_payload(search_query, min_delay_sec=0.04, max_delay_sec=0.08)
    print(f"[+] Typed {typed_chars} characters.", flush=True)
    time.sleep(0.8)

    # 6. Press Enter (VK_RETURN) to launch Notepad & wait 2.0s
    print("[*] [KEYBOARD] Pressing Enter (VK_RETURN) to launch Notepad...", flush=True)
    user32.keybd_event(VK_RETURN, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
    print("[*] Waiting 2.0s for Notepad to load and display...", flush=True)
    time.sleep(2.0)

    # 7. Locate Notepad and Sweep 3
    notepad_hwnd = find_notepad_window()
    if notepad_hwnd and win32gui is not None:
        rect = win32gui.GetWindowRect(notepad_hwnd)
        # Position slightly below titlebar in text area
        notepad_cx = (rect[0] + rect[2]) // 2
        notepad_cy = rect[1] + max(80, int((rect[3] - rect[1]) * 0.45))
        title = win32gui.GetWindowText(notepad_hwnd)
        print(f"[+] [WINDOW FOUND] Notepad HWND {notepad_hwnd} ('{title}') at ({notepad_cx}, {notepad_cy})", flush=True)
        # Ensure Notepad is active foreground window
        user32.SetForegroundWindow(notepad_hwnd)
        time.sleep(0.3)
    else:
        print("[!] Notepad HWND not found via Win32 enum; defaulting Sweep 3 to screen center...", flush=True)
        notepad_cx = center_x
        notepad_cy = center_y

    cur_x, cur_y = get_current_cursor_pos()
    print(f"[*] [SWEEP 3] Gliding cursor into Notepad window at ({notepad_cx}, {notepad_cy}) over 1.0s...", flush=True)
    human_mouse_move(cur_x, cur_y, notepad_cx, notepad_cy, duration=1.0)
    time.sleep(0.1)

    # Left click to focus the text area
    print("[*] Focusing Notepad editor canvas...", flush=True)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.04)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
    time.sleep(0.3)

    # 8. Type demonstration phrase with realistic human delays
    demo_phrase = "Hello! Miku's human-like execution curves are now working perfectly."
    print(f"\n[*] [TYPING DEMO] Typing: '{demo_phrase}'...", flush=True)
    chars_dispatched = dispatch_typing_payload(demo_phrase, min_delay_sec=0.03, max_delay_sec=0.065)
    print(f"[+] Successfully typed {chars_dispatched} characters with realistic delays!", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("[+] VISUAL DEMONSTRATION COMPLETE: All sweeps and typing executed successfully.", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    try:
        run_visual_demo()
    except KeyboardInterrupt:
        print("\n\n[!] [EMERGENCY STOP] Script aborted by user via KeyboardInterrupt (CTRL+C).", flush=True)
        print("[+] Cursor control immediately returned to user.", flush=True)
        sys.exit(0)
