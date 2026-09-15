"""
miku_hud.py — Real-Time Non-Blocking Win32 Desktop Agent HUD Overlay

Displays an active, transparent, non-blocking floating status HUD on the top right
corner of the Windows desktop to visually stream MIKU's autonomous thoughts, actions,
perception latency, and step status in real time.
"""

import ctypes
from ctypes import wintypes
import os
import sys
import threading
import time
from typing import Optional, Tuple

if sys.platform == "win32":
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    kernel32 = ctypes.windll.kernel32

    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.DefWindowProcW.restype = wintypes.LPARAM
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = wintypes.LPARAM
else:
    user32 = None
    gdi32 = None
    kernel32 = None

# Win32 Constants
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

LWA_COLORKEY = 0x00000001
LWA_ALPHA = 0x00000002

WM_DESTROY = 0x0002
WM_PAINT = 0x000F
WM_CLOSE = 0x0010

KEY_COLOR_RGB = 0x00FF00FF  # Magenta Key


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", wintypes.BYTE * 32),
    ]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class MikuHUD:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and (sys.platform == "win32")
        self._hwnd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ready_event = threading.Event()

        self.status_text = "MIKU OS AGENT: Ready"
        self.step_info = "Subsystems: 100% Operational"

        if self.enabled:
            self._start_thread()

    def _start_thread(self):
        self._running = True
        self._thread = threading.Thread(target=self._win32_thread_main, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=2.0)

    def _win32_thread_main(self):
        WNDPROC = ctypes.WINFUNCTYPE(
            wintypes.LPARAM, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_PAINT:
                ps = PAINTSTRUCT()
                hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))

                rect = RECT()
                user32.GetClientRect(hwnd, ctypes.byref(rect))

                # Background fill with transparent key
                bg_brush = gdi32.CreateSolidBrush(KEY_COLOR_RGB)
                user32.FillRect(hdc, ctypes.byref(rect), bg_brush)
                gdi32.DeleteObject(bg_brush)

                # Draw dark rounded HUD box
                box_brush = gdi32.CreateSolidBrush(0x001E1E1E)  # Dark slate BGR
                border_pen = gdi32.CreatePen(0, 2, 0x0000FF00)   # Green border
                old_brush = gdi32.SelectObject(hdc, box_brush)
                old_pen = gdi32.SelectObject(hdc, border_pen)

                gdi32.RoundRect(hdc, 5, 5, rect.right - 5, rect.bottom - 5, 12, 12)

                # Render text
                gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
                gdi32.SetTextColor(hdc, 0x0000FF00)  # Bright Green

                # Title Text
                text1 = f" [MIKU HUD] {self.status_text}"
                gdi32.TextOutW(hdc, 15, 12, text1, len(text1))

                gdi32.SetTextColor(hdc, 0x00FFFFFF)  # White text
                text2 = f" Step: {self.step_info}"
                gdi32.TextOutW(hdc, 15, 32, text2, len(text2))

                gdi32.SelectObject(hdc, old_brush)
                gdi32.SelectObject(hdc, old_pen)
                gdi32.DeleteObject(box_brush)
                gdi32.DeleteObject(border_pen)

                user32.EndPaint(hwnd, ctypes.byref(ps))
                return 0

            elif msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0

            lparam_signed = ctypes.c_ssize_t(lparam).value
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam_signed)

        wndproc_cb = WNDPROC(wnd_proc)
        class_name = f"MikuHUDClass_{id(self)}"
        hinst = kernel32.GetModuleHandleW(None)

        wc = WNDCLASSW()
        wc.lpfnWndProc = ctypes.cast(wndproc_cb, ctypes.c_void_p)
        wc.hInstance = hinst
        wc.lpszClassName = class_name
        wc.hbrBackground = gdi32.CreateSolidBrush(KEY_COLOR_RGB)

        user32.RegisterClassW(ctypes.byref(wc))

        ex_style = (
            WS_EX_TOPMOST
            | WS_EX_TRANSPARENT
            | WS_EX_LAYERED
            | WS_EX_NOACTIVATE
            | WS_EX_TOOLWINDOW
        )
        style = WS_POPUP | WS_VISIBLE

        # Position HUD at upper-right corner
        screen_w = user32.GetSystemMetrics(0)
        hud_w = 420
        hud_h = 60
        hud_x = max(screen_w - hud_w - 20, 20)
        hud_y = 20

        hwnd = user32.CreateWindowExW(
            ex_style, class_name, "MIKU Desktop HUD", style,
            hud_x, hud_y, hud_w, hud_h,
            None, None, hinst, None
        )

        if not hwnd:
            self._ready_event.set()
            return

        self._hwnd = hwnd
        user32.SetLayeredWindowAttributes(hwnd, KEY_COLOR_RGB, 235, LWA_COLORKEY | LWA_ALPHA)
        self._ready_event.set()

        msg = wintypes.MSG()
        while self._running:
            b_ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if b_ret <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def update_status(self, status: str, step: Optional[str] = None):
        """Updates the status and step info displayed on the floating HUD."""
        self.status_text = status
        if step is not None:
            self.step_info = step

        if self.enabled and self._hwnd:
            user32.InvalidateRect(self._hwnd, None, True)

    def close(self):
        """Destroys the Win32 HUD window cleanly."""
        if not self.enabled or not self._running:
            return
        self._running = False
        if self._hwnd:
            user32.SendMessageW(self._hwnd, WM_CLOSE, 0, 0)
            self._hwnd = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


_global_hud: Optional[MikuHUD] = None


def get_global_hud() -> MikuHUD:
    global _global_hud
    if _global_hud is None:
        _global_hud = MikuHUD(enabled=True)
    return _global_hud


def update_miku_hud(status: str, step: Optional[str] = None):
    """Global helper function to update MIKU's live desktop HUD status."""
    try:
        hud = get_global_hud()
        hud.update_status(status, step)
    except Exception:
        pass
