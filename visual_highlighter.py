"""
visual_highlighter.py — Non-Blocking Win32 Screen Target Overlay Highlighter

Creates a transparent, always-on-top, click-through visual overlay window on Windows
to highlight target UI elements while waiting for human fast-confirm authorization.

CONSTRAINTS:
1. Runs in a separate daemon thread so it never blocks the asyncio loop or terminal input.
2. Uses native Win32 WS_EX_TRANSPARENT + WS_EX_LAYERED + WS_EX_NOACTIVATE via ctypes
   for zero-focus, 100% click-through overlay without Tcl/Tkinter thread issues.
3. Draws a thick red bounding box and center targeting reticle.
"""

import ctypes
from ctypes import wintypes
import sys
import threading
import time
from typing import Optional, Tuple

if sys.platform == "win32":
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    kernel32 = ctypes.windll.kernel32

    # Set argument & return types for Win32 API 64-bit safety
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

# Color Key (Pure Magenta #FF00FF for 100% transparency key)
KEY_COLOR_RGB = 0x00FF00FF


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


class TargetHighlighter:
    """
    Native Win32 overlay manager drawing a red target box non-blockingly.
    """

    def __init__(
        self,
        color: str = "red",
        color_rgb: Optional[int] = None,
        line_width: int = 4,
        enabled: bool = True,
    ):
        if color_rgb is not None:
            self.color_rgb = color_rgb
        else:
            # Default red in GDI (BGR format: 0x000000FF)
            self.color_rgb = 0x000000FF

        self.line_width = line_width
        self.enabled = enabled

        self._hwnd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ready_event = threading.Event()
        self._active_target: Optional[dict] = None

        if self.enabled and sys.platform == "win32":
            self._start_thread()

    def _start_thread(self):
        self._running = True
        self._thread = threading.Thread(target=self._win32_thread_main, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=2.0)

    def _win32_thread_main(self):
        """Native Win32 Window Message Loop."""
        WNDPROC = ctypes.WINFUNCTYPE(
            wintypes.LPARAM, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_PAINT:
                ps = PAINTSTRUCT()
                hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
                
                # Fill canvas background with magenta transparency key
                rect = RECT()
                user32.GetClientRect(hwnd, ctypes.byref(rect))
                bg_brush = gdi32.CreateSolidBrush(KEY_COLOR_RGB)
                user32.FillRect(hdc, ctypes.byref(rect), bg_brush)
                gdi32.DeleteObject(bg_brush)

                if self._active_target:
                    pen = gdi32.CreatePen(0, self.line_width, self.color_rgb)
                    old_pen = gdi32.SelectObject(hdc, pen)
                    gdi32.SetBkMode(hdc, 1)
                    
                    pad = 10
                    box_w = rect.right - rect.left - pad * 2
                    box_h = rect.bottom - rect.top - pad * 2

                    if box_w > 0 and box_h > 0:
                        null_brush = gdi32.GetStockObject(5)
                        old_brush = gdi32.SelectObject(hdc, null_brush)
                        gdi32.Rectangle(hdc, pad, pad, pad + box_w, pad + box_h)
                        gdi32.SelectObject(hdc, old_brush)

                        # Corner reticles
                        ret = min(15, box_w // 3, box_h // 3)
                        r_l, r_t, r_r, r_b = pad, pad, pad + box_w, pad + box_h
                        gdi32.MoveToEx(hdc, r_l, r_t, None)
                        gdi32.LineTo(hdc, r_l + ret, r_t)
                        gdi32.MoveToEx(hdc, r_l, r_t, None)
                        gdi32.LineTo(hdc, r_l, r_t + ret)
                        gdi32.MoveToEx(hdc, r_r - ret, r_t, None)
                        gdi32.LineTo(hdc, r_r, r_t)
                        gdi32.MoveToEx(hdc, r_r, r_t, None)
                        gdi32.LineTo(hdc, r_r, r_t + ret)
                        gdi32.MoveToEx(hdc, r_l, r_b, None)
                        gdi32.LineTo(hdc, r_l + ret, r_b)
                        gdi32.MoveToEx(hdc, r_l, r_b - ret, None)
                        gdi32.LineTo(hdc, r_l, r_b)
                        gdi32.MoveToEx(hdc, r_r - ret, r_b, None)
                        gdi32.LineTo(hdc, r_r, r_b)
                        gdi32.MoveToEx(hdc, r_r, r_b - ret, None)
                        gdi32.LineTo(hdc, r_r, r_b)

                    gdi32.SelectObject(hdc, old_pen)
                    gdi32.DeleteObject(pen)

                user32.EndPaint(hwnd, ctypes.byref(ps))
                return 0

            elif msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0

            # Convert unsigned/large int to signed 64-bit LPARAM for DefWindowProcW
            lparam_signed = ctypes.c_ssize_t(lparam).value
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam_signed)

        wndproc_cb = WNDPROC(wnd_proc)

        class_name = f"FastConfirmHighlighter_{id(self)}"
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
        style = WS_POPUP

        hwnd = user32.CreateWindowExW(
            ex_style,
            class_name,
            "FastConfirm Target Overlay",
            style,
            0, 0, 10, 10,
            None, None, hinst, None
        )

        if not hwnd:
            self._ready_event.set()
            return

        self._hwnd = hwnd
        user32.SetLayeredWindowAttributes(hwnd, KEY_COLOR_RGB, 255, LWA_COLORKEY)
        self._ready_event.set()

        msg = wintypes.MSG()
        while self._running:
            b_ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if b_ret <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def start_highlight(
        self,
        x: int,
        y: int,
        w: Optional[int] = None,
        h: Optional[int] = None,
        bounds: Optional[Tuple[int, int, int, int]] = None,
        label: Optional[str] = None,
    ):
        """Displays the visual red target highlight overlay non-blockingly."""
        if not self.enabled or not self._hwnd:
            return

        if bounds is not None and len(bounds) == 4:
            left, top, right, bottom = bounds
        elif w is not None and h is not None and w > 0 and h > 0:
            left = x - w // 2
            top = y - h // 2
            right = x + w // 2
            bottom = y + h // 2
        else:
            half_s = 30
            left = x - half_s
            top = y - half_s
            right = x + half_s
            bottom = y + half_s

        box_w = max(right - left, 10)
        box_h = max(bottom - top, 10)

        pad = 10
        win_left = max(left - pad, 0)
        win_top = max(top - pad, 0)
        win_w = box_w + pad * 2
        win_h = box_h + pad * 2

        self._active_target = {"left": left, "top": top, "right": right, "bottom": bottom, "label": label}

        user32.SetWindowPos(self._hwnd, -1, win_left, win_top, win_w, win_h, 0x0010 | 0x0040)
        user32.InvalidateRect(self._hwnd, None, True)

    def stop_highlight(self):
        """Hides and clears the target highlight overlay immediately."""
        if not self.enabled or not self._hwnd:
            return
        self._active_target = None
        user32.SetWindowPos(self._hwnd, 0, 0, 0, 0, 0, 0x0080 | 0x0010 | 0x0002 | 0x0001)

    def close(self):
        """Destroys the Win32 window cleanly."""
        if not self.enabled or not self._running:
            return
        self._running = False
        if self._hwnd:
            user32.SendMessageW(self._hwnd, WM_CLOSE, 0, 0)
            self._hwnd = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
