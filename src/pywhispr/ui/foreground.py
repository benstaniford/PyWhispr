"""Showing a window where the user is actually looking.

A tray app owns no main window, so its dialogs open on the primary screen behind
whatever has focus — easy to miss entirely on a multi-monitor desktop. Windows
also refuses foreground to a process that is not already in front, so being
topmost matters more than asking for focus.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, QRect, QTimer
from PySide6.QtGui import QCursor, QGuiApplication, QScreen
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

# Qt can lose topmost when the flag is changed around a show(), and a compositor
# may re-order us straight afterwards, so it is claimed once more shortly after.
RE_ASSERT_MS = 400


def show_in_front(window: QWidget) -> None:
    """Show ``window`` on the active screen, above other applications."""
    from PySide6.QtCore import Qt

    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    window.show()
    centre_on_active_screen(window)
    window.raise_()
    window.activateWindow()
    _force_foreground(window)
    # Parented to the window, so closing it before the timer fires cannot leave a
    # callback holding a deleted widget.
    timer = QTimer(window if isinstance(window, QObject) else None)
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: _re_assert(window))
    timer.start(RE_ASSERT_MS)


def centre_on_active_screen(window: QWidget) -> None:
    """Centre on the screen holding the pointer, not whichever one Qt calls first."""
    screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
    if screen is None:
        return
    area = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(area.center())
    window.move(frame.topLeft())


def screen_for_foreground_window() -> QScreen | None:
    """The screen holding the window the user is typing into.

    The pointer is not a reliable stand-in here: dictation is hands-off, so the
    mouse is wherever it was left. Falls back to the primary screen whenever the
    foreground window or its monitor cannot be resolved — which is everything
    off Windows.
    """
    rect = _foreground_monitor_rect()
    if rect is not None:
        # Qt's Windows plugin lays screens out in the native virtual-desktop pixel
        # coordinates, so a Win32 monitor rect can be handed straight to screenAt:
        # devicePixelRatio scales painting, not the layout. Checked on a mixed
        # 100%/125% desktop, where every availableGeometry() equalled its rcWork.
        # Matching on QScreen.name() is what does *not* work — on Windows that is
        # the monitor's model name ("ZOWIE XL LCD"), not the GDI device name.
        screen = QGuiApplication.screenAt(rect.center())
        if screen is not None:
            return screen
        log.debug("No Qt screen at the foreground monitor's centre")
    return QGuiApplication.primaryScreen()


def _foreground_monitor_rect() -> QRect | None:
    """Rect of the monitor most of the foreground window sits on, in Win32 coords."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32),
            ]

        MONITOR_DEFAULTTONEAREST = 0x0002
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return None
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        r = info.rcMonitor
        return QRect(r.left, r.top, r.right - r.left, r.bottom - r.top)
    except Exception:
        log.debug("Could not resolve the foreground window's monitor", exc_info=True)
        return None


def _force_foreground(window: QWidget) -> None:
    """The Win32 half of the same request; Qt's is advisory here."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_SHOWWINDOW = 0x0040
        hwnd = int(window.winId())
        user32 = ctypes.windll.user32
        user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
        )
        user32.SetForegroundWindow(hwnd)
    except Exception:
        log.debug("Could not force the window to the foreground", exc_info=True)


def _re_assert(window: QWidget) -> None:
    try:
        if window.isVisible():
            window.raise_()
            window.activateWindow()
    except RuntimeError:  # the C++ widget went away first
        pass


__all__ = ["centre_on_active_screen", "screen_for_foreground_window", "show_in_front"]
