"""Showing a window where the user is actually looking.

A tray app owns no main window, so its dialogs open on the primary screen behind
whatever has focus — easy to miss entirely on a multi-monitor desktop. Windows
also refuses foreground to a process that is not already in front, so being
topmost matters more than asking for focus.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QCursor, QGuiApplication
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


__all__ = ["centre_on_active_screen", "show_in_front"]
