from unittest.mock import patch

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QWidget

from pywhispr.ui import foreground


class TestShowInFront:
    def test_shown_topmost(self, qtbot):
        from PySide6.QtCore import Qt

        window = QWidget()
        qtbot.addWidget(window)
        foreground.show_in_front(window)
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        assert window.isVisible()

    def test_survives_a_screenless_session(self, qtbot):
        """No screen at the cursor and no primary screen: still shown, not crashed."""
        window = QWidget()
        qtbot.addWidget(window)
        with (
            patch.object(foreground.QGuiApplication, "screenAt", return_value=None),
            patch.object(foreground.QGuiApplication, "primaryScreen", return_value=None),
        ):
            foreground.show_in_front(window)
        assert window.isVisible()


class TestCentring:
    def test_uses_the_screen_under_the_pointer(self, qtbot):
        """A tray app has no main window, so Qt would otherwise pick the primary one."""

        class Screen:
            def availableGeometry(self):
                return QRect(-1920, 0, 1920, 1080)  # a monitor to the left

        window = QWidget()
        qtbot.addWidget(window)
        window.resize(400, 200)
        with (
            patch.object(foreground.QGuiApplication, "screenAt", return_value=Screen()),
            patch.object(foreground.QCursor, "pos", return_value=QPoint(-1000, 500)),
        ):
            foreground.centre_on_active_screen(window)
        assert QRect(-1920, 0, 1920, 1080).contains(window.geometry().center())


class TestScreenForForegroundWindow:
    def test_picks_the_screen_holding_the_focused_window(self):
        """The pointer is no help: dictation is hands-off, so the mouse sits
        wherever it was left."""
        left = object()
        monitor = QRect(-1920, 0, 1920, 1080)
        with (
            patch.object(foreground, "_foreground_monitor_rect", return_value=monitor),
            patch.object(
                foreground.QGuiApplication, "screenAt", return_value=left
            ) as screen_at,
        ):
            assert foreground.screen_for_foreground_window() is left
        assert screen_at.call_args.args[0] == monitor.center()

    def test_falls_back_to_primary_without_a_foreground_monitor(self):
        primary = object()
        with (
            patch.object(foreground, "_foreground_monitor_rect", return_value=None),
            patch.object(foreground.QGuiApplication, "primaryScreen", return_value=primary),
        ):
            assert foreground.screen_for_foreground_window() is primary

    def test_falls_back_to_primary_when_qt_knows_no_such_screen(self):
        """A monitor can be unplugged between Qt's enumeration and now."""
        primary = object()
        with (
            patch.object(
                foreground, "_foreground_monitor_rect", return_value=QRect(0, 0, 800, 600)
            ),
            patch.object(foreground.QGuiApplication, "screenAt", return_value=None),
            patch.object(foreground.QGuiApplication, "primaryScreen", return_value=primary),
        ):
            assert foreground.screen_for_foreground_window() is primary

    def test_a_broken_win32_call_is_not_fatal(self):
        """Never raise out of here — the overlay is shown from the recording path."""
        with patch.object(foreground.sys, "platform", "win32"), patch.dict(
            "sys.modules", {"ctypes": None}
        ):
            assert foreground._foreground_monitor_rect() is None

    def test_off_windows_there_is_no_monitor_to_resolve(self):
        with patch.object(foreground.sys, "platform", "darwin"):
            assert foreground._foreground_monitor_rect() is None
