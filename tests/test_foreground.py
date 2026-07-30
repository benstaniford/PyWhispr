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
