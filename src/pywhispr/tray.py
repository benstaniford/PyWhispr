"""System tray icon: status, start/stop, recall, settings, quit."""

from __future__ import annotations

import logging
from importlib.resources import files

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from pywhispr import flavor
from pywhispr.config import config_path
from pywhispr.logging_setup import log_path

log = logging.getLogger(__name__)

SETTINGS_TEXT = "Settings…"


def app_pixmap() -> QPixmap:
    """The shh-gesture app icon (black line art on transparent)."""
    return QPixmap(str(files("pywhispr") / "assets" / "icon.png"))


def _make_icon(active: bool = False) -> QIcon:
    pixmap = app_pixmap()
    if active:
        # Recording: tint the line art red instead of using the mask form.
        tinted = QPixmap(pixmap.size())
        tinted.fill(QColor(0, 0, 0, 0))
        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(220, 60, 60))
        painter.end()
        pixmap = tinted
    icon = QIcon(pixmap)
    icon.setIsMask(not active)  # adapts to macOS menu bar light/dark
    return icon


class TrayIcon(QSystemTrayIcon):
    """Start/stop, recall, settings, quit — and nothing else.

    Everything that is a *preference* moved to ui/settings_dialog.py: a tray menu
    is a place for the two or three things you reach mid-sentence, and this one had
    grown a row per feature.
    """

    def __init__(
        self,
        on_quit,
        on_toggle=None,
        on_show_history=None,
        on_settings=None,
        parent=None,
    ):
        super().__init__(_make_icon(), parent)
        self._idle_icon = _make_icon(active=False)
        self._active_icon = _make_icon(active=True)

        menu = QMenu()
        if on_toggle is not None:
            toggle = QAction("Start/stop dictation", menu)
            toggle.triggered.connect(on_toggle)
            menu.addAction(toggle)
            menu.addSeparator()
            # Deliberately no `activated` handler: clicking the icon must only
            # open the menu. A stray click that silently starts recording is
            # worse than one that does nothing.
        if on_show_history is not None:
            history = QAction("Recent dictations…", menu)
            history.triggered.connect(on_show_history)
            menu.addAction(history)
        if on_settings is not None:
            settings = QAction(SETTINGS_TEXT, menu)
            settings.triggered.connect(on_settings)
            menu.addAction(settings)
        menu.addSeparator()
        quit_action = QAction(f"Quit {flavor.PRODUCT_NAME}", menu)
        quit_action.triggered.connect(on_quit)
        menu.addAction(quit_action)
        self.setContextMenu(menu)
        self.set_status("Starting…")

    def set_status(self, text: str, active: bool = False) -> None:
        self.setToolTip(f"{flavor.PRODUCT_NAME} — {text}")
        self.setIcon(self._active_icon if active else self._idle_icon)

    # Kept on the tray rather than moved with the menu entries: the settings page
    # is what calls them now, but "where is the config/log" is knowledge about the
    # desktop, and open_path lives here.
    def open_config(self) -> None:
        self.open_path(config_path())

    def open_log(self) -> None:
        path = log_path()
        # Nothing has been logged yet if the file could not be created; opening
        # the folder at least shows the user where to look.
        self.open_path(path if path.exists() else path.parent)

    @staticmethod
    def open_path(path) -> None:
        """Hand a file or folder to the desktop. Public: the app opens the plugins
        folder through it, having had to create the folder first."""
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def notify(self, title: str, message: str) -> None:
        if self.supportsMessages():
            self.showMessage(title, message, QSystemTrayIcon.MessageIcon.Warning)
