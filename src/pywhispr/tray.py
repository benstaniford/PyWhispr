"""System tray icon: status, open config, quit."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from pywhispr.config import config_path


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
    def __init__(self, on_quit, on_toggle=None, parent=None):
        super().__init__(_make_icon(), parent)
        self._idle_icon = _make_icon(active=False)
        self._active_icon = _make_icon(active=True)

        menu = QMenu()
        if on_toggle is not None:
            toggle = QAction("Start/stop dictation", menu)
            toggle.triggered.connect(on_toggle)
            menu.addAction(toggle)
            menu.addSeparator()
            # On Windows a plain left-click toggles too (macOS opens the menu).
            self.activated.connect(
                lambda reason: reason == QSystemTrayIcon.ActivationReason.Trigger and on_toggle()
            )
        open_config = QAction("Open config file", menu)
        open_config.triggered.connect(self._open_config)
        menu.addAction(open_config)
        menu.addSeparator()
        quit_action = QAction("Quit PyWhispr", menu)
        quit_action.triggered.connect(on_quit)
        menu.addAction(quit_action)
        self.setContextMenu(menu)
        self.set_status("Starting…")

    def set_status(self, text: str, active: bool = False) -> None:
        self.setToolTip(f"PyWhispr — {text}")
        self.setIcon(self._active_icon if active else self._idle_icon)

    def _open_config(self) -> None:
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_path())))

    def notify(self, title: str, message: str) -> None:
        if self.supportsMessages():
            self.showMessage(title, message, QSystemTrayIcon.MessageIcon.Warning)
