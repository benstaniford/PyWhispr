"""System tray icon: status, open config, quit."""

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

# One entry, both directions — see the note on _refresh_gpu_label.
GPU_ENABLE_TEXT = "Enable GPU acceleration…"
GPU_DISABLE_TEXT = "Disable GPU acceleration…"


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
    def __init__(
        self,
        on_quit,
        on_toggle=None,
        on_change_hotkey=None,
        on_edit_vocabulary=None,
        on_open_plugins=None,
        on_enable_gpu=None,
        on_disable_gpu=None,
        gpu_active=None,
        on_show_history=None,
        on_set_server=None,
        parent=None,
    ):
        super().__init__(_make_icon(), parent)
        self._idle_icon = _make_icon(active=False)
        self._active_icon = _make_icon(active=True)
        self._on_enable_gpu = on_enable_gpu
        self._on_disable_gpu = on_disable_gpu
        self._gpu_active = gpu_active
        self._gpu_action = None

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
        if on_change_hotkey is not None:
            change_hotkey = QAction("Change hotkey…", menu)
            change_hotkey.triggered.connect(on_change_hotkey)
            menu.addAction(change_hotkey)
        if on_edit_vocabulary is not None:
            vocabulary = QAction("Vocabulary…", menu)
            vocabulary.triggered.connect(on_edit_vocabulary)
            menu.addAction(vocabulary)
        if on_open_plugins is not None:
            # Opens the folder rather than managing anything: plugins are files,
            # they load at startup, and nothing here ever installs one.
            plugins = QAction("Open plugins folder…", menu)
            plugins.triggered.connect(on_open_plugins)
            menu.addAction(plugins)
        if on_set_server is not None:
            # Lite only: where transcription is sent. Provided by the app only in
            # the Lite build, so the full app's menu is unchanged.
            set_server = QAction("Set server…", menu)
            set_server.triggered.connect(on_set_server)
            menu.addAction(set_server)
        if on_enable_gpu is not None:
            self._gpu_action = QAction(GPU_ENABLE_TEXT, menu)
            self._gpu_action.triggered.connect(self._gpu_clicked)
            menu.addAction(self._gpu_action)
            # Pulled from the predicate every time the menu opens rather than
            # pushed from the app, because the answer changes where no signal
            # reaches: a tray-triggered setup finishing (app._on_gpu_setup_finished
            # returns early for exactly that case), "pywhispr disable-gpu" in a
            # terminal, a directory deleted by hand. One connection covers them all.
            menu.aboutToShow.connect(self._refresh_gpu_label)
        open_config = QAction("Open config file", menu)
        open_config.triggered.connect(self._open_config)
        menu.addAction(open_config)
        # The packaged Windows build has no console, so this is the only way a
        # user can get at what actually went wrong.
        open_log = QAction("Open log file", menu)
        open_log.triggered.connect(self._open_log)
        menu.addAction(open_log)
        menu.addSeparator()
        quit_action = QAction(f"Quit {flavor.PRODUCT_NAME}", menu)
        quit_action.triggered.connect(on_quit)
        menu.addAction(quit_action)
        self.setContextMenu(menu)
        self.set_status("Starting…")

    def _refresh_gpu_label(self) -> None:
        if self._gpu_action is not None:
            self._gpu_action.setText(
                GPU_DISABLE_TEXT if self._gpu_is_active() else GPU_ENABLE_TEXT
            )

    def _gpu_clicked(self, _checked: bool = False) -> None:
        """Enable or disable, asked again now rather than read off the label.

        ``_checked`` is ``triggered(bool checked = false)``: PySide6 hands that bool
        to any slot that will accept an argument, so connecting a handler with an
        optional first parameter straight to this signal silently passes it False.
        That is what used to call ``app._enable_gpu(asked_by_user=False)`` on every
        click — skipping the "not available" answer and offering a CUDA download on
        machines that cannot run one. Hence the indirection and the ignored argument.
        """
        handler = self._on_disable_gpu if self._gpu_is_active() else self._on_enable_gpu
        if handler is not None:
            handler()

    def _gpu_is_active(self) -> bool:
        """Is GPU acceleration on? Anything unanswerable counts as off.

        A raising predicate must not stop the menu opening — a tray app whose menu
        will not open is the whole UI gone — and "off" is the recoverable direction:
        the worst it costs is an offer the user declines.
        """
        if self._gpu_active is None:
            return False
        try:
            return bool(self._gpu_active())
        except Exception:
            log.debug("Could not tell whether GPU acceleration is on", exc_info=True)
            return False

    def set_status(self, text: str, active: bool = False) -> None:
        self.setToolTip(f"{flavor.PRODUCT_NAME} — {text}")
        self.setIcon(self._active_icon if active else self._idle_icon)

    def _open_config(self) -> None:
        self.open_path(config_path())

    def _open_log(self) -> None:
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
