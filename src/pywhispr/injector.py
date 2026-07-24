"""Insert text into the focused app via clipboard + simulated paste.

Per-keystroke synthesis is slow and breaks with non-ASCII input methods, so
the reliable cross-platform approach is: save clipboard → set text →
Cmd/Ctrl+V → restore clipboard.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, QTimer, Signal

log = logging.getLogger(__name__)


class TextInjector(QObject):
    """Runs the paste sequence on the GUI thread without blocking it.

    QClipboard is only safe on the main thread, and sleeping there would
    freeze the overlay, so the delays are QTimer hops. ``finished`` is
    emitted when the sequence completes (clipboard restored).
    """

    finished = Signal()

    def __init__(self, paste_delay_ms: int = 150, restore_delay_ms: int = 300):
        super().__init__()
        self._paste_delay_ms = paste_delay_ms
        self._restore_delay_ms = restore_delay_ms
        self._old_text: str | None = None

    def insert(self, text: str) -> None:
        """Paste ``text`` into whatever app has keyboard focus. Main thread only."""
        clipboard = self._clipboard()
        mime = clipboard.mimeData()
        # Only restore plain text; putting images/files back reliably is not
        # worth the complexity, so those are left overwritten.
        self._old_text = clipboard.text() if mime is not None and mime.hasText() else None

        clipboard.setText(text)
        # Delay lets the clipboard settle before pasting (Windows especially).
        QTimer.singleShot(self._paste_delay_ms, self._paste)

    def _clipboard(self):
        from PySide6.QtWidgets import QApplication

        return QApplication.clipboard()

    def _paste(self) -> None:
        self._send_paste_keystroke()
        # The target app must read the clipboard before we restore it.
        QTimer.singleShot(self._restore_delay_ms, self._restore)

    def _send_paste_keystroke(self) -> None:
        from pynput.keyboard import Controller, Key

        controller = Controller()
        modifier = Key.cmd if sys.platform == "darwin" else Key.ctrl
        with controller.pressed(modifier):
            controller.tap("v")

    def _restore(self) -> None:
        if self._old_text is not None:
            self._clipboard().setText(self._old_text)
            self._old_text = None
        log.debug("Insert sequence finished")
        self.finished.emit()
