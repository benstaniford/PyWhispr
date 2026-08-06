"""Insert text into the focused app via clipboard + simulated paste.

Per-keystroke synthesis is slow and breaks with non-ASCII input methods, so
the reliable cross-platform approach is: save clipboard → set text →
Cmd/Ctrl+V → restore clipboard.

On macOS, synthesizing the paste keystroke requires the Accessibility
permission. Without it we fall back to clipboard-only mode: the transcript is
left on the clipboard (previous contents are NOT restored) and the user
pastes manually.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, QTimer, Signal

from pywhispr import perf
from pywhispr.platform_setup import check_macos_accessibility

log = logging.getLogger(__name__)


class TextInjector(QObject):
    """Runs the paste sequence on the GUI thread without blocking it.

    QClipboard is only safe on the main thread, and sleeping there would
    freeze the overlay, so the delays are QTimer hops. ``finished(auto_pasted)``
    is emitted when the sequence completes; ``auto_pasted`` is False when the
    text was only copied to the clipboard (Accessibility not granted).
    """

    finished = Signal(bool)

    def __init__(self, paste_delay_ms: int = 150, restore_delay_ms: int = 300):
        super().__init__()
        self._paste_delay_ms = paste_delay_ms
        self._restore_delay_ms = restore_delay_ms
        self._old_text: str | None = None

    def insert(self, text: str) -> None:
        """Paste ``text`` into whatever app has keyboard focus. Main thread only."""
        clipboard = self._clipboard()

        if not self.can_auto_paste():
            # Clipboard-only mode: the transcript must stay on the clipboard,
            # so the previous contents are deliberately not saved/restored.
            clipboard.setText(text)
            log.info("Accessibility not granted: left %d chars on clipboard", len(text))
            self.finished.emit(False)
            return

        mime = clipboard.mimeData()
        # Only restore plain text; putting images/files back reliably is not
        # worth the complexity, so those are left overwritten.
        self._old_text = clipboard.text() if mime is not None and mime.hasText() else None
        perf.mark("clipboard-read")

        clipboard.setText(text)
        perf.mark("clipboard-set")
        # Delay lets the clipboard settle before pasting (Windows especially).
        QTimer.singleShot(self._paste_delay_ms, self._paste)

    def can_auto_paste(self) -> bool:
        return check_macos_accessibility()

    def _clipboard(self):
        from PySide6.QtWidgets import QApplication

        return QApplication.clipboard()

    def _paste(self) -> None:
        perf.mark("paste-timer")
        self._send_paste_keystroke()
        perf.mark("keystroke-sent")
        # The target app must read the clipboard before we restore it.
        QTimer.singleShot(self._restore_delay_ms, self._restore)

    def _send_paste_keystroke(self) -> None:
        from pynput.keyboard import Controller, Key

        controller = Controller()
        perf.mark("controller-built")
        modifier = Key.cmd if sys.platform == "darwin" else Key.ctrl
        with controller.pressed(modifier):
            controller.tap("v")

    def _restore(self) -> None:
        perf.mark("restore-timer")
        if self._old_text is not None:
            self._clipboard().setText(self._old_text)
            self._old_text = None
        perf.mark("clipboard-restored")
        log.debug("Insert sequence finished")
        self.finished.emit(True)
