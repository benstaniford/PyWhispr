"""Insert text into the focused app via clipboard + simulated paste.

Per-keystroke synthesis is slow and breaks with non-ASCII input methods, so
the reliable cross-platform approach is: save clipboard → set text →
Cmd/Ctrl+V → restore clipboard.

On macOS, synthesizing the paste keystroke requires the Accessibility
permission. Without it we fall back to clipboard-only mode: the transcript is
left on the clipboard (previous contents are NOT restored) and the user
pastes manually.

In that mode the user cannot see when the transcript has landed, so a Cmd+V
pressed a moment too early pastes whatever was copied before. ``hold_clipboard``
empties the clipboard when a recording starts, so an early paste pastes
nothing, and ``release_clipboard`` puts the old contents back if the dictation
ends without a transcript.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QMimeData, QObject, QTimer, Signal

from pywhispr import richclip
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
        # Clipboard-only mode: what was on the clipboard before the recording
        # emptied it, until a transcript replaces it or it is given back.
        self._held: QMimeData | None = None

    def hold_clipboard(self) -> None:
        """Empty the clipboard for the length of a dictation. Main thread only.

        Clipboard-only mode only: with auto-paste the user never presses Cmd+V,
        and the paste sequence saves and restores the clipboard itself. Every
        format is kept, not just text — clearing an image or a copied file and
        then failing to give it back would be data loss of our own making.
        """
        if self._held is not None or self.can_auto_paste():
            return
        clipboard = self._clipboard()
        self._held = _copy_mime(clipboard.mimeData())
        clipboard.clear()
        log.debug("Clipboard held for the dictation (%d formats)", len(self._held.formats()))

    def release_clipboard(self) -> None:
        """Give back what ``hold_clipboard`` took, if a transcript did not replace it.

        Skipped when something new was copied in the meantime: that is the user's
        latest clipboard, and putting an older one over it would lose it.
        """
        held, self._held = self._held, None
        if held is None or not held.formats():
            return
        clipboard = self._clipboard()
        current = clipboard.mimeData()
        if current is not None and current.formats():
            log.debug("Clipboard changed during the dictation; not restoring")
            return
        clipboard.setMimeData(held)
        log.debug("Clipboard restored after a dictation with no transcript")

    def insert(self, text: str, rich: tuple[tuple[int, int, str], ...] = ()) -> None:
        """Paste ``text`` into whatever app has keyboard focus. Main thread only.

        ``rich`` is ``(start, end, html)`` spans over ``text`` for content plain text
        cannot express — a Teams custom emoji is a tenant-hosted image, not a
        character. When present, the clipboard carries **both** an HTML rendering and
        the plain text, and the target application picks. Anything that does not read
        HTML, or any platform where we cannot write it, gets the words: the emoji
        arrives as its name, which is exactly how Teams' own copy degrades.
        """
        if not self.can_auto_paste():
            # Clipboard-only mode: the transcript must stay on the clipboard,
            # so the previous contents are deliberately not saved/restored —
            # including whatever hold_clipboard was keeping.
            self._held = None
            self._clipboard().setText(text)
            log.info("Accessibility not granted: left %d chars on clipboard", len(text))
            self.finished.emit(False)
            return

        # Accessibility granted mid-dictation: put the held clipboard back first,
        # so the ordinary save/restore below preserves it rather than an empty one.
        self.release_clipboard()
        clipboard = self._clipboard()
        mime = clipboard.mimeData()
        # Only restore plain text; putting images/files back reliably is not
        # worth the complexity, so those are left overwritten.
        self._old_text = clipboard.text() if mime is not None and mime.hasText() else None

        if not self._set_rich(text, rich):
            clipboard.setText(text)
        # Delay lets the clipboard settle before pasting (Windows especially).
        QTimer.singleShot(self._paste_delay_ms, self._paste)

    def can_auto_paste(self) -> bool:
        return check_macos_accessibility()

    @staticmethod
    def _set_rich(text: str, rich: tuple[tuple[int, int, str], ...]) -> bool:
        """Offer an HTML rendering as well as the text. False means text only.

        Deliberately not via ``QMimeData.setHtml``: on Windows that writes a
        ``CF_HTML`` with no ``<html>``/``<body>`` root, which WebView2 applications
        like Teams reject outright, silently taking the plain text instead. See
        richclip for the whole story.
        """
        if not rich:
            return False
        try:
            return richclip.set_rich(richclip.render(text, list(rich)), text)
        except Exception:
            # A formatted paste is a luxury; the transcript is not.
            log.exception("Rich paste failed; falling back to plain text")
            return False

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
        self.finished.emit(True)


def _copy_mime(source: QMimeData | None) -> QMimeData:
    """A detached copy of every format. The clipboard's own object changes under us."""
    copy = QMimeData()
    if source is not None:
        for fmt in source.formats():
            copy.setData(fmt, source.data(fmt))
    return copy
