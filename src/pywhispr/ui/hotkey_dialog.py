"""Dialog that records a new global hotkey from an actual keypress.

While the dialog has keyboard focus, Qt delivers key events natively — no
Input Monitoring or other permission is involved in capturing the chord.
"""

from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from pywhispr.hotkey import DOUBLE_TAP_PREFIX, pretty_chord, validate_chord

DOUBLE_TAP_CAPTURE_WINDOW = 0.5  # seconds between taps while recording

# Qt swaps Command and Control on macOS: ControlModifier is ⌘, MetaModifier is ⌃.
_MODIFIER_TOKENS_DARWIN = [
    (Qt.KeyboardModifier.ControlModifier, "<cmd>"),
    (Qt.KeyboardModifier.MetaModifier, "<ctrl>"),
    (Qt.KeyboardModifier.AltModifier, "<alt>"),
    (Qt.KeyboardModifier.ShiftModifier, "<shift>"),
]
_MODIFIER_TOKENS_OTHER = [
    (Qt.KeyboardModifier.MetaModifier, "<cmd>"),  # Windows/Super key
    (Qt.KeyboardModifier.ControlModifier, "<ctrl>"),
    (Qt.KeyboardModifier.AltModifier, "<alt>"),
    (Qt.KeyboardModifier.ShiftModifier, "<shift>"),
]

_SPECIAL_KEYS = {
    Qt.Key.Key_Space: "<space>",
    Qt.Key.Key_Tab: "<tab>",
    Qt.Key.Key_Return: "<enter>",
    Qt.Key.Key_Enter: "<enter>",
    Qt.Key.Key_Backspace: "<backspace>",
    Qt.Key.Key_Delete: "<delete>",
    Qt.Key.Key_Home: "<home>",
    Qt.Key.Key_End: "<end>",
    Qt.Key.Key_PageUp: "<page_up>",
    Qt.Key.Key_PageDown: "<page_down>",
    Qt.Key.Key_Up: "<up>",
    Qt.Key.Key_Down: "<down>",
    Qt.Key.Key_Left: "<left>",
    Qt.Key.Key_Right: "<right>",
}

_PURE_MODIFIER_KEYS = {
    Qt.Key.Key_Shift,
    Qt.Key.Key_Control,
    Qt.Key.Key_Meta,
    Qt.Key.Key_Alt,
    Qt.Key.Key_AltGr,
    Qt.Key.Key_CapsLock,
}


def key_event_to_chord(
    key: int, modifiers: Qt.KeyboardModifier, platform: str = sys.platform
) -> str | None:
    """Translate a Qt key event into a pynput-style chord string.

    Returns None for presses that can't form a chord (bare modifier keys,
    unsupported keys).
    """
    if key in _PURE_MODIFIER_KEYS:
        return None

    mod_tokens = _MODIFIER_TOKENS_DARWIN if platform == "darwin" else _MODIFIER_TOKENS_OTHER
    parts = [token for flag, token in mod_tokens if modifiers & flag]

    if key in _SPECIAL_KEYS:
        parts.append(_SPECIAL_KEYS[key])
    elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F20:
        parts.append(f"<f{key - Qt.Key.Key_F1 + 1}>")
    elif Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
        parts.append(chr(key).lower())
    elif Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        parts.append(chr(key))
    else:
        return None

    return "+".join(parts)


def modifier_key_token(key: int, platform: str = sys.platform) -> str | None:
    """Map a bare modifier keypress to its chord token (respecting Qt's ⌘/⌃ swap)."""
    if key == Qt.Key.Key_Alt:
        return "<alt>"
    if key == Qt.Key.Key_Shift:
        return "<shift>"
    if key == Qt.Key.Key_Control:
        return "<cmd>" if platform == "darwin" else "<ctrl>"
    if key == Qt.Key.Key_Meta:
        return "<ctrl>" if platform == "darwin" else "<cmd>"
    return None


class HotkeyCaptureDialog(QDialog):
    """Modal dialog: press the desired chord, then Save."""

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change hotkey")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(340)
        self._chord: str | None = None
        self._last_mod_token: str | None = None
        self._last_mod_time = 0.0

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Press the new dictation hotkey\n(a chord, or double-tap a modifier key):")
        )

        self._display = QLabel(pretty_chord(current))
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setStyleSheet("font-size: 20px; font-weight: bold; padding: 12px;")
        layout.addWidget(self._display)

        self._hint = QLabel("Esc cancels")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet("color: gray;")
        layout.addWidget(self._hint)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        layout.addWidget(self._buttons)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Route every key here, even ones widgets would normally consume (Tab,
        # Enter, arrows). Released automatically when the dialog closes.
        self.grabKeyboard()

    def hideEvent(self, event) -> None:
        self.releaseKeyboard()
        super().hideEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.isAutoRepeat():  # a held modifier must not count as two taps
            return
        if (
            event.key() == Qt.Key.Key_Escape
            and not event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        ):
            self.reject()
            return

        chord = key_event_to_chord(event.key(), event.modifiers())
        if chord is None:
            chord = self._detect_double_tap(event.key())
            if chord is None:
                return
        else:
            self._last_mod_token = None

        try:
            validate_chord(chord)
        except ValueError:
            self._hint.setText("Unsupported combination — add a modifier (⌘/Ctrl/Alt/Shift)")
            return

        self._chord = chord
        self._display.setText(pretty_chord(chord))
        self._hint.setText("Esc cancels")
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)

    def _detect_double_tap(self, key: int) -> str | None:
        """Two quick presses of the same bare modifier → a double-tap chord."""
        token = modifier_key_token(key)
        if token is None:
            self._last_mod_token = None
            return None
        now = time.monotonic()
        if self._last_mod_token == token and now - self._last_mod_time <= DOUBLE_TAP_CAPTURE_WINDOW:
            self._last_mod_token = None
            return f"{DOUBLE_TAP_PREFIX}{token}"
        self._last_mod_token = token
        self._last_mod_time = now
        self._hint.setText(f"Tap {token.strip('<>').title()} again for a double-tap hotkey")
        return None

    @staticmethod
    def capture(current: str, parent=None) -> str | None:
        """Show the dialog; return the new chord string, or None if cancelled."""
        dialog = HotkeyCaptureDialog(current, parent)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog._chord:
            return dialog._chord
        return None
