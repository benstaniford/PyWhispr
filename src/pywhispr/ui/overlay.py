"""Frameless always-on-top overlay shown while recording/transcribing.

Run ``python -m pywhispr.ui.overlay`` for a standalone demo with fake levels.
"""

from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget

from pywhispr.ui.waveform import WaveformWidget

PILL_WIDTH = 280
PILL_HEIGHT = 64
BOTTOM_MARGIN = 48
BACKGROUND = QColor(20, 20, 24, 235)
TOPMOST_INTERVAL_MS = 1000  # re-assert while visible; other apps steal the top slot


class OverlayWindow(QWidget):
    """Small pill at the bottom-center of the screen. Never takes focus —
    stealing focus would make the final paste land in the overlay instead of
    the user's app."""

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        self.setFixedSize(PILL_WIDTH, PILL_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 14, 24, 14)

        self.waveform = WaveformWidget(self)
        layout.addWidget(self.waveform)

        self.status_label = QLabel(self)
        self.status_label.setStyleSheet("color: white; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        self.status_label.hide()

        # WindowStaysOnTopHint only wins the ordering at show time: anything that
        # later goes topmost itself (Teams calls, video players, other overlays)
        # ends up above us. Nudge ourselves back while we're on screen.
        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(TOPMOST_INTERVAL_MS)
        self._topmost_timer.timeout.connect(self._raise_to_top)

    def _raise_to_top(self) -> None:
        if not self.isVisible():
            return
        if sys.platform == "win32":
            # HWND_TOPMOST, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE: reclaim the
            # topmost band without moving focus. raise_() only reorders within it.
            ctypes.windll.user32.SetWindowPos(int(self.winId()), -1, 0, 0, 0, 0, 0x13)
        else:
            self.raise_()

    def _show_on_top(self) -> None:
        self.show()
        self._raise_to_top()
        self._topmost_timer.start()

    def _move_to_bottom_center(self) -> None:
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - self.height() - BOTTOM_MARGIN
        self.move(x, y)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BACKGROUND)
        radius = self.height() / 2
        painter.drawRoundedRect(self.rect(), radius, radius)

    def show_recording(self) -> None:
        self.status_label.hide()
        self.waveform.show()
        self.waveform.start()
        self._move_to_bottom_center()
        self._show_on_top()

    def show_status(self, text: str) -> None:
        """Swap the waveform for a short status message (e.g. 'Transcribing…')."""
        self.waveform.stop()
        self.waveform.hide()
        self.status_label.setText(text)
        self.status_label.show()
        self._move_to_bottom_center()
        self._show_on_top()

    def hide_overlay(self) -> None:
        self.waveform.stop()
        self._topmost_timer.stop()
        self.hide()

    @Slot(float)
    def on_level(self, level: float) -> None:
        self.waveform.push_level(level)


def _demo() -> None:
    """Standalone visual check: animates fake levels, then a status message."""
    import math
    import sys

    from PySide6.QtCore import QTimer

    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show_recording()

    tick = 0

    def fake_level():
        nonlocal tick
        tick += 1
        overlay.on_level(abs(math.sin(tick / 3.0)) * 0.8 + 0.1)
        if tick == 40:
            overlay.show_status("Transcribing…")
        if tick == 60:
            overlay.hide_overlay()
            app.quit()

    timer = QTimer()
    timer.timeout.connect(fake_level)
    timer.start(100)
    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()
