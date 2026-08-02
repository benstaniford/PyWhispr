"""Frameless always-on-top overlay shown while recording/transcribing.

Run ``python -m pywhispr.ui.overlay`` for a standalone demo with fake levels.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget

from pywhispr.ui.waveform import WaveformWidget

log = logging.getLogger(__name__)

PILL_WIDTH = 280
PILL_HEIGHT = 64
BOTTOM_MARGIN = 48
BACKGROUND = QColor(20, 20, 24, 235)

# Other always-on-top windows keep re-claiming the top of the topmost band (a
# docked Deckmaster does it whenever it is activated), so one claim at show()
# time is not enough - we re-claim for as long as we are visible.
TOPMOST_RE_ASSERT_MS = 500


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

        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(TOPMOST_RE_ASSERT_MS)
        self._topmost_timer.timeout.connect(self._keep_on_top)

    def _keep_on_top(self) -> None:
        """Put the pill at the very top of the z-order, without taking focus.

        ``WindowStaysOnTopHint`` only gets us into the topmost *band*; inside it
        the order is whoever claimed it last, so a docked Deckmaster (also
        topmost) sits above us. ``HWND_TOPMOST`` does not help - for a window
        that is already topmost it is a no-op for ordering - so the claim has to
        be ``HWND_TOP``. ``SWP_NOACTIVATE`` is what keeps this passive: the
        overlay must never pull focus off whatever the user is typing into.
        """
        # Off Windows (incl. the offscreen platform) there is no usable HWND
        # behind winId(), and handing that to SetWindowPos crashes.
        if QGuiApplication.platformName() != "windows":
            return
        try:
            import ctypes

            HWND_TOP = 0
            HWND_TOPMOST = -1
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            # First re-establish WS_EX_TOPMOST (a hide/show cycle or a foreign
            # SetWindowPos can drop it), then order within the band.
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
            user32.SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, flags)
        except Exception:
            log.debug("Could not raise the overlay to the top", exc_info=True)

    def _show_on_top(self) -> None:
        self.show()
        self._keep_on_top()
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
