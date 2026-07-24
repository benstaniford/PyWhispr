"""Animated audio-level bars for the recording overlay."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

BAR_COUNT = 24
BAR_COLOR = QColor(255, 255, 255, 230)
IDLE_FRACTION = 0.08  # minimum bar height so silence still shows a pulse line


class WaveformWidget(QWidget):
    """Scrolling bars driven by mic level updates (0..1), Superwhisper style."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._levels: deque[float] = deque([0.0] * BAR_COUNT, maxlen=BAR_COUNT)
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(33)  # ~30 fps
        self._repaint_timer.timeout.connect(self.update)

    def start(self) -> None:
        self._levels.extend([0.0] * BAR_COUNT)
        self._repaint_timer.start()

    def stop(self) -> None:
        self._repaint_timer.stop()

    @Slot(float)
    def push_level(self, level: float) -> None:
        self._levels.append(max(0.0, min(1.0, level)))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BAR_COLOR)

        w, h = self.width(), self.height()
        slot = w / BAR_COUNT
        bar_w = max(2.0, slot * 0.55)
        for i, level in enumerate(self._levels):
            frac = max(IDLE_FRACTION, level)
            bar_h = max(bar_w, frac * h)
            x = i * slot + (slot - bar_w) / 2
            y = (h - bar_h) / 2
            painter.drawRoundedRect(x, y, bar_w, bar_h, bar_w / 2, bar_w / 2)
