"""Animated audio-level bars for the recording overlay."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

BAR_COUNT = 24
BAR_COLOR = QColor(255, 255, 255, 230)
IDLE_FRACTION = 0.08  # minimum bar height so silence still shows a pulse line

FRAME_MS = 16  # ~60 fps: the motion is the point, so don't skimp here
LEVEL_GAIN = 1.9  # rms_level rarely approaches 1.0 at normal speaking volume
LEVEL_KNEE = 0.75  # <1 expands the quiet end, so ordinary speech fills the pill
LEVEL_ATTACK = 0.60  # how fast the smoothed level rises towards a new peak
LEVEL_RELEASE = 0.30  # …and how fast it falls back once you stop
BAR_ATTACK = 0.55
BAR_RELEASE = 0.38
WOBBLE_DEPTH = 0.22  # per-bar oscillation, so a steady tone still looks alive


def _profile(i: int) -> float:
    """Bell weighting: the middle bars swing furthest, the ends stay short."""
    x = (i - (BAR_COUNT - 1) / 2) / ((BAR_COUNT - 1) / 2)  # -1..1
    return 0.35 + 0.65 * math.cos(x * math.pi / 2) ** 1.5


class WaveformWidget(QWidget):
    """Bars fixed in place, each easing towards a height driven by mic level.

    Nothing scrolls: a level update retargets every bar at once and the
    animation timer interpolates, so the shape breathes instead of marching
    right-to-left.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._profile = [_profile(i) for i in range(BAR_COUNT)]
        # Irrational-ish speeds and offsets keep the bars from moving in lockstep.
        self._speeds = [3.1 + 1.7 * ((i * 0.618) % 1.0) for i in range(BAR_COUNT)]
        self._phases = [(i * 2.399) % (2 * math.pi) for i in range(BAR_COUNT)]
        self._heights = [IDLE_FRACTION] * BAR_COUNT
        self._level = 0.0
        self._target_level = 0.0
        self._clock = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._heights = [IDLE_FRACTION] * BAR_COUNT
        self._level = self._target_level = 0.0
        self._clock = 0.0
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    @Slot(float)
    def push_level(self, level: float) -> None:
        level = max(0.0, min(1.0, level))
        self._target_level = min(1.0, (level * LEVEL_GAIN) ** LEVEL_KNEE)

    def _tick(self) -> None:
        self._clock += FRAME_MS / 1000.0
        rising = self._target_level > self._level
        self._level += (self._target_level - self._level) * (
            LEVEL_ATTACK if rising else LEVEL_RELEASE
        )

        for i in range(BAR_COUNT):
            wobble = 1.0 + WOBBLE_DEPTH * math.sin(
                self._clock * self._speeds[i] + self._phases[i]
            )
            target = max(IDLE_FRACTION, self._level * self._profile[i] * wobble)
            target = min(1.0, target)
            rate = BAR_ATTACK if target > self._heights[i] else BAR_RELEASE
            self._heights[i] += (target - self._heights[i]) * rate

        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BAR_COLOR)

        w, h = self.width(), self.height()
        slot = w / BAR_COUNT
        bar_w = max(2.0, slot * 0.55)
        for i, frac in enumerate(self._heights):
            bar_h = max(bar_w, frac * h)
            x = i * slot + (slot - bar_w) / 2
            y = (h - bar_h) / 2
            painter.drawRoundedRect(x, y, bar_w, bar_h, bar_w / 2, bar_w / 2)
