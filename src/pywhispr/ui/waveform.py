"""Animated audio-level bars for the recording overlay."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

BAR_COUNT = 24
PAUSE_FRACTION = 0.6  # height of the paused sign's two bars
PAUSE_MS = 380  # edges-to-centre fade on pause; the same, outwards, on resume
PAUSE_SOFTNESS = 0.25  # width of the fading front, as a fraction of the half-width
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
        # Uneven speeds and offsets keep the bars from moving in lockstep.
        self._speeds = [3.1 + 1.7 * ((i * 0.618) % 1.0) for i in range(BAR_COUNT)]
        self._phases = [(i * 2.399) % (2 * math.pi) for i in range(BAR_COUNT)]
        self._heights = [IDLE_FRACTION] * BAR_COUNT
        self._level = self._target_level = 0.0
        self._clock = 0.0
        self._paused = False
        self._fold = 0.0  # 0 = full meter, 1 = folded into the pause sign
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)

    def set_paused(self, paused: bool) -> None:
        """Fold the bars into a pause sign, or unfold them back into the meter.

        The outer bars fade first and the front sweeps to the centre, leaving the
        middle two, which settle at PAUSE_FRACTION; resuming plays it backwards.
        """
        self._paused = paused
        if not self._timer.isActive():
            self._timer.start()

    def _visibility(self, i: int) -> float:
        """0..1 opacity of bar ``i`` at the current point of the fold."""
        mid = BAR_COUNT // 2
        if i in (mid - 1, mid):
            return 1.0
        # 0 at the centre pair, 1 at the ends.
        d = (abs(i - (BAR_COUNT - 1) / 2) - 0.5) / ((BAR_COUNT - 1) / 2 - 0.5)
        p = self._fold * self._fold * (3 - 2 * self._fold)  # smoothstep ease
        front = 1.0 - p * (1.0 + PAUSE_SOFTNESS)  # sweeps from past the edge inwards
        return max(0.0, min(1.0, (front - d) / PAUSE_SOFTNESS + 1.0))

    def start(self) -> None:
        self._heights = [IDLE_FRACTION] * BAR_COUNT
        self._level = self._target_level = 0.0
        self._clock = 0.0
        self._paused = False
        self._fold = 0.0
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

        step = FRAME_MS / PAUSE_MS
        self._fold = min(1.0, self._fold + step) if self._paused else max(0.0, self._fold - step)
        mid = BAR_COUNT // 2
        for i in range(BAR_COUNT):
            if self._fold > 0.0 and i in (mid - 1, mid):
                # The pause sign's bars ease towards its height as the rest fold away.
                target = self._heights[i] + (PAUSE_FRACTION - self._heights[i]) * self._fold
                self._heights[i] += (target - self._heights[i]) * BAR_RELEASE
                continue
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
            alpha = self._visibility(i)
            if alpha <= 0.0:
                continue
            color = QColor(BAR_COLOR)
            color.setAlphaF(BAR_COLOR.alphaF() * alpha)
            painter.setBrush(color)
            bar_h = max(bar_w, frac * h)
            x = i * slot + (slot - bar_w) / 2
            y = (h - bar_h) / 2
            painter.drawRoundedRect(x, y, bar_w, bar_h, bar_w / 2, bar_w / 2)
