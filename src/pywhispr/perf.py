"""Stage timings for the part of a dictation the existing logs do not cover.

``"Transcription took %.1fs"`` and ``"Transcribed %d characters"`` bracket the
model, and everything after them — hiding the overlay, the filler/vocabulary/join
passes, the clipboard write, the synthesized Ctrl+V, the clipboard restore — is
one unlogged blob. That blob is where "the pill disappeared but the text turned
up two seconds later" happens, so it needs to be broken out.

Off unless ``PYWHISPR_TIMING`` is set, so the packaged app can be asked for the
numbers without a rebuild. One cycle at a time: local dictation is serialised by
the state machine, and the network API never gets here.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

ENV_VAR = "PYWHISPR_TIMING"

# A Qt timer that wanted 100 ms and got this much more has been sitting behind
# something on the main thread — which is a different fault from a slow paste.
LOOP_STALL_MS = 50.0


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() not in ("", "0", "false", "no")


class Stages:
    """Marks named instants and logs them as one line of deltas."""

    def __init__(self, name: str):
        self.name = name
        self._marks: list[tuple[str, float]] = [("start", time.perf_counter())]

    def mark(self, label: str) -> None:
        self._marks.append((label, time.perf_counter()))

    def log(self) -> None:
        start = self._marks[0][1]
        previous = start
        parts = []
        for label, at in self._marks[1:]:
            parts.append(f"{label}=+{(at - previous) * 1000:.0f}ms")
            previous = at
        log.info(
            "%s: %s (total %.0fms)", self.name, " ".join(parts), (previous - start) * 1000
        )


# ponytail: one module-level cycle, because the state machine allows exactly one.
# If the API ever injects text too, this needs to be per-cycle instead.
_current: Stages | None = None


def begin(name: str) -> None:
    global _current
    _current = Stages(name) if enabled() else None


def mark(label: str) -> None:
    if _current is not None:
        _current.mark(label)


def end() -> None:
    global _current
    if _current is not None:
        _current.log()
        _current = None


def install_loop_monitor(parent) -> None:
    """Log whenever the Qt main thread makes a 100 ms timer late.

    Tells apart "the paste itself was slow" from "our own event loop was
    starved", which the QTimer hops in the injector depend on.
    """
    if not enabled():
        return
    from PySide6.QtCore import QTimer

    interval_ms = 100
    state = {"at": time.perf_counter()}

    def tick() -> None:
        now = time.perf_counter()
        late_ms = (now - state["at"]) * 1000 - interval_ms
        state["at"] = now
        if late_ms > LOOP_STALL_MS:
            log.info("Main thread event loop was %.0fms late", late_ms)

    timer = QTimer(parent)
    timer.setInterval(interval_ms)
    timer.timeout.connect(tick)
    timer.start()
