"""A short in-memory history of finished transcripts, for re-pasting one.

Auto-paste goes wherever the keyboard focus happens to be, so a dictation aimed
at a text box that had lost focus lands somewhere useless — and the audio is
gone, so it cannot be re-transcribed. Keeping the last few transcripts costs
nothing and turns that from a loss into a re-paste.

Memory only, deliberately: transcripts are whatever the user said, and the app
already refuses to write that anywhere. They go when the app does.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

# How many transcripts to keep. Bounded on purpose: this is the "that went to
# the wrong window" undo, not a notes app, and every entry is one more line of
# the user's speech sitting in memory. Ten covers noticing several dictations
# later; the picker scrolls rather than growing past MAX_VISIBLE_ROWS.
HISTORY_SIZE = 10

PREVIEW_CHARS = 72


class TranscriptHistory:
    """The last HISTORY_SIZE transcripts, newest first."""

    def __init__(self, size: int = HISTORY_SIZE):
        self._items: deque[str] = deque(maxlen=size)

    def remember(self, text: str) -> None:
        """Record a finished transcript. Blanks and an immediate repeat are ignored."""
        if not text or not text.strip():
            return
        if self._items and self._items[0] == text:
            return  # the same text twice running is one entry, not two
        self._items.appendleft(text)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()


def preview(text: str, width: int = PREVIEW_CHARS) -> str:
    """One line, short enough for a list row: whitespace collapsed, then clipped."""
    line = " ".join(text.split())
    if len(line) <= width:
        return line
    return line[: width - 1].rstrip() + "…"
