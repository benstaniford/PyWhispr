"""What sits immediately before the caret, so a transcript can be joined onto it.

Two sources, tried in order. On macOS the focused app is asked directly through
the Accessibility API, which stays correct even if the user typed something by
hand between dictations. Plenty of apps refuse to answer — Electron, Java and
terminals mostly do not implement it — so the fallback is to remember the text
we inserted last, thrown away as soon as focus moves to another app or enough
time passes.

Both are best-effort. ``None`` means "no idea, paste verbatim"; an empty string
means "the caret really is at the start of the field", which is real information
and is kept distinct.

The AX read needs the Accessibility permission — the same grant auto-pasting
already requires, so this asks for nothing new. It also means the caret can only
be read from a granted app bundle: ``AXIsProcessTrusted()`` is false for a plain
``python`` run from a terminal, where this quietly returns ``None``.
"""

from __future__ import annotations

import logging
import sys
import time

from pywhispr.platform_setup import check_macos_accessibility

log = logging.getLogger(__name__)

CONTEXT_CHARS = 64
# AX calls are synchronous IPC to the target app's main thread, and we run on
# the GUI thread, so an unresponsive app would freeze the tray and the overlay.
# The default timeout is measured in seconds; this is generous for a healthy app.
AX_TIMEOUT_SECONDS = 0.25
MEMORY_SECONDS = 90.0


def read_preceding_text(max_chars: int = CONTEXT_CHARS) -> str | None:
    """The last ``max_chars`` characters before the caret, or None if unknown."""
    if sys.platform != "darwin":
        # Windows would need UI Automation's TextPattern: a new dependency and a
        # much larger surface. The memory fallback carries it instead.
        return None
    return _read_preceding_macos(max_chars)


def _read_preceding_macos(max_chars: int) -> str | None:
    if not check_macos_accessibility():
        return None
    try:
        # HIServices, not the ApplicationServices umbrella: every AX symbol
        # lives here, and the umbrella additionally pulls in CoreText for
        # nothing. Imported lazily, as elsewhere in this codebase.
        import HIServices as ax
        from CoreFoundation import CFRange

        system = ax.AXUIElementCreateSystemWide()
        # Inherited by the elements we reach through this one.
        ax.AXUIElementSetMessagingTimeout(system, AX_TIMEOUT_SECONDS)

        err, focused = ax.AXUIElementCopyAttributeValue(
            system, ax.kAXFocusedUIElementAttribute, None
        )
        if err != ax.kAXErrorSuccess or focused is None:
            log.debug("No focused accessibility element (err=%s)", err)
            return None

        err, value = ax.AXUIElementCopyAttributeValue(
            focused, ax.kAXSelectedTextRangeAttribute, None
        )
        if err != ax.kAXErrorSuccess or value is None:
            # The everyday case, not a fault: many apps expose no caret at all.
            log.debug("Caret position unavailable (err=%s)", err)
            return None

        ok, caret_range = ax.AXValueGetValue(value, ax.kAXValueCFRangeType, None)
        if not ok or caret_range is None:
            log.debug("Caret range unreadable")
            return None

        location, length = int(caret_range[0]), int(caret_range[1])
        # A selection is about to be replaced by the paste, so what ends up in
        # front of our text is still whatever precedes the selection's start.
        if length:
            log.debug("Caret has a %d-char selection; joining at its start", length)
        if location <= 0:
            return ""

        start = max(0, location - max_chars)
        wanted = ax.AXValueCreate(ax.kAXValueCFRangeType, CFRange(start, location - start))
        err, chunk = ax.AXUIElementCopyParameterizedAttributeValue(
            focused, ax.kAXStringForRangeParameterizedAttribute, wanted, None
        )
        if err != ax.kAXErrorSuccess or chunk is None:
            log.debug("Could not read the text before the caret (err=%s)", err)
            return None

        text = str(chunk)
        # Length only: this could be anything the user has focused, including a
        # password field. It must never reach the log.
        log.debug("Read %d chars of caret context", len(text))
        return text
    except Exception:
        # objc.error, a pyobjc signature surprise, an element that died between
        # calls — none of it is worth more than a debug line.
        log.debug("Accessibility caret read failed", exc_info=True)
        return None


def frontmost_app_id() -> str | None:
    """Identify where the caret is, so remembered context can be invalidated."""
    try:
        if sys.platform == "darwin":
            from AppKit import NSWorkspace

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is None:
                return None
            # Helper processes report no bundle id.
            return app.bundleIdentifier() or app.localizedName() or str(app.processIdentifier())
        if sys.platform == "win32":
            import ctypes

            # Per-window rather than per-process, so moving between two
            # documents of one app counts as a move — which is right, it is a
            # different caret.
            return str(ctypes.windll.user32.GetForegroundWindow())
    except Exception:
        log.debug("Could not identify the frontmost app", exc_info=True)
    return None


class ContextTracker:
    """Best-effort answer to "what is in front of the caret?".

    Asks the focused app first and falls back to what we last inserted. Nothing
    here raises: every path degrades to ``None``, which the caller treats as
    "paste the transcript unchanged".
    """

    def __init__(self, *, max_chars: int = CONTEXT_CHARS, memory_seconds: float = MEMORY_SECONDS):
        self._max_chars = max_chars
        self._memory_seconds = memory_seconds
        self._text: str | None = None
        self._app: str | None = None
        self._at = 0.0

    def preceding_text(self) -> str | None:
        live = read_preceding_text(self._max_chars)
        if live is not None:
            return live  # including "" — the caret is genuinely at the start

        if self._text is None:
            return None
        if time.monotonic() - self._at > self._memory_seconds:
            log.debug("Dropping remembered context: too old")
            self.invalidate()
            return None

        app = frontmost_app_id()
        # Only a positive mismatch counts. On Linux there is no id at all, and
        # treating that as a mismatch would disable the fallback entirely.
        if app is not None and self._app is not None and app != self._app:
            log.debug("Dropping remembered context: focus moved to another app")
            self.invalidate()
            return None
        return self._text

    def remember(self, inserted: str) -> None:
        """Record what we just pasted: the caret now sits directly after it."""
        self._text = inserted[-self._max_chars :]
        self._app = frontmost_app_id()
        self._at = time.monotonic()

    def invalidate(self) -> None:
        self._text = None
        self._app = None
        self._at = 0.0
