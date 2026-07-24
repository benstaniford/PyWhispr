"""Global hotkey listening via pynput."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.3  # swallow OS key-repeat while the chord is held


class HotkeyListener:
    """Fires ``on_toggle`` each time the configured chord is pressed.

    The callback runs on pynput's listener thread — relay it to the main
    thread (e.g. by emitting a Qt signal) before touching any UI state.
    """

    def __init__(self, hotkey: str, on_toggle: Callable[[], None]):
        self._hotkey = hotkey
        self._on_toggle = on_toggle
        self._listener = None
        self._last_fired = 0.0

    def _fire(self) -> None:
        now = time.monotonic()
        if now - self._last_fired < DEBOUNCE_SECONDS:
            return
        self._last_fired = now
        self._on_toggle()

    def start(self) -> None:
        from pynput import keyboard

        # Validates the chord string (raises ValueError on bad syntax).
        keyboard.HotKey.parse(self._hotkey)
        self._listener = keyboard.GlobalHotKeys({self._hotkey: self._fire})
        self._listener.start()
        log.info("Listening for hotkey %s", self._hotkey)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
