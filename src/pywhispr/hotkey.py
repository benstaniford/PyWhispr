"""Global hotkey listening.

On macOS the Carbon ``RegisterEventHotKey`` API (via quickmachotkey) is used:
it registers one specific chord with the OS and therefore needs **no** Input
Monitoring or Accessibility permission — unlike pynput, which sniffs every
keystroke. Windows/Linux keep pynput.

Chord strings use pynput's syntax everywhere (e.g. ``<cmd>+<shift>+<space>``)
so the config file stays portable across platforms.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.3  # swallow OS key-repeat while the chord is held


class HotkeyListener:
    """Fires ``on_toggle`` each time the configured chord is pressed.

    The callback may run on a background thread (pynput) or the main event
    loop (macOS) — relay it through a Qt signal before touching UI state.
    """

    def __init__(self, hotkey: str, on_toggle: Callable[[], None]):
        self._hotkey = hotkey
        self._on_toggle = on_toggle
        self._last_fired = 0.0

    def _fire(self) -> None:
        now = time.monotonic()
        if now - self._last_fired < DEBOUNCE_SECONDS:
            return
        self._last_fired = now
        self._on_toggle()

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class PynputHotkeyListener(HotkeyListener):
    """pynput GlobalHotKeys — Windows/Linux (needs no special permissions there)."""

    def __init__(self, hotkey: str, on_toggle: Callable[[], None]):
        super().__init__(hotkey, on_toggle)
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        # Validates the chord string (raises ValueError on bad syntax).
        keyboard.HotKey.parse(self._hotkey)
        self._listener = keyboard.GlobalHotKeys({self._hotkey: self._fire})
        self._listener.start()
        log.info("Listening for hotkey %s (pynput)", self._hotkey)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


# pynput-style token → macOS virtual key code constant name (quickmachotkey.constants)
_MAC_KEY_NAMES = {
    "<space>": "kVK_Space",
    "<tab>": "kVK_Tab",
    "<enter>": "kVK_Return",
    "<return>": "kVK_Return",
    "<esc>": "kVK_Escape",
    "<backspace>": "kVK_Delete",
    "<delete>": "kVK_ForwardDelete",
    "<home>": "kVK_Home",
    "<end>": "kVK_End",
    "<page_up>": "kVK_PageUp",
    "<page_down>": "kVK_PageDown",
    "<up>": "kVK_UpArrow",
    "<down>": "kVK_DownArrow",
    "<left>": "kVK_LeftArrow",
    "<right>": "kVK_RightArrow",
    **{f"<f{n}>": f"kVK_F{n}" for n in range(1, 21)},
}

_MAC_MODIFIER_NAMES = {
    "<cmd>": "cmdKey",
    "<ctrl>": "controlKey",
    "<alt>": "optionKey",
    "<shift>": "shiftKey",
}


def parse_mac_chord(hotkey: str) -> tuple[int, int]:
    """Translate a pynput-style chord into (virtualKey, modifierMask) for Carbon."""
    from quickmachotkey import constants, mask

    modifiers = []
    key_code: int | None = None
    for token in (t.strip().lower() for t in hotkey.split("+")):
        if token in _MAC_MODIFIER_NAMES:
            modifiers.append(getattr(constants, _MAC_MODIFIER_NAMES[token]))
        elif token in _MAC_KEY_NAMES:
            key_code = getattr(constants, _MAC_KEY_NAMES[token])
        elif len(token) == 1 and (token.isascii() and (token.isalpha() or token.isdigit())):
            key_code = getattr(constants, f"kVK_ANSI_{token.upper()}")
        else:
            raise ValueError(f"Unsupported key {token!r} in hotkey {hotkey!r}")
    if key_code is None:
        raise ValueError(f"Hotkey {hotkey!r} has no non-modifier key")
    if not modifiers:
        raise ValueError(f"Hotkey {hotkey!r} needs at least one modifier (cmd/ctrl/alt/shift)")
    return key_code, mask(*modifiers)


class MacHotkeyListener(HotkeyListener):
    """Carbon RegisterEventHotKey — permission-free on macOS.

    The handler fires on the process's main CFRunLoop, which Qt drives, so
    it runs on the Qt main thread.
    """

    def __init__(self, hotkey: str, on_toggle: Callable[[], None]):
        super().__init__(hotkey, on_toggle)
        self._registration = None

    def start(self) -> None:
        from quickmachotkey import quickHotKey

        virtual_key, modifier_mask = parse_mac_chord(self._hotkey)

        def handler() -> None:  # quickHotKey sets attributes on the callable,
            self._fire()  # which fails for bound methods

        self._registration = quickHotKey(
            virtualKey=virtual_key, modifierMask=modifier_mask
        )(handler)
        log.info("Listening for hotkey %s (RegisterEventHotKey)", self._hotkey)

    def stop(self) -> None:
        if self._registration is not None:
            self._registration.unregister()
            self._registration = None


def create_hotkey_listener(hotkey: str, on_toggle: Callable[[], None]) -> HotkeyListener:
    if sys.platform == "darwin":
        return MacHotkeyListener(hotkey, on_toggle)
    return PynputHotkeyListener(hotkey, on_toggle)


def validate_chord(hotkey: str) -> None:
    """Raise ValueError if this platform's listener can't register the chord."""
    if sys.platform == "darwin":
        parse_mac_chord(hotkey)
    else:
        from pynput import keyboard

        keyboard.HotKey.parse(hotkey)
        tokens = [t.strip().lower() for t in hotkey.split("+")]
        if not any(t in _MAC_MODIFIER_NAMES for t in tokens):
            raise ValueError(f"Hotkey {hotkey!r} needs at least one modifier (cmd/ctrl/alt/shift)")
