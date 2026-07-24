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


DOUBLE_TAP_PREFIX = "double-tap:"
DOUBLE_TAP_INTERVAL = 0.4  # max seconds between the two taps

_PYNPUT_MODIFIER_SETS = {
    "<cmd>": ("cmd", "cmd_l", "cmd_r"),
    "<ctrl>": ("ctrl", "ctrl_l", "ctrl_r"),
    "<alt>": ("alt", "alt_l", "alt_r", "alt_gr"),
    "<shift>": ("shift", "shift_l", "shift_r"),
}


def _double_tap_token(hotkey: str) -> str:
    token = hotkey[len(DOUBLE_TAP_PREFIX) :].strip().lower()
    if token not in _PYNPUT_MODIFIER_SETS:
        raise ValueError(
            f"Unsupported double-tap key {token!r}; use one of {list(_PYNPUT_MODIFIER_SETS)}"
        )
    return token


class DoubleTapListener(HotkeyListener):
    """Fires when a modifier key is tapped twice in quick succession.

    Watches raw key events via pynput, so unlike the chord listeners this
    requires the Input Monitoring permission on macOS. A tap pair only counts
    if no other key was pressed in between (so e.g. rapid Cmd+C, Cmd+V does
    not trigger a cmd double-tap).
    """

    def __init__(self, hotkey: str, on_toggle: Callable[[], None]):
        super().__init__(hotkey, on_toggle)
        self._token = _double_tap_token(hotkey)
        self._listener = None
        self._last_tap = 0.0
        self._interrupted = False  # another key was pressed since the last tap
        self._held = False  # guards against OS key-repeat while held

    def _handle_press(self, is_target: bool, now: float) -> None:
        if not is_target:
            self._interrupted = True
            return
        if self._held:
            return
        self._held = True
        if not self._interrupted and now - self._last_tap <= DOUBLE_TAP_INTERVAL:
            self._last_tap = 0.0
            self._fire()
        else:
            self._last_tap = now
        self._interrupted = False

    def start(self) -> None:
        from pynput import keyboard

        targets = {
            getattr(keyboard.Key, name)
            for name in _PYNPUT_MODIFIER_SETS[self._token]
            if hasattr(keyboard.Key, name)
        }

        def on_press(key):
            self._handle_press(key in targets, time.monotonic())

        def on_release(key):
            if key in targets:
                self._held = False

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        log.info("Listening for double-tap of %s (pynput)", self._token)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def create_hotkey_listener(hotkey: str, on_toggle: Callable[[], None]) -> HotkeyListener:
    if hotkey.startswith(DOUBLE_TAP_PREFIX):
        return DoubleTapListener(hotkey, on_toggle)
    if sys.platform == "darwin":
        return MacHotkeyListener(hotkey, on_toggle)
    return PynputHotkeyListener(hotkey, on_toggle)


def validate_chord(hotkey: str) -> None:
    """Raise ValueError if this platform's listener can't register the chord."""
    if hotkey.startswith(DOUBLE_TAP_PREFIX):
        _double_tap_token(hotkey)
    elif sys.platform == "darwin":
        parse_mac_chord(hotkey)
    else:
        from pynput import keyboard

        keyboard.HotKey.parse(hotkey)
        tokens = [t.strip().lower() for t in hotkey.split("+")]
        if not any(t in _MAC_MODIFIER_NAMES for t in tokens):
            raise ValueError(f"Hotkey {hotkey!r} needs at least one modifier (cmd/ctrl/alt/shift)")


def pretty_chord(chord: str) -> str:
    """'<cmd>+<shift>+<space>' → 'Cmd+Shift+Space'; 'double-tap:<alt>' → 'Double-tap Alt'."""
    if chord.startswith(DOUBLE_TAP_PREFIX):
        token = chord[len(DOUBLE_TAP_PREFIX) :]
        return f"Double-tap {token.strip('<>').title()}"
    return "+".join(t.strip("<>").replace("_", " ").title() for t in chord.split("+"))
