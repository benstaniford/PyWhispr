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


class DoubleTapState:
    """Timing logic for a modifier double-tap, independent of the event source.

    Feed it discrete taps of the target modifier (with a flag saying whether
    another key intervened). Returns True on the tap that completes a pair.
    """

    def __init__(self, interval: float = DOUBLE_TAP_INTERVAL):
        self._interval = interval
        self._last_tap = 0.0

    def tap(self, now: float, interrupted: bool = False) -> bool:
        if not interrupted and 0.0 < now - self._last_tap <= self._interval:
            self._last_tap = 0.0  # consume, so a third tap doesn't re-fire
            return True
        self._last_tap = now
        return False

    def interrupt(self) -> None:
        """Another key was pressed; cancel any pending first tap."""
        self._last_tap = 0.0


class DoubleTapGesture:
    """Turns raw modifier press/release events into activate/release calls.

    ``on_activate`` fires when a double-tap is detected (on the second press).
    ``on_release`` fires when that second (activating) tap is released, with
    how long it was held — letting the app offer push-to-talk: hold the second
    tap and release to stop, or tap quickly to leave recording latched.
    """

    def __init__(self, on_activate, on_release, interval: float = DOUBLE_TAP_INTERVAL):
        self._on_activate = on_activate
        self._on_release = on_release
        self._state = DoubleTapState(interval)
        self._armed = False  # a double-tap fired; watching the activating tap's release
        self._press_time = 0.0

    def press(self, now: float, interrupted: bool = False) -> None:
        if self._state.tap(now, interrupted):
            self._armed = True
            self._press_time = now
            self._on_activate()

    def release(self, now: float) -> None:
        if self._armed:
            self._armed = False
            if self._on_release is not None:
                self._on_release(now - self._press_time)

    def interrupt(self) -> None:
        self._state.interrupt()


class DoubleTapListener(HotkeyListener):
    """Double-tap detection via pynput (Windows/Linux).

    Requires no special permission on Windows. Not used on macOS — pynput's
    keyboard.Listener builds its keycode map with Carbon HIToolbox calls off
    the main thread (keycode_context), which segfaults when combined with our
    Carbon hotkey registration; MacDoubleTapListener is used there instead.
    """

    def __init__(
        self,
        hotkey: str,
        on_toggle: Callable[[], None],
        on_release: Callable[[float], None] | None = None,
    ):
        super().__init__(hotkey, on_toggle)
        self._token = _double_tap_token(hotkey)
        self._listener = None
        self._gesture = DoubleTapGesture(self._fire, on_release)
        self._held = False  # guards against OS key-repeat while held

    def _handle_press(self, is_target: bool, now: float) -> None:
        if not is_target:
            self._gesture.interrupt()
            return
        if self._held:
            return
        self._held = True
        self._gesture.press(now)

    def _handle_release(self, is_target: bool, now: float) -> None:
        if is_target:
            self._held = False
            self._gesture.release(now)

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
            self._handle_release(key in targets, time.monotonic())

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        log.info("Listening for double-tap of %s (pynput)", self._token)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


# Device-independent NSEvent modifier flags (AppKit).
_NS_MODIFIER_FLAGS = {
    "<shift>": 1 << 17,
    "<ctrl>": 1 << 18,
    "<alt>": 1 << 19,
    "<cmd>": 1 << 20,
}
_NS_MOD_MASK = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20)


class MacDoubleTapListener(HotkeyListener):
    """Double-tap detection on macOS via a Cocoa NSEvent global monitor.

    The handler runs on the main thread's Cocoa run loop (driven by Qt), so it
    avoids pynput's off-main-thread Carbon keycode setup entirely. Global
    monitors need the Input Monitoring permission; without it the handler
    simply never fires (no crash).
    """

    def __init__(
        self,
        hotkey: str,
        on_toggle: Callable[[], None],
        on_release: Callable[[float], None] | None = None,
    ):
        super().__init__(hotkey, on_toggle)
        self._token = _double_tap_token(hotkey)
        self._flag = _NS_MODIFIER_FLAGS[self._token]
        self._monitor = None
        self._gesture = DoubleTapGesture(self._fire, on_release)
        self._was_down = False

    def _on_flags_changed(self, flags: int) -> None:
        target_down = bool(flags & self._flag)
        other_mods_down = bool(flags & _NS_MOD_MASK & ~self._flag)
        now = time.monotonic()
        if target_down and not self._was_down:  # rising edge = a tap
            self._gesture.press(now, interrupted=other_mods_down)
        elif not target_down and self._was_down:  # falling edge = release
            self._gesture.release(now)
        self._was_down = target_down

    def start(self) -> None:
        from AppKit import NSEvent, NSEventMaskFlagsChanged, NSEventMaskKeyDown

        def handler(event):
            if event.type() == 10:  # NSEventTypeKeyDown: a normal key breaks the pair
                self._gesture.interrupt()
            else:
                self._on_flags_changed(int(event.modifierFlags()))

        # Keep a reference: the monitor object is the block's owner.
        self._monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskFlagsChanged | NSEventMaskKeyDown, handler
        )
        log.info("Listening for double-tap of %s (NSEvent monitor)", self._token)

    def stop(self) -> None:
        if self._monitor is not None:
            from AppKit import NSEvent

            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None


def create_hotkey_listener(
    hotkey: str,
    on_toggle: Callable[[], None],
    on_release: Callable[[float], None] | None = None,
) -> HotkeyListener:
    """Build the right listener for ``hotkey``.

    ``on_release(held_seconds)`` is invoked by double-tap listeners when the
    activating (second) tap is released, enabling push-to-talk; other listener
    types ignore it.
    """
    if hotkey.startswith(DOUBLE_TAP_PREFIX):
        if sys.platform == "darwin":
            return MacDoubleTapListener(hotkey, on_toggle, on_release)
        return DoubleTapListener(hotkey, on_toggle, on_release)
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
