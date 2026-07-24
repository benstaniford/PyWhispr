import sys

import pytest

from pywhispr.hotkey import (
    DOUBLE_TAP_INTERVAL,
    DoubleTapListener,
    DoubleTapState,
    MacDoubleTapListener,
    create_hotkey_listener,
    pretty_chord,
    validate_chord,
)


class TestDoubleTapState:
    def test_two_quick_taps_fire(self):
        s = DoubleTapState()
        assert s.tap(10.0) is False
        assert s.tap(10.2) is True

    def test_slow_taps_do_not_fire(self):
        s = DoubleTapState()
        s.tap(10.0)
        assert s.tap(10.0 + DOUBLE_TAP_INTERVAL + 0.1) is False

    def test_interrupt_cancels_pending_tap(self):
        s = DoubleTapState()
        s.tap(10.0)
        s.interrupt()
        assert s.tap(10.1) is False

    def test_interrupted_flag_on_tap(self):
        s = DoubleTapState()
        s.tap(10.0)
        assert s.tap(10.1, interrupted=True) is False

    def test_third_tap_does_not_refire(self):
        s = DoubleTapState()
        s.tap(10.0)
        assert s.tap(10.2) is True
        assert s.tap(10.3) is False  # pair consumed; this starts a new pair


class TestPynputListener:
    def _listener(self):
        fired = []
        return DoubleTapListener("double-tap:<alt>", lambda: fired.append(1)), fired

    def test_two_quick_taps_fire(self):
        lst, fired = self._listener()
        lst._handle_press(True, 10.0)
        lst._held = False
        lst._handle_press(True, 10.2)
        assert fired == [1]

    def test_interrupting_key_resets(self):
        lst, fired = self._listener()
        lst._handle_press(True, 10.0)
        lst._held = False
        lst._handle_press(False, 10.05)  # another key
        lst._handle_press(True, 10.1)
        assert fired == []

    def test_key_repeat_while_held_ignored(self):
        lst, fired = self._listener()
        lst._handle_press(True, 10.0)
        lst._handle_press(True, 10.1)  # repeat, no release between
        assert fired == []


class TestMacListener:
    def _listener(self):
        fired = []
        return MacDoubleTapListener("double-tap:<cmd>", lambda: fired.append(1)), fired

    def test_rising_edges_fire(self):
        lst, fired = self._listener()
        cmd = 1 << 20
        lst._on_flags_changed(cmd)  # press
        lst._on_flags_changed(0)  # release
        lst._on_flags_changed(cmd)  # press again
        assert fired == [1]

    def test_held_without_release_does_not_fire(self):
        lst, fired = self._listener()
        cmd = 1 << 20
        lst._on_flags_changed(cmd)
        lst._on_flags_changed(cmd)  # still down, no rising edge
        assert fired == []

    def test_other_modifier_present_blocks_fire(self):
        lst, fired = self._listener()
        cmd, shift = 1 << 20, 1 << 17
        lst._on_flags_changed(cmd)
        lst._on_flags_changed(0)
        lst._on_flags_changed(cmd | shift)  # cmd+shift, not a clean cmd tap
        assert fired == []

    def test_factory_uses_mac_listener_on_darwin(self):
        lst = create_hotkey_listener("double-tap:<alt>", lambda: None)
        expected = MacDoubleTapListener if sys.platform == "darwin" else DoubleTapListener
        assert isinstance(lst, expected)


class TestValidation:
    def test_valid_double_tap_chords(self):
        for mod in ("<cmd>", "<ctrl>", "<alt>", "<shift>"):
            validate_chord(f"double-tap:{mod}")

    def test_invalid_double_tap_key_rejected(self):
        with pytest.raises(ValueError):
            validate_chord("double-tap:<space>")
        with pytest.raises(ValueError):
            validate_chord("double-tap:x")

    def test_pretty_double_tap(self):
        assert pretty_chord("double-tap:<alt>") == "Double-tap Alt"
        assert pretty_chord("double-tap:<cmd>") == "Double-tap Cmd"
