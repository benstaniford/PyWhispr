import pytest

from pywhispr.hotkey import (
    DOUBLE_TAP_INTERVAL,
    DoubleTapListener,
    pretty_chord,
    validate_chord,
)


@pytest.fixture
def listener():
    fired = []
    lst = DoubleTapListener("double-tap:<alt>", lambda: fired.append(True))
    return lst, fired


class TestDoubleTapLogic:
    def test_two_quick_taps_fire(self, listener):
        lst, fired = listener
        lst._handle_press(True, 10.0)
        lst._held = False
        lst._handle_press(True, 10.2)
        assert fired == [True]

    def test_slow_taps_do_not_fire(self, listener):
        lst, fired = listener
        lst._handle_press(True, 10.0)
        lst._held = False
        lst._handle_press(True, 10.0 + DOUBLE_TAP_INTERVAL + 0.1)
        assert fired == []

    def test_interrupting_key_resets(self, listener):
        # Cmd+C then Cmd+V quickly must not read as a double-tap.
        lst, fired = listener
        lst._handle_press(True, 10.0)
        lst._held = False
        lst._handle_press(False, 10.05)  # the 'c'
        lst._handle_press(True, 10.1)
        assert fired == []

    def test_key_repeat_while_held_ignored(self, listener):
        lst, fired = listener
        lst._handle_press(True, 10.0)
        lst._handle_press(True, 10.1)  # repeat: no release in between
        assert fired == []

    def test_third_tap_does_not_double_fire(self, listener):
        lst, fired = listener
        for t in (10.0, 10.2, 10.3):
            lst._handle_press(True, t)
            lst._held = False
        assert fired == [True]


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
