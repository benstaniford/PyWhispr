import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("macOS chord parsing tests", allow_module_level=True)

from pywhispr.hotkey import MacHotkeyListener, create_hotkey_listener, parse_mac_chord


def test_default_chord_parses():
    from quickmachotkey.constants import cmdKey, kVK_Space, shiftKey

    key, mods = parse_mac_chord("<cmd>+<shift>+<space>")
    assert key == kVK_Space
    assert mods == cmdKey | shiftKey


def test_letter_digit_and_fkey_chords_parse():
    from quickmachotkey.constants import kVK_ANSI_9, kVK_ANSI_D, kVK_F19

    assert parse_mac_chord("<ctrl>+<alt>+d")[0] == kVK_ANSI_D
    assert parse_mac_chord("<alt>+9")[0] == kVK_ANSI_9
    assert parse_mac_chord("<cmd>+<f19>")[0] == kVK_F19


def test_chord_without_modifier_rejected():
    with pytest.raises(ValueError):
        parse_mac_chord("<space>")


def test_chord_without_key_rejected():
    with pytest.raises(ValueError):
        parse_mac_chord("<cmd>+<shift>")


def test_unknown_token_rejected():
    with pytest.raises(ValueError):
        parse_mac_chord("<cmd>+<media_play>")


def test_factory_picks_carbon_listener_on_macos():
    listener = create_hotkey_listener("<cmd>+<shift>+<space>", lambda: None)
    assert isinstance(listener, MacHotkeyListener)


def test_register_unregister_cycle():
    fired = []
    listener = create_hotkey_listener("<cmd>+<alt>+<f18>", fired.append)
    listener.start()
    listener.stop()
    listener.stop()  # idempotent
