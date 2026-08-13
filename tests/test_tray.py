from unittest.mock import MagicMock

import pytest

from pywhispr.tray import SETTINGS_TEXT, TrayIcon


def labels(tray):
    return [action.text() for action in tray.contextMenu().actions() if action.text()]


@pytest.fixture
def build(qapp):
    """A real TrayIcon. Never shown: an offscreen tray needs nothing on screen."""
    icons = []

    def make(**kwargs):
        icon = TrayIcon(on_quit=MagicMock(), **kwargs)
        icons.append(icon)  # kept alive for the duration of the test
        return icon

    yield make


class TestWhatIsLeftInTheMenu:
    def test_only_the_things_a_tray_is_for(self, build):
        tray = build(
            on_toggle=MagicMock(), on_show_history=MagicMock(), on_settings=MagicMock()
        )
        assert labels(tray) == [
            "Start/stop dictation",
            "Recent dictations…",
            SETTINGS_TEXT,
            "Quit PyWhispr",
        ]

    def test_the_settings_that_used_to_live_here_are_gone(self, build):
        """Every one of these moved to the settings page."""
        text = " ".join(labels(build(on_toggle=MagicMock(), on_settings=MagicMock())))
        for gone in ("hotkey", "Vocabulary", "plugins", "GPU", "config file", "log file"):
            assert gone not in text

    def test_quit_is_always_there(self, build):
        assert "Quit PyWhispr" in labels(build())


class TestWhatClickingDoes:
    def test_settings_opens_the_settings_page(self, build):
        on_settings = MagicMock()
        tray = build(on_settings=on_settings)
        for action in tray.contextMenu().actions():
            if action.text() == SETTINGS_TEXT:
                action.trigger()
        on_settings.assert_called_once()

    def test_no_settings_handler_leaves_the_entry_out(self, build):
        assert SETTINGS_TEXT not in labels(build())
