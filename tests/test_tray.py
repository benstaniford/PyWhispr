from unittest.mock import MagicMock

import pytest

from pywhispr import tray as tray_module
from pywhispr.tray import GPU_DISABLE_TEXT, GPU_ENABLE_TEXT, TrayIcon


def gpu_action(tray):
    """The GPU entry, whichever way round it currently reads."""
    for action in tray.contextMenu().actions():
        if action.text() in (GPU_ENABLE_TEXT, GPU_DISABLE_TEXT):
            return action
    return None


def open_menu(tray):
    """What Qt does when the user clicks the tray icon, minus the popup."""
    tray.contextMenu().aboutToShow.emit()


@pytest.fixture
def build(qapp):
    """A real TrayIcon. Never shown: an offscreen tray needs nothing on screen."""
    icons = []

    def make(**kwargs):
        icon = TrayIcon(on_quit=MagicMock(), **kwargs)
        icons.append(icon)  # kept alive for the duration of the test
        return icon

    yield make


class TestWhetherTheEntryIsThere:
    def test_absent_where_no_gpu_path_could_run(self, build):
        """macOS passes no handler, and then there must be no entry at all."""
        assert gpu_action(build()) is None

    def test_present_where_one_could(self, build):
        assert gpu_action(build(on_enable_gpu=MagicMock())) is not None

    def test_the_other_entries_are_untouched(self, build):
        labels = [a.text() for a in build().contextMenu().actions()]
        assert "Open config file" in labels
        assert "Quit PyWhispr" in labels


class TestWhichWayRoundItReads:
    def test_it_offers_to_enable_when_acceleration_is_off(self, build):
        tray = build(on_enable_gpu=MagicMock(), gpu_active=lambda: False)
        open_menu(tray)
        assert gpu_action(tray).text() == GPU_ENABLE_TEXT

    def test_it_offers_to_disable_once_acceleration_is_on(self, build):
        tray = build(on_enable_gpu=MagicMock(), gpu_active=lambda: True)
        open_menu(tray)
        assert gpu_action(tray).text() == GPU_DISABLE_TEXT

    def test_it_follows_the_answer_between_openings(self, build):
        """The point of asking on aboutToShow: enabling it elsewhere retitles it."""
        state = {"on": False}
        tray = build(on_enable_gpu=MagicMock(), gpu_active=lambda: state["on"])
        open_menu(tray)
        assert gpu_action(tray).text() == GPU_ENABLE_TEXT
        state["on"] = True
        open_menu(tray)
        assert gpu_action(tray).text() == GPU_DISABLE_TEXT

    def test_no_predicate_reads_as_off(self, build):
        tray = build(on_enable_gpu=MagicMock())
        open_menu(tray)
        assert gpu_action(tray).text() == GPU_ENABLE_TEXT

    def test_a_predicate_that_raises_leaves_the_menu_usable(self, build):
        """A menu that will not open is the whole UI; "off" is the safe way to be wrong."""

        def boom():
            raise RuntimeError("no idea")

        tray = build(on_enable_gpu=MagicMock(), gpu_active=boom)
        open_menu(tray)
        assert gpu_action(tray).text() == GPU_ENABLE_TEXT


class TestWhatClickingDoes:
    def test_clicking_enables_when_acceleration_is_off(self, build):
        enable, disable = MagicMock(), MagicMock()
        tray = build(on_enable_gpu=enable, on_disable_gpu=disable, gpu_active=lambda: False)
        gpu_action(tray).trigger()
        enable.assert_called_once_with()
        disable.assert_not_called()

    def test_clicking_disables_when_acceleration_is_on(self, build):
        enable, disable = MagicMock(), MagicMock()
        tray = build(on_enable_gpu=enable, on_disable_gpu=disable, gpu_active=lambda: True)
        gpu_action(tray).trigger()
        disable.assert_called_once_with()
        enable.assert_not_called()

    def test_the_handler_is_called_with_no_arguments(self, build):
        """triggered(bool checked = false), and PySide6 passes it to whatever will
        take it. Connected straight to app._enable_gpu that meant
        asked_by_user=False on every click, which offered a CUDA download on
        machines that have no CUDA."""
        enable = MagicMock()
        tray = build(on_enable_gpu=enable, gpu_active=lambda: False)
        gpu_action(tray).trigger()
        assert enable.call_args.args == ()
        assert enable.call_args.kwargs == {}

    def test_a_click_with_nothing_wired_to_disable_does_nothing(self, build):
        tray = build(on_enable_gpu=MagicMock(), gpu_active=lambda: True)
        gpu_action(tray).trigger()  # must not raise

    def test_it_asks_again_at_click_time_rather_than_trusting_the_label(self, build):
        """Menu opened, acceleration enabled elsewhere, then the stale row clicked."""
        state = {"on": False}
        enable, disable = MagicMock(), MagicMock()
        tray = build(
            on_enable_gpu=enable, on_disable_gpu=disable, gpu_active=lambda: state["on"]
        )
        open_menu(tray)
        state["on"] = True
        gpu_action(tray).trigger()
        disable.assert_called_once_with()
        enable.assert_not_called()


def test_the_labels_say_what_they_do():
    assert tray_module.GPU_ENABLE_TEXT.startswith("Enable GPU")
    assert tray_module.GPU_DISABLE_TEXT.startswith("Disable GPU")
