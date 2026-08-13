from unittest.mock import MagicMock, patch

import pytest

from pywhispr.config import Config
from pywhispr.ui.settings_dialog import MISSING_SUFFIX, SYSTEM_DEFAULT, SettingsDialog

DEVICES = [(0, "Microphone Array"), (1, "Yeti Stereo Microphone")]


@pytest.fixture
def devices():
    with patch("pywhispr.ui.settings_dialog.input_devices", return_value=DEVICES):
        yield DEVICES


@pytest.fixture
def build(qapp, devices):
    dialogs = []

    def make(cfg=None, **actions):
        dialog = SettingsDialog(cfg or Config(), **actions)
        dialogs.append(dialog)  # kept alive for the duration of the test
        return dialog

    yield make


class TestMicrophoneList:
    def test_the_system_default_is_always_first(self, build):
        dialog = build()
        assert dialog._mic.itemText(0) == SYSTEM_DEFAULT
        assert dialog._mic.itemData(0) is None
        assert dialog._mic.currentIndex() == 0

    def test_every_input_device_is_listed(self, build):
        dialog = build()
        listed = [dialog._mic.itemText(i) for i in range(dialog._mic.count())]
        assert listed == [SYSTEM_DEFAULT, "Microphone Array", "Yeti Stereo Microphone"]

    def test_the_saved_choice_is_selected(self, build):
        dialog = build(Config(input_device_name="Yeti Stereo Microphone"))
        assert dialog._mic.currentData() == "Yeti Stereo Microphone"

    def test_a_legacy_index_selects_the_device_it_points_at(self, build):
        dialog = build(Config(input_device=1))
        assert dialog._mic.currentData() == "Yeti Stereo Microphone"

    def test_an_unplugged_choice_is_kept_and_marked(self, build):
        """Opening settings while the device is out must not silently reset it."""
        dialog = build(Config(input_device_name="Podcaster"))
        assert dialog._mic.currentData() == "Podcaster"
        assert dialog._mic.currentText() == "Podcaster" + MISSING_SUFFIX
        assert "not connected" in dialog._mic_note.text()

    def test_no_note_when_the_choice_is_present(self, build):
        assert build(Config(input_device_name="Microphone Array"))._mic_note.text() == ""

    def test_no_devices_at_all_says_so(self, qapp):
        with patch("pywhispr.ui.settings_dialog.input_devices", return_value=[]):
            dialog = SettingsDialog(Config())
        assert "No input devices" in dialog._mic_note.text()


class TestSaving:
    def _saved(self, dialog) -> Config:
        with patch.object(dialog, "accept"):
            dialog._save()
        return dialog.config

    def test_the_microphone_is_saved_by_name(self, build):
        dialog = build()
        dialog._mic.setCurrentIndex(dialog._mic.findData("Yeti Stereo Microphone"))
        cfg = self._saved(dialog)
        assert cfg.input_device_name == "Yeti Stereo Microphone"

    def test_choosing_the_default_clears_a_legacy_index_too(self, build):
        """Otherwise the stale index would outlive the choice made here."""
        dialog = build(Config(input_device=1))
        dialog._mic.setCurrentIndex(0)
        cfg = self._saved(dialog)
        assert cfg.input_device_name is None
        assert cfg.input_device is None

    def test_the_toggles_round_trip(self, build):
        dialog = build(Config(remove_fillers=True, play_sounds=True))
        dialog._remove_fillers.setChecked(False)
        cfg = self._saved(dialog)
        assert cfg.remove_fillers is False
        assert cfg.play_sounds is True  # untouched

    def test_reset_phrases_are_a_comma_separated_list(self, build):
        dialog = build()
        dialog._reset_phrases.setText("clear clear, scratch scratch ,")
        assert self._saved(dialog).voice_reset_phrases == ["clear clear", "scratch scratch"]

    def test_an_empty_phrase_list_turns_the_feature_off(self, build):
        dialog = build()
        dialog._reset_phrases.setText("  ")
        assert self._saved(dialog).voice_reset_phrases == []

    def test_an_empty_api_host_falls_back_to_the_default(self, build):
        dialog = build()
        dialog._api_host.setText("")
        assert self._saved(dialog).api_host == "0.0.0.0"

    def test_cancelling_leaves_the_original_config_alone(self, build):
        cfg = Config(remove_fillers=True)
        dialog = build(cfg)
        dialog._remove_fillers.setChecked(False)
        assert cfg.remove_fillers is True  # the dialog edits a copy


class TestActions:
    def test_the_hotkey_row_shows_and_records_a_new_chord(self, build):
        capture = MagicMock(return_value="<ctrl>+<alt>+j")
        dialog = build(on_change_hotkey=capture)
        dialog._change_hotkey()
        assert dialog.config.hotkey == "<ctrl>+<alt>+j"
        assert dialog._hotkey_label.text() == "<ctrl>+<alt>+j"

    def test_a_cancelled_capture_changes_nothing(self, build):
        dialog = build(on_change_hotkey=MagicMock(return_value=None))
        before = dialog.config.hotkey
        dialog._change_hotkey()
        assert dialog.config.hotkey == before

    def test_no_gpu_handler_means_no_gpu_row(self, build):
        assert not hasattr(build(), "_gpu_button")

    def test_the_gpu_button_reads_the_way_round_it_is(self, build):
        dialog = build(on_enable_gpu=MagicMock(), gpu_active=lambda: True)
        assert dialog._gpu_button.text() == "Disable…"

    def test_a_predicate_that_raises_reads_as_off(self, build):
        def boom():
            raise RuntimeError("no idea")

        dialog = build(on_enable_gpu=MagicMock(), gpu_active=boom)
        assert dialog._gpu_button.text() == "Enable…"

    def test_clicking_asks_again_rather_than_trusting_the_label(self, build):
        state = {"on": False}
        enable, disable = MagicMock(), MagicMock()
        dialog = build(
            on_enable_gpu=enable, on_disable_gpu=disable, gpu_active=lambda: state["on"]
        )
        state["on"] = True
        dialog._gpu_clicked()
        disable.assert_called_once_with()
        enable.assert_not_called()


class TestLiteBuild:
    def test_the_server_field_replaces_the_api_rows(self, qapp, devices, monkeypatch):
        monkeypatch.setattr("pywhispr.flavor.IS_LITE", True)
        dialog = SettingsDialog(Config(server_url="http://box:9149"))
        assert dialog._server_url.text() == "http://box:9149"
        assert not hasattr(dialog, "_api_port")
        dialog._server_url.setText(" http://other:9149 ")
        with patch.object(dialog, "accept"):
            dialog._save()
        assert dialog.config.server_url == "http://other:9149"
