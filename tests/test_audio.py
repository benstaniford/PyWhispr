from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pywhispr.audio import AudioRecorder, rms_level


def test_silence_is_zero():
    assert rms_level(np.zeros(1600, dtype=np.float32)) == 0.0


def test_full_scale_sine_is_near_one():
    t = np.arange(1600) / 16000
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert rms_level(sine) > 0.9


def test_quiet_speech_level_is_mid_range():
    t = np.arange(1600) / 16000
    sine = (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    level = rms_level(sine)
    assert 0.2 < level < 0.8


class TestReset:
    """No PortAudio here: the stream object is only started, stopped and closed."""

    def _recording(self) -> AudioRecorder:
        recorder = AudioRecorder()
        recorder._stream = MagicMock()
        recorder._blocks = [np.ones(1600, dtype=np.float32)]
        return recorder

    def test_drops_the_buffer_and_keeps_recording(self):
        recorder = self._recording()
        recorder.reset()
        assert recorder.recording  # still streaming
        recorder._blocks.append(np.full(800, 0.5, dtype=np.float32))  # said afterwards
        audio = recorder.stop()
        assert len(audio) == 800  # only what came after the reset
        assert np.allclose(audio, 0.5)

    def test_reset_when_not_recording_raises(self):
        with pytest.raises(RuntimeError):
            AudioRecorder().reset()


def test_level_is_monotonic_in_amplitude():
    t = np.arange(1600) / 16000
    base = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    levels = [rms_level(a * base) for a in (0.01, 0.05, 0.2, 0.8)]
    assert levels == sorted(levels)
    assert all(0.0 <= lv <= 1.0 for lv in levels)


class TestDeviceLookup:
    """Devices are persisted by name: an index is a position in a list that
    renumbers whenever any other device is unplugged."""

    def _query(self, *names):
        return [
            {"name": name, "max_input_channels": channels} for name, channels in names
        ]

    def test_only_devices_that_can_record_are_listed(self):
        import sys
        from unittest.mock import MagicMock as M

        fake = M()
        fake.query_devices.return_value = self._query(("Speakers", 0), ("Yeti", 2))
        with patch.dict(sys.modules, {"sounddevice": fake}):
            from pywhispr.audio import input_devices

            assert input_devices() == [(1, "Yeti")]

    def test_a_portaudio_that_will_not_answer_is_no_devices(self):
        import sys
        from unittest.mock import MagicMock as M

        fake = M()
        fake.query_devices.side_effect = OSError("no PortAudio")
        with patch.dict(sys.modules, {"sounddevice": fake}):
            from pywhispr.audio import input_devices

            assert input_devices() == []

    def test_find_device_returns_the_current_index(self):
        with self._lists(shown=[(0, "Array"), (3, "Yeti")]):
            from pywhispr.audio import find_device

            assert find_device("Yeti") == 3

    def test_an_absent_device_is_none_rather_than_a_guess(self):
        with self._lists(shown=[(0, "Array")]):
            from pywhispr.audio import find_device

            assert find_device("Yeti") is None

    def _lists(self, shown, every=None):
        """Patch both device lists — leaving one real makes the test depend on
        whatever microphones the machine running it happens to have."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("pywhispr.audio.input_devices", return_value=shown))
        stack.enter_context(
            patch("pywhispr.audio.all_input_devices", return_value=every if every else shown)
        )
        return stack


class TestOneEntryPerDevice:
    """PortAudio lists every microphone once per Windows host API, so the shown
    list is one host API's worth — see PREFERRED_HOST_API."""

    HOST_APIS = ["MME", "Windows DirectSound", "Windows WASAPI", "Windows WDM-KS"]
    DEVICES = [
        ("Microsoft Sound Mapper - Input", 2, 0),
        ("Microphone (Logitech PRO X Wire", 1, 0),  # MME truncates at 31 characters
        ("Primary Sound Capture Driver", 2, 1),
        ("Microphone (Logitech PRO X Wireless Gaming Headset)", 1, 1),
        ("Microphone (Logitech BRIO)", 2, 1),
        ("Microphone (Logitech PRO X Wireless Gaming Headset)", 1, 2),
        ("Microphone (PRO X Wireless Gaming Headset)", 1, 3),
    ]

    def _sounddevice(self, devices=None):
        import sys

        devices = self.DEVICES if devices is None else devices
        rows = [
            {"name": name, "max_input_channels": channels, "hostapi": hostapi}
            for name, channels, hostapi in devices
        ]
        fake = MagicMock()
        fake.query_devices.side_effect = lambda index=None: rows if index is None else rows[index]
        fake.query_hostapis.return_value = [{"name": name} for name in self.HOST_APIS]
        return patch.dict(sys.modules, {"sounddevice": fake})

    def test_each_microphone_is_listed_once(self):
        from pywhispr.audio import input_devices

        with self._sounddevice():
            assert input_devices() == [
                (3, "Microphone (Logitech PRO X Wireless Gaming Headset)"),
                (4, "Microphone (Logitech BRIO)"),
            ]

    def test_the_host_apis_own_default_pseudo_device_is_not_offered(self):
        from pywhispr.audio import input_devices

        with self._sounddevice():
            assert not [name for _index, name in input_devices() if "Primary Sound" in name]

    def test_every_host_api_is_still_available_for_resolving(self):
        from pywhispr.audio import all_input_devices

        with self._sounddevice():
            assert len(all_input_devices()) == len(self.DEVICES)

    def test_without_the_preferred_host_api_everything_is_listed(self):
        """A duplicate-ridden list still lets someone pick a microphone; an empty
        one does not."""
        from pywhispr.audio import input_devices

        mme_only = [row for row in self.DEVICES if row[2] == 0]
        with self._sounddevice(mme_only):
            assert len(input_devices()) == len(mme_only)


class TestAStoredNameIsNotStranded:
    """The choice is persisted by name and re-resolved at every recording, so a
    name saved while every host API was listed has to keep resolving."""

    def _sounddevice(self):
        return TestOneEntryPerDevice()._sounddevice()

    def test_a_name_the_shown_list_has_resolves_to_the_shown_device(self):
        from pywhispr.audio import find_device

        with self._sounddevice():
            assert find_device("Microphone (Logitech BRIO)") == 4

    def test_an_mme_truncated_name_resolves_to_the_full_device(self):
        from pywhispr.audio import find_device

        with self._sounddevice():
            index = find_device("Microphone (Logitech PRO X Wire")
            assert index == 3  # the DirectSound entry, not MME's index 1

    def test_a_truncated_name_shows_as_the_device_it_resolved_to(self):
        from pywhispr.audio import display_name

        with self._sounddevice():
            assert (
                display_name("Microphone (Logitech PRO X Wire")
                == "Microphone (Logitech PRO X Wireless Gaming Headset)"
            )

    def test_a_name_only_another_host_api_has_still_resolves(self):
        from pywhispr.audio import find_device

        with self._sounddevice():
            assert find_device("Microphone (PRO X Wireless Gaming Headset)") == 6

    def test_a_short_name_is_never_prefix_matched(self):
        """Only MME's truncation justifies a prefix match; a short name that is
        simply gone must stay gone rather than reach a longer device."""
        from pywhispr.audio import find_device

        with self._sounddevice():
            assert find_device("Microphone") is None

    def test_an_ambiguous_prefix_is_refused(self):
        """Two devices whose names differ only past MME's cut are exactly the case
        a name match cannot decide — the default with a warning beats a coin toss."""
        from pywhispr.audio import find_device

        truncated = "Microphone (Logitech BRIO 4K Pr"
        with TestOneEntryPerDevice()._sounddevice(
            [
                (truncated, 2, 0),
                ("Microphone (Logitech BRIO 4K Pro One)", 2, 1),
                ("Microphone (Logitech BRIO 4K Pro Two)", 2, 1),
            ]
        ):
            assert find_device(truncated) == 0  # its own exact MME entry, if still there

        with TestOneEntryPerDevice()._sounddevice(
            [
                ("Microphone (Logitech BRIO 4K Pro One)", 2, 1),
                ("Microphone (Logitech BRIO 4K Pro Two)", 2, 1),
            ]
        ):
            assert find_device(truncated) is None

    def test_a_device_that_is_really_gone_is_still_none(self):
        from pywhispr.audio import find_device

        with self._sounddevice():
            assert find_device("Yeti Stereo Microphone") is None
