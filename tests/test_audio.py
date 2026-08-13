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
        with patch("pywhispr.audio.input_devices", return_value=[(0, "Array"), (3, "Yeti")]):
            from pywhispr.audio import find_device

            assert find_device("Yeti") == 3

    def test_an_absent_device_is_none_rather_than_a_guess(self):
        with patch("pywhispr.audio.input_devices", return_value=[(0, "Array")]):
            from pywhispr.audio import find_device

            assert find_device("Yeti") is None
