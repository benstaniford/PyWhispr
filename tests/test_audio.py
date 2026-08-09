from unittest.mock import MagicMock

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
