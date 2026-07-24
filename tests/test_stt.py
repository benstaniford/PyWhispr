from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pywhispr.config import Config
from pywhispr.stt import create_backend
from pywhispr.stt.wav import read_wav_mono_16k

FIXTURE = Path(__file__).parent / "fixtures" / "hello_world.wav"


class TestBackendSelection:
    def test_apple_silicon_gets_mlx(self):
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            backend = create_backend(Config())
        assert "parakeet-mlx" in backend.name

    def test_windows_gets_onnx(self):
        with patch("sys.platform", "win32"):
            backend = create_backend(Config())
        assert "onnx-asr" in backend.name

    def test_model_override_is_passed_through(self):
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            backend = create_backend(Config(model_override="someone/custom-model"))
        assert "someone/custom-model" in backend.name


class TestWavReading:
    def test_fixture_reads_as_mono_float32_16k(self):
        audio = read_wav_mono_16k(FIXTURE)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) > 16000  # fixture is longer than 1 second
        assert np.abs(audio).max() <= 1.0

    def test_resamples_non_16k(self, tmp_path):
        import wave

        path = tmp_path / "44k.wav"
        rate = 44100
        t = np.linspace(0, 1.0, rate, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(samples.tobytes())

        audio = read_wav_mono_16k(path)
        assert abs(len(audio) - 16000) <= 1


@pytest.mark.model
def test_real_model_transcribes_fixture():
    """Downloads and runs the actual Parakeet model. Run with: pytest -m model"""
    backend = create_backend(Config())
    backend.load()
    audio = read_wav_mono_16k(FIXTURE)
    text = backend.transcribe(audio).lower()
    assert "hello world" in text
    assert "transcription" in text
