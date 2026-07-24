import io
import wave
from pathlib import Path

import numpy as np
import pytest

from pywhispr.stt.wav import pcm_to_mono_16k, read_wav_bytes_mono_16k, read_wav_mono_16k

FIXTURE = Path(__file__).parent / "fixtures" / "hello_world.wav"


def write_wav(path_or_buf, audio, rate=16000, channels=1, width=2):
    with wave.open(path_or_buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        if width == 2:
            wf.writeframes((np.asarray(audio) * 32767).astype("<i2").tobytes())
        else:
            wf.writeframes(np.asarray(audio, dtype="<f4").tobytes())


class TestReadWavBytes:
    def test_matches_the_path_reader(self):
        assert np.array_equal(
            read_wav_bytes_mono_16k(FIXTURE.read_bytes()), read_wav_mono_16k(FIXTURE)
        )

    def test_float32_wav(self):
        buf = io.BytesIO()
        write_wav(buf, np.full(1600, 0.5, dtype=np.float32), width=4)
        audio = read_wav_bytes_mono_16k(buf.getvalue())
        assert audio.dtype == np.float32
        assert audio[0] == pytest.approx(0.5)

    def test_stereo_is_downmixed(self):
        buf = io.BytesIO()
        interleaved = np.zeros(3200, dtype=np.float32)
        interleaved[::2] = 0.5  # left only
        write_wav(buf, interleaved, channels=2)
        audio = read_wav_bytes_mono_16k(buf.getvalue())
        assert audio.shape == (1600,)
        assert audio[0] == pytest.approx(0.25, abs=1e-3)

    def test_rejects_non_wav(self):
        with pytest.raises(ValueError):
            read_wav_bytes_mono_16k(b"not a wav at all")

    def test_rejects_8bit(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(16000)
            wf.writeframes(b"\x80" * 1600)
        with pytest.raises(ValueError, match="8-bit"):
            read_wav_bytes_mono_16k(buf.getvalue())


class TestPcm:
    def test_f32le_passthrough(self):
        audio = pcm_to_mono_16k(np.full(1600, 0.25, dtype="<f4").tobytes())
        assert audio.dtype == np.float32
        assert audio.shape == (1600,)
        assert audio[0] == pytest.approx(0.25)

    def test_s16le(self):
        audio = pcm_to_mono_16k(np.full(1600, 8192, dtype="<i2").tobytes(), fmt="s16le")
        assert audio[0] == pytest.approx(0.25)

    def test_resamples(self):
        audio = pcm_to_mono_16k(np.zeros(44100, dtype="<f4").tobytes(), sample_rate=44100)
        assert audio.shape[0] == pytest.approx(16000, rel=0.01)

    def test_downmixes(self):
        interleaved = np.zeros(3200, dtype="<f4")
        interleaved[::2] = 0.5
        audio = pcm_to_mono_16k(interleaved.tobytes(), channels=2)
        assert audio.shape == (1600,)
        assert audio[0] == pytest.approx(0.25)

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="unknown pcm format"):
            pcm_to_mono_16k(b"\x00" * 16, fmt="mp3")

    def test_rejects_partial_frame(self):
        with pytest.raises(ValueError, match="whole number"):
            pcm_to_mono_16k(b"\x00" * 6, channels=2)
