"""Minimal wav/PCM decoding for the CLI, the network API and tests (stdlib only).

Everything here converts to the one format the STT backends accept: mono
float32 at 16 kHz. There is no codec support — wav and headerless PCM only.
Browser ``MediaRecorder`` output (WebM/Opus) cannot be decoded without ffmpeg;
web clients should send raw float32 PCM from an ``AudioContext`` instead.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE

# Raw PCM encodings accepted by the API, mapped to their numpy dtype.
PCM_FORMATS = {"f32le": np.dtype("<f4"), "s16le": np.dtype("<i2")}


def _to_mono_16k(audio: np.ndarray, channels: int, rate: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz, both no-ops if already correct."""
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(audio) > 1:
        n_out = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0.0, len(audio) - 1, n_out), np.arange(len(audio)), audio)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _read_wav(wf: wave.Wave_read) -> np.ndarray:
    """Decode an open wav reader to mono float32 at 16 kHz."""
    width = wf.getsampwidth()
    frames = wf.readframes(wf.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        # 32-bit wav is either IEEE float (what browsers and DAWs emit) or
        # int32. The wave module hides the format tag, so infer from range.
        raw = np.frombuffer(frames, dtype="<f4")
        audio = (
            raw.astype(np.float32)
            if np.all(np.abs(raw) <= 1.0)
            else np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        )
    else:
        raise ValueError(f"only 16-bit PCM or 32-bit wav is supported, got {width * 8}-bit")

    return _to_mono_16k(audio, wf.getnchannels(), wf.getframerate())


def read_wav_mono_16k(path: str | Path) -> np.ndarray:
    """Read a wav file as mono float32 at 16 kHz, downmixing/resampling if needed."""
    try:
        with wave.open(str(path), "rb") as wf:
            return _read_wav(wf)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def read_wav_bytes_mono_16k(data: bytes) -> np.ndarray:
    """Read wav bytes as mono float32 at 16 kHz."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            return _read_wav(wf)
    except wave.Error as exc:
        raise ValueError(f"not a readable wav file: {exc}") from exc


def pcm_to_mono_16k(
    data: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1, fmt: str = "f32le"
) -> np.ndarray:
    """Decode headerless interleaved PCM as mono float32 at 16 kHz."""
    dtype = PCM_FORMATS.get(fmt)
    if dtype is None:
        raise ValueError(f"unknown pcm format {fmt!r}, expected one of {sorted(PCM_FORMATS)}")
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if sample_rate < 1:
        raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")

    frame_bytes = dtype.itemsize * channels
    if len(data) % frame_bytes:
        raise ValueError(f"pcm length {len(data)} is not a whole number of {frame_bytes}B frames")

    audio = np.frombuffer(data, dtype=dtype).astype(np.float32)
    if dtype.kind == "i":
        audio /= 32768.0
    return _to_mono_16k(audio, channels, sample_rate)
