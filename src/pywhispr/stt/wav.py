"""Minimal wav reading for the CLI and tests (stdlib only, 16-bit PCM)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE


def read_wav_mono_16k(path: str | Path) -> np.ndarray:
    """Read a wav file as mono float32 at 16 kHz, downmixing/resampling if needed."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav is supported")
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(
            np.linspace(0.0, len(audio) - 1, n_out), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio
