"""Microphone capture via sounddevice/PortAudio."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE

log = logging.getLogger(__name__)

BLOCK_SIZE = 1600  # 100 ms at 16 kHz → level updates 10x/sec


def rms_level(block: np.ndarray) -> float:
    """Perceptual-ish level in [0, 1] from a float32 audio block.

    Normal speech RMS sits around 0.02–0.2, so scale logarithmically over
    roughly -50 dBFS..0 dBFS to get a useful meter range.
    """
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    if rms <= 0.0:
        return 0.0
    db = 20.0 * np.log10(rms)
    return float(np.clip((db + 50.0) / 50.0, 0.0, 1.0))


class AudioRecorder:
    """Records mono float32 audio at 16 kHz until stopped.

    ``on_level`` (if given) is called from the PortAudio callback thread with a
    0..1 level roughly 10 times per second — keep it cheap and thread-safe
    (emitting a Qt signal is fine; Qt queues it to the GUI thread).
    """

    def __init__(
        self,
        device: int | None = None,
        on_level: Callable[[float], None] | None = None,
    ):
        self._device = device
        self._on_level = on_level
        self._stream = None
        self._blocks: list[np.ndarray] = []

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            raise RuntimeError("Already recording")
        self._blocks = []

        def callback(indata, frames, time_info, status):
            if status:
                log.warning("Audio input status: %s", status)
            block = indata[:, 0].copy()
            self._blocks.append(block)
            if self._on_level is not None:
                self._on_level(rms_level(block))

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
        )
        self._stream.start()
        log.debug("Recording started (device=%s)", self._device)

    def stop(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Not recording")
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        audio = (
            np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype=np.float32)
        )
        self._blocks = []
        log.debug("Recording stopped: %.1fs captured", len(audio) / SAMPLE_RATE)
        return audio
