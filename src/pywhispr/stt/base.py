"""Speech-to-text backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 16000


class STTBackend(ABC):
    """A local speech-to-text engine.

    Implementations accept mono float32 audio at 16 kHz as an in-memory numpy
    array, so the whole layer is testable without a microphone or UI.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend/model description for logs."""

    @property
    def download_mb(self) -> int:
        """Roughly what a first run will fetch, for the progress window."""
        return 2450

    @abstractmethod
    def load(self) -> None:
        """Download (if needed) and load the model. May take minutes on first run."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe mono float32 audio and return the text (stripped)."""
