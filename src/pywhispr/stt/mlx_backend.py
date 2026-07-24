"""Parakeet on Apple Silicon via MLX."""

from __future__ import annotations

import logging

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE, STTBackend

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class MlxBackend(STTBackend):
    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or DEFAULT_MODEL
        self._model = None

    @property
    def name(self) -> str:
        return f"parakeet-mlx ({self._model_id})"

    def load(self) -> None:
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self._model_id)
        if self._model.preprocessor_config.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"Model expects {self._model.preprocessor_config.sample_rate} Hz audio, "
                f"but PyWhispr records at {SAMPLE_RATE} Hz"
            )
        log.info("Loaded %s", self.name)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        # Feed audio directly through the mel pipeline; model.transcribe() only
        # accepts file paths and shells out to ffmpeg.
        mel = get_logmel(mx.array(audio.astype(np.float32)), self._model.preprocessor_config)
        result = self._model.generate(mel)[0]
        return result.text.strip()
