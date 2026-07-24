"""Parakeet via ONNX Runtime (NVIDIA CUDA, with CPU fallback)."""

from __future__ import annotations

import logging

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE, STTBackend

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"


class OnnxBackend(STTBackend):
    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or DEFAULT_MODEL
        self._model = None

    @property
    def name(self) -> str:
        return f"onnx-asr ({self._model_id})"

    def load(self) -> None:
        import onnx_asr
        import onnxruntime

        available = onnxruntime.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if "CUDAExecutionProvider" not in providers:
            log.warning(
                "CUDAExecutionProvider not available (found: %s) — transcription will run "
                "on CPU. For an RTX GPU, install onnxruntime-gpu>=1.22 with a CUDA 12.8+ "
                "runtime and driver >=570.",
                ", ".join(available),
            )
        self._model = onnx_asr.load_model(self._model_id, providers=providers)
        log.info("Loaded %s with providers %s", self.name, providers)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")
        text = self._model.recognize(audio.astype(np.float32), sample_rate=sample_rate)
        return text.strip()
