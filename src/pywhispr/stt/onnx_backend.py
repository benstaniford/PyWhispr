"""Parakeet via ONNX Runtime (NVIDIA CUDA, with CPU fallback)."""

from __future__ import annotations

import logging
import time

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE, STTBackend

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

CPU_ONLY = ["CPUExecutionProvider"]
PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class OnnxBackend(STTBackend):
    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or DEFAULT_MODEL
        self._model = None
        self._providers: list[str] = []

    @property
    def name(self) -> str:
        return f"onnx-asr ({self._model_id})"

    def load(self) -> None:
        import onnx_asr
        import onnxruntime

        advertised = onnxruntime.get_available_providers()
        log.info(
            "onnxruntime %s on %s; providers advertised: %s",
            getattr(onnxruntime, "__version__", "?"),
            onnxruntime.get_device(),
            ", ".join(advertised),
        )

        providers = [p for p in PREFERRED_PROVIDERS if p in advertised] or CPU_ONLY
        if "CUDAExecutionProvider" not in providers:
            log.warning(
                "CUDAExecutionProvider not available (found: %s) — transcription will run "
                "on CPU. For an RTX GPU, install onnxruntime-gpu>=1.22 with a CUDA 12.8+ "
                "runtime and driver >=570.",
                ", ".join(advertised),
            )

        log.info(
            "Loading %s with providers %s (first run downloads ~600 MB)",
            self.name,
            providers,
        )
        started = time.monotonic()
        try:
            self._load_with(onnx_asr, providers)
        except Exception as exc:
            # onnxruntime-gpu advertises CUDAExecutionProvider even when the
            # machine has no CUDA runtime; the failure only surfaces here, when
            # a session is created. Falling back keeps a GPU-less Windows box
            # working instead of leaving the app stuck on "Loading model…".
            if providers == CPU_ONLY:
                raise
            log.warning(
                "Loading with %s failed (%s: %s) — retrying on CPU only",
                providers,
                type(exc).__name__,
                exc,
            )
            log.debug("Provider failure detail", exc_info=True)
            self._load_with(onnx_asr, CPU_ONLY)

        log.info(
            "Loaded %s in %.1fs using %s", self.name, time.monotonic() - started, self._providers
        )

    def _load_with(self, onnx_asr, providers: list[str]) -> None:
        self._model = onnx_asr.load_model(self._model_id, providers=providers)
        self._providers = providers

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")
        started = time.monotonic()
        text = self._model.recognize(audio.astype(np.float32), sample_rate=sample_rate)
        log.debug(
            "Transcribed %.1fs of audio in %.2fs",
            len(audio) / sample_rate,
            time.monotonic() - started,
        )
        return text.strip()
