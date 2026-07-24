"""STT backend selection."""

from __future__ import annotations

import platform
import sys

from pywhispr.config import Config
from pywhispr.stt.base import SAMPLE_RATE, STTBackend

__all__ = ["SAMPLE_RATE", "STTBackend", "create_backend"]


def create_backend(cfg: Config) -> STTBackend:
    """Pick the STT backend for this machine: MLX on Apple Silicon, ONNX elsewhere."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        from pywhispr.stt.mlx_backend import MlxBackend

        return MlxBackend(cfg.model_override)

    from pywhispr.stt.onnx_backend import OnnxBackend

    return OnnxBackend(cfg.model_override)
