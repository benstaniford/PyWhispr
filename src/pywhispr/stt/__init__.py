"""STT backend selection."""

from __future__ import annotations

import platform
import sys

from pywhispr.config import Config
from pywhispr.stt.base import SAMPLE_RATE, STTBackend

__all__ = ["SAMPLE_RATE", "STTBackend", "create_backend"]


def create_backend(cfg: Config) -> STTBackend:
    """Pick the STT backend: remote in the Lite build, else MLX/ONNX by platform."""
    from pywhispr import flavor

    if flavor.IS_LITE:
        # PyWhisprLite runs no model locally; it POSTs to a configured server.
        from pywhispr.stt.remote_backend import RemoteBackend

        return RemoteBackend(cfg.server_url)

    if sys.platform == "darwin" and platform.machine() == "arm64":
        from pywhispr.stt.mlx_backend import MlxBackend

        return MlxBackend(cfg.model_override)

    from pywhispr.stt.onnx_backend import OnnxBackend

    return OnnxBackend(
        cfg.model_override,
        quantization=cfg.model_quantization,
        threads=cfg.stt_threads,
        use_gpu=cfg.use_gpu,
    )
