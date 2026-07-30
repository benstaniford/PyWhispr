"""Watching the model download, which otherwise happens invisibly.

The first run fetches ~600 MB through ``huggingface_hub``, and until it finishes
the app says only "Loading model…" — indistinguishable from being stuck. There is
no progress callback to hook (onnx_asr and parakeet-mlx both download internally),
so progress is measured from the outside: how much the Hugging Face cache has
grown since the load started.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# The full-precision Parakeet weights, plus the quantised set the CPU path then
# fetches. Only used to scale a progress bar, so being approximate is fine.
APPROXIMATE_MODEL_MB = 600


def cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def cache_bytes() -> int:
    """Bytes in the Hugging Face cache, ignoring anything unreadable."""
    total = 0
    try:
        for path in cache_dir().rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def model_cached(minimum_mb: int = APPROXIMATE_MODEL_MB // 2) -> bool:
    """Is a model already downloaded, so no progress needs showing?

    Size rather than an exact file list: the repo layout is onnx_asr's business
    and differs between the ONNX and MLX backends.
    """
    return cache_bytes() >= minimum_mb * 1024 * 1024
