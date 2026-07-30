"""Watching the model download, which otherwise happens invisibly.

The first run fetches gigabytes through ``huggingface_hub``, and until it finishes
the app says only "Loading model…" — indistinguishable from being stuck. There is
no progress callback to hook (onnx_asr and parakeet-mlx both download internally),
so progress is measured from the outside: how much the Hugging Face cache has
grown since the load started.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Parakeet v3's full-precision encoder alone is 2.3 GB; the quantised variant is
# ~0.7 GB. Backends report their own figure, so this is only the fallback.
APPROXIMATE_MODEL_MB = 2450


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


def start_model_download(quantization: str | None = None):
    """Fetch the weights in a separate process, returned so it can be killed.

    A step of its own, before anything is verified: the download is the long part
    and belongs on the progress bar, where the check that follows it is seconds.

    Output goes nowhere on purpose. A pipe nobody reads while waiting fills up —
    huggingface_hub writes progress bars to stderr — and then the child blocks on
    write forever, at whatever byte it had reached. The child logs to the same file
    we do, so nothing is lost.
    """
    import subprocess

    from pywhispr.cuda import self_command

    arguments = ["download"]
    if quantization is not None:
        arguments += ["--quantization", quantization]
    return subprocess.Popen(
        self_command(*arguments),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def model_cached(minimum_mb: int = 400) -> bool:
    """Is a model already downloaded, so no progress needs showing?

    Size rather than an exact file list: the repo layout is onnx_asr's business and
    differs between the ONNX and MLX backends. The threshold is well under the
    smallest variant, so a part-downloaded model still counts as "show progress".
    """
    return cache_bytes() >= minimum_mb * 1024 * 1024
