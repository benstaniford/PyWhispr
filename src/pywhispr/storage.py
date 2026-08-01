"""Where the two big downloads live: the model weights and the CUDA libraries.

Together they are around 6 GB, and both default into the user profile on the
system drive. On a machine with a small C: that is the difference between usable
and not, so each has an override — a config key, or an environment variable that
wins over it (the variable is what a packaged run can be pointed at without
touching the config file).

The model cache is redirected through ``HF_HUB_CACHE`` rather than passed as an
argument, because nothing here does the downloading: ``onnx_asr`` and
``parakeet-mlx`` both call ``huggingface_hub`` internally. That constant is read
at import time, so ``apply_overrides`` has to run before anything imports
``huggingface_hub`` — ``cli.main`` calls it before its own lazy imports.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from platformdirs import user_data_dir

from pywhispr.config import APP_NAME, Config

log = logging.getLogger(__name__)

MODEL_ENV = "PYWHISPR_MODEL_DIR"
CUDA_ENV = "PYWHISPR_CUDA_DIR"

# The model cache and the CUDA libraries, plus room to extract the wheels.
REQUIRED_MB = 7000


def default_model_dir() -> Path:
    """Where huggingface_hub would put the weights if we said nothing."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def default_cuda_dir() -> Path:
    return Path(user_data_dir(APP_NAME)) / "cuda"


def model_dir(cfg: Config | None = None) -> Path:
    override = os.environ.get(MODEL_ENV) or (cfg.model_cache_dir if cfg else None)
    return Path(override).expanduser() if override else default_model_dir()


def cuda_dir(cfg: Config | None = None) -> Path:
    override = os.environ.get(CUDA_ENV) or (cfg.cuda_dir if cfg else None)
    return Path(override).expanduser() if override else default_cuda_dir()


def apply_overrides(cfg: Config) -> None:
    """Point huggingface_hub at the configured model directory, and CUDA at its own.

    Both are exported so the helper processes inherit them: the download and the
    GPU check run as separate interpreters, and a path that only existed in this
    one would send them back to the system drive.
    """
    if cfg.model_cache_dir and MODEL_ENV not in os.environ:
        os.environ[MODEL_ENV] = cfg.model_cache_dir
    if cfg.cuda_dir and CUDA_ENV not in os.environ:
        os.environ[CUDA_ENV] = cfg.cuda_dir

    model = os.environ.get(MODEL_ENV)
    if model:
        # HF_HUB_CACHE is the hub directory itself; HF_HOME is its parent, which is
        # where the token and other metadata would otherwise land back on C:.
        target = Path(model).expanduser()
        os.environ["HF_HUB_CACHE"] = str(target)
        os.environ.setdefault("HF_HOME", str(target.parent))
        log.info("Model cache directory: %s", target)
    if os.environ.get(CUDA_ENV):
        log.info("CUDA directory: %s", cuda_dir())


def free_mb(path: Path) -> int | None:
    """Free space on the drive holding ``path``, walking up to a parent that exists.

    A directory the user has only just typed does not exist yet, and neither does
    its parent on an empty drive, so the question is really about the volume.
    """
    for candidate in [path, *path.parents]:
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free // (1024 * 1024)
            except OSError:
                return None
    return None


def set_base_dir(cfg: Config, base: Path) -> None:
    """Put both downloads under one directory the user picked.

    Separate keys rather than one base key, so the two can still be moved
    independently by hand afterwards.
    """
    base = Path(base).expanduser()
    cfg.model_cache_dir = str(base / "models")
    cfg.cuda_dir = str(base / "cuda")
