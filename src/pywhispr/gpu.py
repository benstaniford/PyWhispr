"""What the two GPU paths have in common: is one possible, is one on, turn it off.

CUDA (``cuda.py``) and DirectML (``directml.py``) are alternatives, and everything
outside them wants to ask about "GPU acceleration" rather than about either one.
This is that question, plus the switch behind the tray menu's Enable/Disable entry.

Deliberately Qt-free and cheap to import: ``cuda`` and ``directml`` are imported
inside the functions, both because they pull in ``storage``/``config`` and because
``directml`` must not be touched before :func:`pywhispr.startup.prepare` has decided
which onnxruntime this process gets.

The switch is ``Config.use_gpu``, which leaves the download in place — turning it
back on costs nothing. ``pywhispr disable-gpu`` is the one that reclaims the disk.
"""

from __future__ import annotations

import logging
import sys

from pywhispr.config import Config, save_config

log = logging.getLogger(__name__)


def supported() -> bool:
    """Could any GPU path run on this platform at all?

    Platform, not backend: an Intel Mac runs the ONNX backend too, but its
    onnxruntime is the CPU build (see pyproject.toml) and DirectML has no macOS
    wheel — so the answer is the same list ``cuda.can_offer()`` and
    ``directml.can_offer()`` gate on themselves, deliberately rather than by
    coincidence. Apple Silicon uses MLX, which is on the Metal GPU regardless.
    """
    return sys.platform in ("win32", "linux")


def installed() -> bool:
    """Are either path's libraries on disk? Says nothing about whether they work."""
    from pywhispr import cuda, directml

    return cuda.is_installed() or directml.is_installed()


def active(cfg: Config) -> bool:
    """Is GPU acceleration both installed and switched on?

    What the tray menu's label asks. Not "is a GPU provider loaded": that is False
    between enabling and the restart it needs, which would offer to enable it twice.
    """
    return bool(cfg.use_gpu) and installed()


def turn_on(cfg: Config) -> None:
    """Switch acceleration back on for the next start. No download involved."""
    cfg.use_gpu = True
    save_config(cfg)
    log.info("GPU acceleration switched back on")


def turn_off(cfg: Config) -> None:
    """Switch acceleration off, leaving the libraries where they are.

    Saves the config, so **tests must patch ``pywhispr.gpu.save_config``** or they
    write the developer's own config file.

    Two things go with the flag:

    * ``offer_gpu_setup`` — otherwise the next start sees ``cuda.can_offer()`` go
      True again and offers to set up what was just switched off. The tray entry is
      the way back in, which is the contract ``config.py`` already documents.
    * ``model_quantization`` when it is ``""`` — the empty string is only ever
      written by the first-run CUDA path (``app._offer_gpu_before_downloading``), so
      anything else is the user's own. Back at None it re-decides in both
      directions: int8 for the CPU now, full precision again if acceleration
      returns. ``app._on_gpu_setup_finished`` makes the same fixup when setup fails.
    """
    cfg.use_gpu = False
    cfg.offer_gpu_setup = False
    if cfg.model_quantization == "":
        cfg.model_quantization = None
    save_config(cfg)
    log.info("GPU acceleration switched off; its libraries are still on disk")


__all__ = ["active", "installed", "supported", "turn_off", "turn_on"]
