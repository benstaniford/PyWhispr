"""What every entry point has to do before anything heavy is imported.

There are two ways in and they used to disagree. ``cli.main`` handles the
subcommands, but the packaged executable with no arguments goes straight to
``app.run_app`` (see :mod:`pywhispr.frozen`) — so preparation that lived in
``cli.main`` alone reached the download and GPU-check subprocesses and skipped the
tray app itself. The visible result was a model cache configured onto another
drive that the app kept filling on C:, and a DirectML install that verified
successfully and was then never activated.

So it lives here, both callers use it, and it is idempotent — being called twice
on the ``run`` path is the normal case.
"""

from __future__ import annotations

import logging

from pywhispr.config import Config, load_config

log = logging.getLogger(__name__)

_prepared = False

# Their own paths install or delete the thing this would activate.
_NO_DIRECTML = ("enable-directml", "disable-directml")


def prepare(command: str | None = None, cfg: Config | None = None) -> Config:
    """Apply the storage overrides and the DirectML choice. Returns the config.

    Order matters: huggingface_hub reads its cache path at import time and
    onnxruntime cannot be swapped once imported, so both have to happen before the
    backend is created.
    """
    global _prepared

    config = cfg if cfg is not None else load_config()
    if _prepared:
        return config

    from pywhispr.storage import apply_overrides

    apply_overrides(config)

    if command not in _NO_DIRECTML:
        from pywhispr.directml import activate_if_enabled

        activate_if_enabled(config)

    _prepared = True
    return config


def reset_for_tests() -> None:
    global _prepared
    _prepared = False


__all__ = ["prepare", "reset_for_tests"]
