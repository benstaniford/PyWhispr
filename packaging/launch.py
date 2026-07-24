"""Frozen-app entry point (PyInstaller on macOS, cx_Freeze on Windows)."""

import faulthandler
import logging
import sys

# The Windows gui-base build has no console, so sys.stdout/sys.stderr are None.
# Anything that writes to them (tqdm's HuggingFace download bar on first run,
# stray print()s) then dies with "'NoneType' object has no attribute 'write'".
# Point both at a log file so those writes go somewhere real instead of crashing.
if sys.stderr is None or sys.stdout is None:
    import tempfile
    from pathlib import Path

    _log = open(Path(tempfile.gettempdir()) / "pywhispr-crash.log", "a", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log
    faulthandler.enable(file=_log)
else:
    faulthandler.enable()

from pywhispr.app import run_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.exit(run_app())
