"""Frozen-app entry point (PyInstaller on macOS, cx_Freeze on Windows)."""

import faulthandler
import logging
import sys

# dump a stack on a native crash; the Windows gui-base build has no stderr,
# so fall back to a log file there
if sys.stderr is not None:
    faulthandler.enable()
else:
    import tempfile
    from pathlib import Path

    _crash_log = open(Path(tempfile.gettempdir()) / "pywhispr-crash.log", "a")
    faulthandler.enable(file=_crash_log)

from pywhispr.app import run_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

sys.exit(run_app())
