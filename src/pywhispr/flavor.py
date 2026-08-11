"""Which product this process is: full PyWhispr or the REST-client Lite build.

PyWhisprLite is the same codebase pointed at a remote transcription server
(``api.py``'s ``POST /v1/transcribe``) instead of a local model, for machines
that cannot run one — chiefly Intel macOS, where no STT backend is even
installed (see ``pyproject.toml``'s platform markers). It is a *flavor*, not a
fork: one environment variable decides everything downstream.

``PYWHISPR_LITE=1`` is set for real by the packaged Lite bundle (a PyInstaller
runtime hook — see ``packaging/rthook_lite.py``) and by hand for dev/test on any
machine. Absent, this module is inert and the app behaves exactly as before.

Kept deliberately tiny and free of Qt or any ``pywhispr`` import so it can be
read from anywhere — including ``config.py``, which sets the config directory
from :data:`PRODUCT_NAME` and so must not import the heavier modules.
"""

from __future__ import annotations

import os

# Read once at import: the bundle sets it before any pywhispr import, and nothing
# should flip products mid-run. Tests monkeypatch these attributes directly.
IS_LITE = os.environ.get("PYWHISPR_LITE") == "1"

# The user-facing product name — config directory, window titles, tray text. Lite
# uses its own so a full install and a Lite install coexist without fighting over
# one config.toml.
PRODUCT_NAME = "PyWhisprLite" if IS_LITE else "PyWhispr"

__all__ = ["IS_LITE", "PRODUCT_NAME"]
