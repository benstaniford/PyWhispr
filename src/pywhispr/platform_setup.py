"""Platform-specific setup and permission checks."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

MACOS_PERMISSIONS_HELP = """\
PyWhispr needs two macOS permissions, granted to the app you launch it from
(e.g. Terminal or iTerm2) in System Settings → Privacy & Security:

  1. Input Monitoring  — to hear the global hotkey
  2. Accessibility     — to paste text into other apps

After granting them, fully quit and relaunch your terminal app.
(Microphone access is prompted automatically the first time you record.)
"""


def check_macos_accessibility() -> bool:
    """True if this process may synthesize keystrokes (macOS Accessibility)."""
    if sys.platform != "darwin":
        return True
    import ctypes.util

    path = ctypes.util.find_library("ApplicationServices")
    if path is None:
        return True  # can't check; proceed optimistically
    app_services = ctypes.cdll.LoadLibrary(path)
    return bool(app_services.AXIsProcessTrusted())


def warn_if_missing_permissions() -> bool:
    """Log guidance if required permissions look missing. Returns True if OK."""
    if sys.platform != "darwin":
        return True
    if not check_macos_accessibility():
        log.warning(MACOS_PERMISSIONS_HELP)
        return False
    return True
