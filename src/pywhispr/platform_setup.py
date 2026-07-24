"""Platform-specific setup and permission checks."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

MACOS_PERMISSIONS_HELP = """\
Accessibility permission is not granted, so PyWhispr cannot paste text into
other apps automatically. Dictation still works: the transcript is copied to
the clipboard — press Cmd+V yourself to paste it.

For automatic pasting, grant Accessibility to the app you launch PyWhispr
from (e.g. Terminal) in System Settings → Privacy & Security → Accessibility,
then fully relaunch that app. No other permission is needed; the global
hotkey works without Input Monitoring.
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


def check_macos_input_monitoring() -> bool:
    """True if this process may observe raw key events (macOS Input Monitoring).

    Only needed for double-tap hotkeys; chord hotkeys use RegisterEventHotKey,
    which is permission-free.
    """
    if sys.platform != "darwin":
        return True
    import ctypes.util

    path = ctypes.util.find_library("CoreGraphics")
    if path is None:
        return True  # can't check; proceed optimistically
    core_graphics = ctypes.cdll.LoadLibrary(path)
    try:
        return bool(core_graphics.CGPreflightListenEventAccess())
    except AttributeError:
        return True


def request_macos_input_monitoring() -> None:
    """Prompt for Input Monitoring and add this app to the Settings list."""
    if sys.platform != "darwin":
        return
    import ctypes.util

    path = ctypes.util.find_library("CoreGraphics")
    if path is not None:
        ctypes.cdll.LoadLibrary(path).CGRequestListenEventAccess()


MACOS_INPUT_MONITORING_HELP = """\
Double-tap hotkeys listen for modifier key taps, which macOS gates behind the
Input Monitoring permission. Grant it to PyWhispr (or the terminal you launch
it from) in System Settings → Privacy & Security → Input Monitoring, then
relaunch PyWhispr. Until then the double-tap hotkey will not fire.
Chord hotkeys (e.g. Cmd+Shift+Space) work without any permission.
"""


def warn_if_missing_permissions() -> bool:
    """Log guidance if auto-paste won't work. Returns True if fully functional."""
    if sys.platform != "darwin":
        return True
    if not check_macos_accessibility():
        log.warning(MACOS_PERMISSIONS_HELP)
        return False
    return True
