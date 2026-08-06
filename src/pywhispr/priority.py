"""A bit more CPU for the work that is latency-critical (Windows only).

PyWhispr is a tray app, so it is never the Windows foreground process and never
gets the foreground scheduling boost: its STT worker competes at normal priority
with whatever the user is compiling. Transducer decoding is a Python loop
driving hundreds of tiny ``decoder_joint.run()`` calls on one thread, so it
degrades near-linearly with CPU load; the same contention keeps the global
keyboard hook's Python callback from returning inside Windows'
``LowLevelHooksTimeout``, which is the brief desktop typing hitch.

ABOVE_NORMAL, not HIGH: HIGH pre-empts the shell and the audio service, and a
dictation is not worth making the rest of the machine stutter to fix a stutter.
Nothing here may raise — a scheduling nicety must not lose a transcript.
"""

from __future__ import annotations

import contextlib
import logging
import sys

log = logging.getLogger(__name__)

ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


@contextlib.contextmanager
def boosted():
    """Run the block at ABOVE_NORMAL, putting the old priority class back after.

    The priority class is per *process*, so this only works because everything
    that uses it runs on the one STT worker thread — nested use would restore
    early. Whatever was set before is restored rather than NORMAL, so a user who
    started the app with `start /high` keeps their choice.
    """
    if sys.platform != "win32":
        yield  # POSIX would need `nice`, and the symptom is a Windows one
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    previous = None
    try:
        previous = kernel32.GetPriorityClass(handle) or None
        if previous and not kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS):
            previous = None
    except Exception:
        previous = None
    if previous is None:
        log.debug("Could not raise the process priority; running at whatever we have")
    try:
        yield
    finally:
        if previous is not None:
            with contextlib.suppress(Exception):
                kernel32.SetPriorityClass(handle, previous)
