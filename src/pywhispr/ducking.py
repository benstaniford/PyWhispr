"""Duck other applications' audio while a recording is running.

Windows only for now: Core Audio exposes a per-application session volume
(ISimpleAudioVolume, via pycaw), so every other app can be turned down and put
back without touching the master volume. macOS has no public per-app volume
API, so the setting is ignored there.

Two caveats worth knowing:

- Windows *remembers* per-app mixer levels, so a crash while ducked leaves the
  other apps quiet until the user puts them back by hand. That is why the app
  restores on every path out of RECORDING (including quit) and why duck() and
  restore() swallow everything — a broken ducker must degrade to "no ducking",
  never take the dictation cycle down with it.
- Only sessions on the default output device that exist when the recording
  starts are ducked; an app that starts playing mid-recording plays at full
  volume. Fine for an experiment.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Protocol

log = logging.getLogger(__name__)


class Ducker(Protocol):
    def duck(self) -> None: ...

    def restore(self) -> None: ...


class NoOpDucker:
    """Stands in when ducking is off or unsupported, so callers never branch."""

    def duck(self) -> None:
        pass

    def restore(self) -> None:
        pass


def _windows_sessions():
    from pycaw.pycaw import AudioUtilities

    return AudioUtilities.GetAllSessions()


class SessionDucker:
    """Turn every *other* app's session volume down, then put it back.

    Ducking is relative (each session keeps its place in the mix) and saved
    per-session, so restore() returns exactly the levels that were there.
    """

    def __init__(self, level: float, get_sessions=None, own_pid: int | None = None):
        self._level = min(max(level, 0.0), 1.0)
        self._get_sessions = get_sessions or _windows_sessions
        self._own_pid = os.getpid() if own_pid is None else own_pid
        self._saved: list[tuple[object, float]] = []  # (volume interface, original level)

    def duck(self) -> None:
        if self._saved:
            return  # already ducked; saving again would remember ducked levels as originals
        try:
            sessions = self._get_sessions()
        except Exception:
            # Missing pycaw, COM refusing to come up, no audio device: dictation
            # must carry on, just without the quiet.
            log.exception("Could not enumerate audio sessions; not ducking")
            return
        for session in sessions:
            try:
                if getattr(session, "ProcessId", None) == self._own_pid:
                    continue  # keep our own start/stop cues audible
                volume = session.SimpleAudioVolume
                original = volume.GetMasterVolume()
                volume.SetMasterVolume(original * self._level, None)
            except Exception:
                # A session can die mid-iteration (its app quit); the rest
                # still deserve ducking.
                log.debug("Could not duck one audio session", exc_info=True)
                continue
            self._saved.append((volume, original))
        log.debug("Ducked %d audio session(s) to %d%%", len(self._saved), self._level * 100)

    def restore(self) -> None:
        saved, self._saved = self._saved, []
        for volume, original in saved:
            try:
                volume.SetMasterVolume(original, None)
            except Exception:
                log.debug("Could not restore one audio session", exc_info=True)
        if saved:
            log.debug("Restored %d audio session(s)", len(saved))


def create_ducker(cfg) -> Ducker:
    """The right ducker for this config and platform; NoOp unless both agree."""
    if not cfg.duck_other_audio:
        return NoOpDucker()
    if sys.platform != "win32":
        log.info("duck_other_audio is only supported on Windows; ignoring it")
        return NoOpDucker()
    return SessionDucker(cfg.duck_volume)
