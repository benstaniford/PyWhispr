"""Duck other applications' audio while a recording is running.

Windows and macOS reach the same end by different means, because only one of
them has a per-application volume:

- **Windows** has one. Core Audio exposes a per-application session volume
  (``ISimpleAudioVolume``, via pycaw), so every other app is turned down
  individually and our own cues are left alone.
- **macOS** has none. The CoreAudio process object exposes a pid, a bundle id,
  its devices and three read-only "is running" flags, and ``AVAudioSession``'s
  ``DuckOthers`` option is ``API_UNAVAILABLE(macos)``. So the *default output
  device* is dipped instead — see `coreaudio.py`.

Being system-wide has three consequences, all of them macOS-only:

- **Our own cues are inside the dip.** Windows keeps its start and stop sounds
  audible by skipping its own pid; there is no such escape hatch here, and at
  the default ``duck_volume = 0.0`` the start cue would simply be inaudible.
  Hence ``cue_lead_ms``, which `app.py` waits out before dipping.
- **A hard kill while ducked leaves the whole machine quiet**, not just the
  other apps. Every path out of RECORDING restores (including quit), but
  SIGKILL, a native crash and SIGTERM unwind nothing. The pre-duck level is
  logged at INFO for exactly that reason, and the volume keys put it back.
- **Some devices cannot be ducked at all.** An aggregate or Multi-Output
  device has no volume of its own, and neither do HDMI and many external DACs.
  That is an ordinary answer, not an error: log it and record nothing.

Two caveats shared by both platforms:

- Windows *remembers* per-app mixer levels, so a crash while ducked leaves the
  other apps quiet until the user puts them back by hand. That is why the app
  restores on every path out of RECORDING (including quit) and why duck() and
  restore() swallow everything — a broken ducker must degrade to "no ducking",
  never take the dictation cycle down with it.
- Only what exists when the recording starts is ducked: on Windows a session
  that appears mid-recording plays at full volume, and on macOS a switch to
  another output device mid-recording leaves the new one alone (restore always
  targets the device that was actually turned down).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Protocol

from pywhispr import coreaudio
from pywhispr.config import Config

log = logging.getLogger(__name__)


# start.wav is 180ms, plus however long QSoundEffect takes to get its sink going.
CUE_LEAD_MS = 300


class Ducker(Protocol):
    # How long to wait after the start cue before dipping. Zero everywhere the
    # dip cannot reach our own sounds; see the module docstring.
    cue_lead_ms: int

    def duck(self) -> None: ...

    def restore(self) -> None: ...


class NoOpDucker:
    """Stands in when ducking is off or unsupported, so callers never branch."""

    cue_lead_ms = 0

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

    cue_lead_ms = 0  # our own session is skipped, so the cues need no head start

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


class OutputVolumeDucker:
    """Dip the default output device's volume, then put it back.

    macOS has no per-application output volume, so this is the whole machine —
    our own cues included, which is why cue_lead_ms is not zero.
    """

    cue_lead_ms = CUE_LEAD_MS

    def __init__(self, level: float, get_controls=None):
        self._level = min(max(level, 0.0), 1.0)
        self._get_controls = get_controls or coreaudio.output_volume_controls
        self._saved: list[tuple[object, float]] = []  # (control, original level)

    def duck(self) -> None:
        if self._saved:
            return  # already ducked; saving again would remember ducked levels as originals
        try:
            controls = self._get_controls()
        except Exception:
            # No CoreAudio, no default output device, a device that died between
            # being named and being asked: dictation carries on without the quiet.
            log.exception("Could not find the output volume; not ducking")
            return
        if not controls:
            # An aggregate or Multi-Output device, HDMI, some external DACs: the
            # volume belongs to the hardware and there is nothing to turn down.
            log.info("The output device has no settable volume; not ducking")
            return
        for control in controls:
            try:
                original = control.get()
                control.set(original * self._level)
            except Exception:
                log.debug("Could not duck one output volume control", exc_info=True)
                continue
            self._saved.append((control, original))
        if self._saved:
            # INFO, unlike everything else here: a hard kill leaves the machine
            # at the ducked level, and this line is the record of what to put back.
            log.info(
                "Ducked the output volume from %d%% to %d%% (%d control(s))",
                self._saved[0][1] * 100,
                self._saved[0][1] * self._level * 100,
                len(self._saved),
            )

    def restore(self) -> None:
        # Always the saved controls, never a fresh look at the default output:
        # the device we turned down is the device that must be put back, which is
        # what makes switching outputs mid-recording harmless.
        saved, self._saved = self._saved, []
        for control, original in saved:
            try:
                control.set(original)
            except Exception:
                log.debug("Could not restore one output volume control", exc_info=True)
        if saved:
            log.debug("Restored %d output volume control(s)", len(saved))


def supported(platform: str | None = None) -> bool:
    """Can this platform duck at all? Windows per-app, macOS system-wide.

    Not a default argument (``platform: str = sys.platform``), which would bind
    at import time and quietly defeat the monkeypatched-sys.platform idiom the
    tests are built on.
    """
    return (platform or sys.platform) in ("win32", "darwin")


def system_wide(platform: str | None = None) -> bool:
    """Does ducking here move the whole output device rather than per-app levels?"""
    return (platform or sys.platform) == "darwin"


def create_ducker(cfg: Config) -> Ducker:
    """The right ducker for this config and platform; NoOp unless both agree."""
    if not cfg.duck_other_audio:
        return NoOpDucker()
    if not supported(sys.platform):
        log.info("duck_other_audio is not supported on this platform; ignoring it")
        return NoOpDucker()
    if sys.platform == "darwin":
        return OutputVolumeDucker(cfg.duck_volume)
    return SessionDucker(cfg.duck_volume)
