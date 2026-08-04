import sys

import pytest

from pywhispr.config import Config
from pywhispr.ducking import NoOpDucker, SessionDucker, create_ducker

OWN_PID = 4242


class FakeVolume:
    def __init__(self, level: float):
        self.level = level
        self.calls = 0

    def GetMasterVolume(self) -> float:
        return self.level

    def SetMasterVolume(self, value: float, _ctx) -> None:
        self.calls += 1
        self.level = value


class FakeSession:
    def __init__(self, pid: int, level: float):
        self.ProcessId = pid
        self.SimpleAudioVolume = FakeVolume(level)


def ducker(sessions, level=0.25):
    return SessionDucker(level, get_sessions=lambda: sessions, own_pid=OWN_PID)


class TestDuckAndRestore:
    def test_duck_scales_each_session_relative_to_its_own_level(self):
        loud, quiet = FakeSession(1, 0.8), FakeSession(2, 0.4)
        ducker([loud, quiet]).duck()
        assert loud.SimpleAudioVolume.level == pytest.approx(0.2)
        assert quiet.SimpleAudioVolume.level == pytest.approx(0.1)

    def test_restore_puts_the_original_levels_back(self):
        loud, quiet = FakeSession(1, 0.8), FakeSession(2, 0.4)
        d = ducker([loud, quiet])
        d.duck()
        d.restore()
        assert loud.SimpleAudioVolume.level == pytest.approx(0.8)
        assert quiet.SimpleAudioVolume.level == pytest.approx(0.4)

    def test_own_session_is_left_alone(self):
        ours, theirs = FakeSession(OWN_PID, 1.0), FakeSession(1, 1.0)
        ducker([ours, theirs]).duck()
        assert ours.SimpleAudioVolume.level == 1.0
        assert theirs.SimpleAudioVolume.level == pytest.approx(0.25)

    def test_system_sounds_session_is_ducked(self):
        # ProcessId 0 with no process is the system-sounds session — the
        # notification dings are exactly what should go quiet.
        system = FakeSession(0, 0.6)
        ducker([system]).duck()
        assert system.SimpleAudioVolume.level == pytest.approx(0.15)

    def test_double_duck_does_not_compound(self):
        s = FakeSession(1, 0.8)
        d = ducker([s])
        d.duck()
        d.duck()  # e.g. a stray extra state transition
        assert s.SimpleAudioVolume.level == pytest.approx(0.2)
        d.restore()
        assert s.SimpleAudioVolume.level == pytest.approx(0.8)

    def test_restore_without_duck_is_a_noop(self):
        s = FakeSession(1, 0.8)
        ducker([s]).restore()
        assert s.SimpleAudioVolume.level == 0.8
        assert s.SimpleAudioVolume.calls == 0

    def test_level_is_clamped(self):
        s = FakeSession(1, 0.5)
        ducker([s], level=-3.0).duck()
        assert s.SimpleAudioVolume.level == 0.0


class TestFailureIsQuietlyPartial:
    """COM calls fail whenever a session's app dies; the rest must still work."""

    def test_enumeration_failure_does_not_raise(self):
        def boom():
            raise OSError("no default audio device")

        SessionDucker(0.2, get_sessions=boom, own_pid=OWN_PID).duck()  # must not raise

    def test_one_bad_session_does_not_stop_the_others(self):
        class BadVolume:
            def GetMasterVolume(self):
                raise OSError("session gone")

        bad = FakeSession(1, 0.0)
        bad.SimpleAudioVolume = BadVolume()
        good = FakeSession(2, 0.8)
        d = ducker([bad, good])
        d.duck()
        assert good.SimpleAudioVolume.level == pytest.approx(0.2)
        d.restore()
        assert good.SimpleAudioVolume.level == pytest.approx(0.8)

    def test_failed_restore_still_clears_state(self):
        class DiesOnRestore(FakeVolume):
            def SetMasterVolume(self, value, _ctx):
                if self.calls:  # the duck worked; the restore does not
                    raise OSError("session gone")
                super().SetMasterVolume(value, _ctx)

        gone, alive = FakeSession(1, 0.8), FakeSession(2, 0.4)
        gone.SimpleAudioVolume = DiesOnRestore(0.8)
        d = ducker([gone, alive])
        d.duck()
        d.restore()  # must not raise, and the healthy session comes back
        assert alive.SimpleAudioVolume.level == pytest.approx(0.4)
        assert d._saved == []


class TestCreateDucker:
    def test_disabled_gives_noop(self):
        assert isinstance(create_ducker(Config(duck_other_audio=False)), NoOpDucker)

    def test_default_volume_is_full_silence(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        d = create_ducker(Config(duck_other_audio=True))
        assert d._level == 0.0

    def test_enabled_on_windows_gives_session_ducker(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        d = create_ducker(Config(duck_other_audio=True, duck_volume=0.3))
        assert isinstance(d, SessionDucker)
        assert d._level == pytest.approx(0.3)

    def test_enabled_elsewhere_gives_noop(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert isinstance(create_ducker(Config(duck_other_audio=True)), NoOpDucker)
