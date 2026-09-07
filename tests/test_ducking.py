# Nothing here may touch the real machine: every ducker is built with its
# injection seam (get_sessions= / get_controls=) supplying fakes, so no test
# enumerates a real audio session or moves a real volume.
import sys

import pytest

from pywhispr import coreaudio
from pywhispr.config import Config
from pywhispr.ducking import (
    NoOpDucker,
    OutputVolumeDucker,
    SessionDucker,
    create_ducker,
    supported,
    system_wide,
)

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

    def test_enabled_on_macos_gives_output_volume_ducker(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        d = create_ducker(Config(duck_other_audio=True, duck_volume=0.3))
        assert isinstance(d, OutputVolumeDucker)
        assert d._level == pytest.approx(0.3)
        assert d.cue_lead_ms > 0  # the start cue is inside a system-wide dip

    def test_enabled_elsewhere_gives_noop(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert isinstance(create_ducker(Config(duck_other_audio=True)), NoOpDucker)


class FakeControl:
    def __init__(self, level: float):
        self.level = level
        self.sets = 0

    def get(self) -> float:
        return self.level

    def set(self, value: float) -> None:
        self.sets += 1
        self.level = value


def mac_ducker(controls, level=0.25):
    return OutputVolumeDucker(level, get_controls=lambda: controls)


class TestOutputVolumeDuckAndRestore:
    def test_duck_scales_the_device_relative_to_its_own_level(self):
        control = FakeControl(0.8)
        mac_ducker([control]).duck()
        assert control.level == pytest.approx(0.2)

    def test_restore_puts_back_the_exact_original(self):
        # The real thing reads a Float32, so the value has noise in it; ducking
        # must return that value, not a rounded-off version of it.
        original = 0.7720596790313721
        control = FakeControl(original)
        d = mac_ducker([control])
        d.duck()
        d.restore()
        assert control.level == original

    def test_per_channel_controls_keep_their_balance(self):
        controls = [FakeControl(0.8), FakeControl(0.4), FakeControl(0.6)]
        d = mac_ducker(controls)
        d.duck()
        assert [c.level for c in controls] == pytest.approx([0.2, 0.1, 0.15])
        d.restore()
        assert [c.level for c in controls] == pytest.approx([0.8, 0.4, 0.6])

    def test_ducking_twice_does_not_compound(self):
        control = FakeControl(0.8)
        d = mac_ducker([control])
        d.duck()
        d.duck()
        assert control.level == pytest.approx(0.2)
        d.restore()
        assert control.level == pytest.approx(0.8)

    def test_restore_without_duck_touches_nothing(self):
        control = FakeControl(0.8)
        mac_ducker([control]).restore()
        assert control.sets == 0
        assert control.level == pytest.approx(0.8)

    def test_level_is_clamped_both_ends(self):
        assert OutputVolumeDucker(-3.0)._level == 0.0
        assert OutputVolumeDucker(5.0)._level == 1.0

    def test_default_volume_silences_the_machine_and_still_restores(self):
        # The default config: duck_volume 0.0 mutes the Mac for the recording.
        control = FakeControl(0.8)
        d = mac_ducker([control], level=0.0)
        d.duck()
        assert control.level == 0.0
        d.restore()
        assert control.level == pytest.approx(0.8)

    def test_the_default_seam_is_coreaudio(self):
        # Pins the wiring without calling it, which would move a real volume.
        assert OutputVolumeDucker(0.2)._get_controls is coreaudio.output_volume_controls


class TestOutputVolumeFailureIsQuietlyPartial:
    def test_failing_lookup_does_not_stop_the_recording(self):
        def boom():
            raise OSError("no default output device")

        OutputVolumeDucker(0.25, get_controls=boom).duck()  # must not raise

    def test_a_device_with_no_settable_volume_is_not_an_error(self):
        # An aggregate or Multi-Output device, HDMI, some external DACs.
        d = mac_ducker([])
        d.duck()
        assert d._saved == []
        d.restore()  # must not raise

    def test_one_bad_control_does_not_stop_the_others(self):
        class BadControl(FakeControl):
            def get(self) -> float:
                raise OSError("the device went away")

        alive = FakeControl(0.8)
        d = mac_ducker([BadControl(0.4), alive])
        d.duck()
        assert alive.level == pytest.approx(0.2)
        d.restore()
        assert alive.level == pytest.approx(0.8)

    def test_a_control_dying_on_restore_still_clears_the_saved_state(self):
        class DiesOnRestore(FakeControl):
            def set(self, value: float) -> None:
                super().set(value)
                if self.sets > 1:
                    raise OSError("the device went away")

        alive = FakeControl(0.8)
        d = mac_ducker([DiesOnRestore(0.4), alive])
        d.duck()
        d.restore()
        assert alive.level == pytest.approx(0.8)
        assert d._saved == []


class TestPlatformPredicates:
    @pytest.mark.parametrize(
        ("platform", "can_duck", "whole_machine"),
        [("win32", True, False), ("darwin", True, True), ("linux", False, False)],
    )
    def test_predicates_by_platform(self, platform, can_duck, whole_machine):
        assert supported(platform) is can_duck
        assert system_wide(platform) is whole_machine

    @pytest.mark.parametrize(
        ("platform", "can_duck", "whole_machine"),
        [("win32", True, False), ("darwin", True, True), ("linux", False, False)],
    )
    def test_predicates_read_sys_platform_at_call_time(
        self, monkeypatch, platform, can_duck, whole_machine
    ):
        # The form that would break if either grew a `= sys.platform` default,
        # which binds at import and is the idiom the rest of this file relies on.
        monkeypatch.setattr(sys, "platform", platform)
        assert supported() is can_duck
        assert system_wide() is whole_machine


class TestCoreAudioHelpers:
    # Pure functions only: the rest of coreaudio.py is unmockable OS calls, which
    # is why it has no test file of its own. These two run on every platform.
    def test_fourcc_matches_the_documented_selector(self):
        assert coreaudio._fourcc("vmvc") == 1986885219
        assert coreaudio._fourcc("dOut") == 1682929012

    def test_status_text_names_the_error_the_way_the_headers_do(self):
        assert "who?" in coreaudio.status_text(0x77686F3F)
        assert "0x77686f3f" in coreaudio.status_text(0x77686F3F)
