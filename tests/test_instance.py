"""One running instance, and the handshake an upgrade needs.

No Qt and no `unittest.mock` anywhere in here: this is the module whose whole job
is to work across process and thread boundaries, so the tests use real processes,
real threads and real sockets. (Mocks on a second thread are what hung the suite
in test_app.py — see CLAUDE.md.)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from pywhispr import instance

HOLDER = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    import pywhispr.instance as inst

    state, version, obey = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "obey"
    inst._state_dir = lambda: state
    inst.__version__ = version
    guard = inst.acquire()
    assert guard is not None, "the child could not take ownership"
    if obey:
        guard.on_quit(lambda: (guard.release(), os._exit(0)))
    (state / "ready").write_text("1")
    time.sleep(120)
    """
)


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Somewhere short, on POSIX.

    A unix socket path is limited to about 104 bytes and pytest's `tmp_path`
    spends most of that on its own nesting, so a naive fixture exercises the
    "could not bind" degradation instead of the feature. (Which is itself worth
    knowing: the real path, `~/Library/Application Support/PyWhispr`, spends 68.)
    """
    if sys.platform == "win32":
        directory = tmp_path
    else:
        directory = Path(tempfile.mkdtemp(prefix="pw-", dir="/tmp"))
    monkeypatch.setattr(instance, "_state_dir", lambda: directory)
    yield directory
    if directory is not tmp_path:
        shutil.rmtree(directory, ignore_errors=True)


def _holder(state_dir, version="0.0.1", obey="obey"):
    """A real second process owning the instance, as a pretend other build."""
    script = state_dir / "holder.py"
    script.write_text(HOLDER)
    child = subprocess.Popen([sys.executable, str(script), str(state_dir), version, obey])
    deadline = time.monotonic() + 20
    while not (state_dir / "ready").exists():
        assert child.poll() is None, "the holder exited before taking ownership"
        assert time.monotonic() < deadline, "the holder never became ready"
        time.sleep(0.02)
    return child


class TestOwnership:
    def test_nothing_is_running_to_begin_with(self, state_dir):
        assert instance.is_running() is False

    def test_acquire_takes_ownership_and_records_it(self, state_dir):
        guard = instance.acquire()
        try:
            assert instance.is_running() is True
            state = json.loads((state_dir / "instance.json").read_text())
            assert state["pid"] == os.getpid()
            assert state["version"] == instance.__version__
        finally:
            guard.release()

    def test_release_leaves_nothing_behind(self, state_dir):
        instance.acquire().release()
        assert instance.is_running() is False
        assert not (state_dir / "instance.json").exists()
        assert not (state_dir / "instance.sock").exists()

    def test_release_twice_is_harmless(self, state_dir):
        guard = instance.acquire()
        guard.release()
        guard.release()

    def test_the_same_build_is_refused(self, state_dir):
        """Not displaced: the user has what they asked for, and restarting it would
        cost them a model load for nothing."""
        guard = instance.acquire()
        try:
            assert instance.acquire() is None
        finally:
            guard.release()

    def test_a_broken_mechanism_still_starts_the_app(self, state_dir, monkeypatch):
        """A tray app that declines to start is indistinguishable from a crash."""

        def explode():
            raise OSError("no such thing")

        monkeypatch.setattr(instance, "_take", explode)
        guard = instance.acquire()
        assert guard is not None
        guard.release()  # the null owner: nothing to undo, and it must not raise


class TestRequestQuit:
    def test_nothing_running_is_success(self, state_dir):
        assert instance.request_quit() is True

    def test_the_owner_is_told(self, state_dir):
        guard = instance.acquire()
        told = threading.Event()
        guard.on_quit(told.set)
        try:
            assert instance._signal() is True
            assert told.wait(5) is True
        finally:
            guard.release()

    def test_a_request_before_the_app_is_listening_is_replayed(self, state_dir):
        """The installer can signal during a twenty-second model load."""
        guard = instance.acquire()
        try:
            assert instance._signal() is True
            time.sleep(0.3)
            told = threading.Event()
            guard.on_quit(told.set)
            assert told.is_set() is True
        finally:
            guard.release()

    def test_it_waits_for_a_cooperative_instance_to_go(self, state_dir):
        child = _holder(state_dir)
        try:
            assert instance.request_quit(timeout=20) is True
            assert instance.is_running() is False
            assert child.wait(5) == 0
        finally:
            if child.poll() is None:
                child.kill()

    def test_an_instance_that_ignores_the_request_is_terminated(self, state_dir):
        """The installer must never be blocked: by this point _quit has already
        restored the mixer levels and closed the stream, so the only thing lost is
        a dictation that was being discarded anyway."""
        child = _holder(state_dir, obey="ignore")
        try:
            assert instance.request_quit(timeout=1) is True
            assert instance.is_running() is False
            assert child.wait(5) != 0
        finally:
            if child.poll() is None:
                child.kill()

    def test_no_recorded_pid_means_no_terminating(self, state_dir):
        """A pid we cannot vouch for is not one to kill."""
        assert instance._terminate({}) is False
        assert instance._terminate({"pid": os.getpid()}) is False


class TestDisplacement:
    def test_a_different_build_is_asked_to_go(self, state_dir):
        """The macOS upgrade: the bundle is replaced under a running process, so the
        newer one has to displace the older."""
        child = _holder(state_dir, version="0.0.1")
        try:
            guard = instance.acquire()
            assert guard is not None
            guard.release()
            assert child.wait(5) == 0
        finally:
            if child.poll() is None:
                child.kill()

    def test_a_wedged_different_build_is_terminated(self, state_dir, monkeypatch):
        monkeypatch.setattr(instance, "QUIT_TIMEOUT_SECONDS", 1.0)
        child = _holder(state_dir, version="0.0.1", obey="ignore")
        try:
            guard = instance.acquire()
            assert guard is not None
            guard.release()
        finally:
            if child.poll() is None:
                child.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="the POSIX primitives")
class TestPosixLeftovers:
    def test_a_stale_socket_is_reclaimed(self, state_dir):
        """A killed process leaves the file behind; the flock is what says nothing
        alive owns it, so rebinding is safe."""
        (state_dir / "instance.sock").write_text("not really a socket")
        guard = instance.acquire()
        try:
            assert instance._signal() is True
        finally:
            guard.release()

    def test_a_hand_written_state_file_does_not_stop_the_app(self, state_dir):
        (state_dir / "instance.json").write_text("{ not json")
        guard = instance.acquire()
        assert guard is not None
        guard.release()
