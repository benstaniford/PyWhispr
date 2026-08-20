"""One running instance, and how to ask it to stop.

Two upgrades needed this. On Windows the MSI replaces files the running
``PyWhispr.exe`` still has open, so the installer has to stop the old build first
or Windows Installer falls back to its "files in use" dialog — asking the user to
close an app that has no window. On macOS there is no installer at all: the
bundle is dragged over the old one and the old process keeps running from the
deleted inode, leaving two tray icons, two hotkey registrations and two model
loads.

So: ownership of "the instance" is a platform primitive rather than a
convention, and the owner can be asked to exit gracefully from another process.

**Ownership is not a pid file.** A pid file goes stale the moment a process is
killed, and the resulting guesswork is how single-instance guards end up either
refusing to start after a crash or terminating a stranger that inherited the pid.
Both primitives here are released by the OS however the process dies:

* Windows — a named event. The name exists only while some handle to it is open,
  and ``CreateEventW`` reports ``ERROR_ALREADY_EXISTS`` atomically, so two
  simultaneous launches cannot both win. ``Local\\`` rather than ``Global\\``: the
  install is per-user, another logged-on user's instance is none of our business,
  and ``Global\\`` would need a non-default ACL to be openable at all.
* POSIX — an exclusive ``flock`` on the state file, which is also what makes the
  pid *in* that file trustworthy: only the lock holder ever writes it.

**The quit signal is a watcher thread, not SIGTERM.** Python's signal handlers
only run when the interpreter regains control, and it never does while blocked in
``QApplication.exec()`` — so SIGTERM would need a ``set_wakeup_fd`` +
``QSocketNotifier`` hop or a permanent polling ``QTimer``, and either puts Qt in
here. A thread that blocks until asked is the same shape on both platforms and
keeps this module Qt-free, which is also what lets ``cli.py`` use it without
importing PySide6.

Nothing here logs a path the user did not choose or anything about their words;
pids, versions and counts only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from pywhispr import __version__, flavor

log = logging.getLogger(__name__)

# Long enough for _quit() to stop the API server (which joins its thread with a
# 5s timeout) and close the audio stream, short enough that an installer is not
# left waiting on a wedged app.
QUIT_TIMEOUT_SECONDS = 15.0
TERMINATE_TIMEOUT_SECONDS = 5.0
_POLL_SECONDS = 0.1

_ERROR_ALREADY_EXISTS = 183
_INFINITE = 0xFFFFFFFF
_WAIT_OBJECT_0 = 0
_EVENT_MODIFY_STATE = 0x0002
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_TERMINATE = 0x0001


# -- where the state lives ---------------------------------------------------


def _state_dir() -> Path:
    """The config directory, which is flavour-aware (``logging_setup``'s is not).

    PyWhisprLite is a separate product that must coexist with a full install, so
    it must not find — or stop — the other one's instance.
    """
    from pywhispr.config import config_path

    return config_path().parent


def _state_path() -> Path:
    return _state_dir() / "instance.json"


def _socket_path() -> Path:
    return _state_dir() / "instance.sock"


def _event_name() -> str:
    return rf"Local\{flavor.PRODUCT_NAME}.quit"


def _own_exe() -> str:
    return sys.executable or ""


def _read_state() -> dict:
    """The owner's ``{pid, version, exe}``, or ``{}``. Never raises: a truncated
    or hand-edited file must not stop the app from starting."""
    try:
        with open(_state_path(), encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _state_payload() -> bytes:
    return json.dumps(
        {"pid": os.getpid(), "version": __version__, "exe": _own_exe()},
        indent=2,
    ).encode("utf-8")


# -- Windows: a named event --------------------------------------------------


def _kernel32():
    """Declared per call, the way the rest of the codebase reaches Win32.

    ``use_last_error`` because ``CreateEventW`` answers "you were not first"
    through ``GetLastError``, not through its return value.
    """
    import ctypes
    from ctypes import wintypes

    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    dll.CreateEventW.restype = wintypes.HANDLE
    dll.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    dll.OpenEventW.restype = wintypes.HANDLE
    dll.SetEvent.argtypes = [wintypes.HANDLE]
    dll.SetEvent.restype = wintypes.BOOL
    dll.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    dll.WaitForSingleObject.restype = wintypes.DWORD
    dll.CloseHandle.argtypes = [wintypes.HANDLE]
    dll.CloseHandle.restype = wintypes.BOOL
    dll.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    dll.OpenProcess.restype = wintypes.HANDLE
    dll.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    dll.QueryFullProcessImageNameW.restype = wintypes.BOOL
    dll.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    dll.TerminateProcess.restype = wintypes.BOOL
    return dll


class _WindowsOwner:
    """Holds the named event for the process lifetime."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self._releasing = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def take() -> _WindowsOwner | None:
        import ctypes

        dll = _kernel32()
        # Manual reset, so a request that arrives before the app is listening is
        # still there when it starts to.
        handle = dll.CreateEventW(None, True, False, _event_name())
        already = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if already:
            dll.CloseHandle(handle)
            return None
        return _WindowsOwner(handle)

    def start_watching(self, callback: Callable[[], None]) -> None:
        dll = _kernel32()

        def wait() -> None:
            if dll.WaitForSingleObject(self._handle, _INFINITE) != _WAIT_OBJECT_0:
                return
            if self._releasing:  # our own wake-up, not a request
                return
            callback()

        self._thread = threading.Thread(target=wait, name="pywhispr-quit", daemon=True)
        self._thread.start()

    def write_state(self) -> None:
        _state_path().parent.mkdir(parents=True, exist_ok=True)
        _state_path().write_bytes(_state_payload())

    def release(self) -> None:
        dll = _kernel32()
        self._releasing = True
        dll.SetEvent(self._handle)  # so the waiter is not blocked on a handle we close
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        dll.CloseHandle(self._handle)
        with contextlib.suppress(OSError):
            _state_path().unlink()


# -- POSIX: an flock plus a unix socket --------------------------------------


class _PosixOwner:
    """Holds the flock on the state file, and listens on the socket beside it."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._releasing = False
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def take() -> _PosixOwner | None:
        import fcntl

        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return _PosixOwner(fd)

    def start_watching(self, callback: Callable[[], None]) -> None:
        path = _socket_path()
        # We hold the lock, so nothing alive owns this socket: any file here is a
        # leftover, and rebinding is the only way to be reachable.
        with contextlib.suppress(OSError):
            path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(path))
            server.listen(1)
        except OSError:
            # A socket path is length-limited (~104 bytes) and the config
            # directory is not ours to shorten. Losing the graceful stop is
            # survivable; refusing to start is not.
            server.close()
            log.warning("Could not listen for quit requests", exc_info=True)
            return
        self._server = server

        def wait() -> None:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            connection.close()
            if self._releasing:  # our own wake-up, not a request
                return
            callback()

        self._thread = threading.Thread(target=wait, name="pywhispr-quit", daemon=True)
        self._thread.start()

    def write_state(self) -> None:
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.truncate(self._fd, 0)
        os.write(self._fd, _state_payload())
        os.fsync(self._fd)

    def release(self) -> None:
        self._releasing = True
        if self._server is not None:
            # Connecting to ourselves is what unblocks accept(); closing a
            # listening socket does not reliably wake a thread already in it.
            with contextlib.suppress(OSError), socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            ) as waker:
                waker.connect(str(_socket_path()))
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._server.close()
            with contextlib.suppress(OSError):
                _socket_path().unlink()
        with contextlib.suppress(OSError):
            _state_path().unlink()
        os.close(self._fd)  # releases the flock


class _NullOwner:
    """What ``acquire`` falls back to when the mechanism itself failed.

    A tray app that will not start is indistinguishable from a crash, so a broken
    guard costs the guard and nothing else.
    """

    def start_watching(self, callback: Callable[[], None]) -> None:
        pass

    def write_state(self) -> None:
        pass

    def release(self) -> None:
        pass


def _take():
    return _WindowsOwner.take() if sys.platform == "win32" else _PosixOwner.take()


# -- asking the owner to stop ------------------------------------------------


def is_running() -> bool:
    """Is an instance of this product holding ownership right now?

    True for our own process too: ownership is a lock, and asking about it from
    the process that holds it is not a question this needs to distinguish.
    """
    if sys.platform == "win32":
        dll = _kernel32()
        handle = dll.OpenEventW(_EVENT_MODIFY_STATE, False, _event_name())
        if not handle:
            return False
        dll.CloseHandle(handle)
        return True

    import fcntl

    try:
        fd = os.open(_state_path(), os.O_RDWR)
    except OSError:
        return False  # no state file at all
    try:
        # A second open in this process gets its own file description, so this
        # answers True for our own instance too — which is what a caller in
        # another process needs, and harmless in the one that owns it.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _signal() -> bool:
    """Deliver the quit request. False means nothing was listening."""
    if sys.platform == "win32":
        dll = _kernel32()
        handle = dll.OpenEventW(_EVENT_MODIFY_STATE, False, _event_name())
        if not handle:
            return False
        try:
            return bool(dll.SetEvent(handle))
        finally:
            dll.CloseHandle(handle)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(str(_socket_path()))
        return True
    except OSError:
        return False


def _terminate(state: dict) -> bool:
    """Last resort. Verified against the recorded executable first, because a pid
    outlives nothing and could by then belong to a stranger."""
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        log.warning("No usable pid recorded; cannot terminate")
        return False

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        dll = _kernel32()
        access = _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE
        handle = dll.OpenProcess(access, False, pid)
        if not handle:
            return False
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not dll.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return False
            recorded = str(state.get("exe") or "")
            if buffer.value.casefold() != recorded.casefold():
                log.warning("Pid %d is not the recorded executable; leaving it alone", pid)
                return False
            log.warning("Terminating pid %d", pid)
            return bool(dll.TerminateProcess(handle, 1))
        finally:
            dll.CloseHandle(handle)

    import signal

    try:
        log.warning("Terminating pid %d", pid)
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        log.warning("Could not terminate pid %d", pid, exc_info=True)
        return False


def _wait_until_gone(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while is_running():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def request_quit(timeout: float | None = None) -> bool:
    """Ask any running instance to exit, and make sure it does.

    Escalates to terminating the process, because the caller is usually an
    installer that must not be blocked: the alternative to a lost dictation is a
    half-replaced install. Returns True when nothing is running any more.

    ``timeout`` resolves here rather than in the signature so that the module
    constant is the one place it is configured.
    """
    if timeout is None:
        timeout = QUIT_TIMEOUT_SECONDS
    if not is_running():
        log.debug("Nothing is running")
        return True

    state = _read_state()
    log.info(
        "Asking the running instance to exit (pid %s, version %s)",
        state.get("pid"),
        state.get("version"),
    )
    if _signal():
        if _wait_until_gone(timeout):
            log.info("The running instance exited")
            return True
        log.warning("Still running after %.0fs; escalating", timeout)
    else:
        # Nothing is listening — an old build with no quit command, or a socket
        # that could not be bound — so there is nothing to wait for. Waiting the
        # full grace period anyway would just be an installer standing still.
        log.warning("Nothing was listening for the request; escalating at once")
    _terminate(state)
    if _wait_until_gone(TERMINATE_TIMEOUT_SECONDS):
        return True
    log.error("Could not stop the running instance")
    return False


# -- ownership ---------------------------------------------------------------


class Guard:
    """Ownership, held for the process lifetime, plus the hook the app quits on.

    Injected into ``PyWhisprApp`` rather than reached as module state, so the test
    suite — which builds the app directly and never goes through ``run_app`` —
    creates no named events and no sockets.
    """

    def __init__(self, owner) -> None:
        self._owner = owner
        self._lock = threading.Lock()
        self._callback: Callable[[], None] | None = None
        self._requested = False
        self._released = False
        owner.write_state()
        owner.start_watching(self._on_request)

    def on_quit(self, callback: Callable[[], None]) -> None:
        """Register what a request should do, replaying one that already arrived.

        The installer can signal during a twenty-second model load, long before
        the app is in a state to be told about it.
        """
        with self._lock:
            self._callback = callback
            pending = self._requested
        if pending:
            log.info("Replaying a quit request that arrived before the app was ready")
            callback()

    def _on_request(self) -> None:
        """Called on the watcher thread. The callback is a Qt signal's ``emit``,
        which is the same cross-thread hop the transcription worker uses."""
        log.info("Another process asked us to exit")
        with self._lock:
            self._requested = True
            callback = self._callback
        if callback is not None:
            callback()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._owner.release()
        except Exception:
            log.exception("Releasing instance ownership failed")


def acquire() -> Guard | None:
    """Take ownership, displacing a different build if one is running.

    ``None`` means "do not start": an instance of *this* build is already running.
    Every other outcome — including the mechanism failing outright — returns a
    guard, because a tray app that silently declines to start is worse than a
    duplicate one.
    """
    try:
        return _acquire()
    except Exception:
        log.exception("Instance ownership failed; starting anyway")
        return Guard(_NullOwner())


def _acquire() -> Guard | None:
    owner = _take()
    if owner is not None:
        return Guard(owner)

    state = _read_state()
    if state.get("version") == __version__:
        # The same build: the user has what they asked for, and there is no tray
        # icon yet to say so with.
        log.info("Already running (pid %s); not starting a second copy", state.get("pid"))
        return None

    # A different *version* is the upgrade case: on macOS the bundle has just been
    # replaced under a still-running process, and on Windows it means the
    # installer's custom action did not manage it. Deliberately not "a different
    # executable" as well, tempting as it looks — that would have `uv run
    # pywhispr` during development silently kill the installed app, and the
    # installed app kill the dev run.
    log.info(
        "A different build is running (version %s, pid %s); asking it to exit",
        state.get("version"),
        state.get("pid"),
    )
    request_quit()
    owner = _take()
    if owner is None:
        log.error("The running instance would not give up ownership; not starting")
        return None
    return Guard(owner)


__all__ = ["Guard", "acquire", "is_running", "request_quit", "QUIT_TIMEOUT_SECONDS"]
