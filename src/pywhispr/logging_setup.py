"""Central logging setup, plus the startup environment report.

Why this exists: the Windows build is a console-less GUI executable, so
``sys.stdout``/``sys.stderr`` are ``None`` and anything logged to them is lost.
When the app misbehaves there — the model never finishes loading, a hotkey does
nothing, the process dies without a window — a **file** is the only evidence.
So logging always goes to a rotating file in the platform log directory, and
the tray menu can open it.

Two files are written, both in :func:`log_dir`:

``pywhispr.log``
    The application log (rotating). Everything ``logging`` produces, plus
    uncaught exceptions from any thread and Qt's own warnings.
``pywhispr-stderr.log``
    Raw stdout/stderr for frozen builds that have no console: tqdm's model
    download bar, stray ``print()``s, and ``faulthandler``'s native-crash
    tracebacks. Only created when the process really has no console.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import platform
import signal
import sys
import threading
from pathlib import Path

from platformdirs import user_log_dir

from pywhispr import __version__

APP_NAME = "PyWhispr"
LOG_FILENAME = "pywhispr.log"
STDERR_FILENAME = "pywhispr-stderr.log"

MAX_BYTES = 2_000_000
BACKUP_COUNT = 3
FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"

# Env var so a user can get debug logs out of the packaged app, where there is
# no command line to pass --verbose on.
DEBUG_ENV_VAR = "PYWHISPR_DEBUG"

log = logging.getLogger(__name__)

_configured = False


def log_dir() -> Path:
    """Platform log directory: ``%LOCALAPPDATA%\\PyWhispr\\Logs`` on Windows,
    ``~/Library/Logs/PyWhispr`` on macOS, ``~/.local/state/PyWhispr/log`` on Linux."""
    return Path(user_log_dir(APP_NAME, appauthor=False))


def log_path() -> Path:
    return log_dir() / LOG_FILENAME


def stderr_path() -> Path:
    return log_dir() / STDERR_FILENAME


def debug_enabled() -> bool:
    return os.environ.get(DEBUG_ENV_VAR, "").strip().lower() not in ("", "0", "false", "no")


def redirect_stdio_if_headless() -> Path | None:
    """Give a console-less build real stdout/stderr, returning the file used.

    A frozen Windows GUI process has ``sys.stdout is sys.stderr is None``.
    Every write to them then raises ``AttributeError: 'NoneType' object has no
    attribute 'write'`` — which is how the first-run HuggingFace download bar
    used to kill the app before it ever loaded a model. Returns ``None`` when
    the streams were already fine (normal console runs).
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None

    path: Path | None = stderr_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError:
        # Nowhere writable: swallowing the writes still beats crashing on them.
        stream = open(os.devnull, "w")
        path = None

    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return path


def setup_logging(verbose: bool | None = None, *, to_file: bool = True) -> Path | None:
    """Configure root logging. Idempotent; returns the log file path, if any.

    Call once, as early as possible — before importing anything heavy, so that
    import-time failures are captured too.
    """
    global _configured
    if _configured:
        return log_path() if to_file else None

    level = logging.DEBUG if (debug_enabled() if verbose is None else verbose) else logging.INFO
    formatter = logging.Formatter(FORMAT)
    root = logging.getLogger()
    root.setLevel(level)

    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        root.addHandler(console)

    file_path: Path | None = None
    if to_file:
        try:
            log_dir().mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                log_path(), maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            handler.setFormatter(formatter)
            root.addHandler(handler)
            file_path = log_path()
        except OSError as exc:  # read-only home, locked file, ...
            root.warning("Could not open log file %s: %s", log_path(), exc)

    _install_exception_hooks()
    _enable_faulthandler()
    _configured = True
    return file_path


def install_qt_message_handler() -> None:
    """Route Qt's own warnings into our log.

    Qt normally prints to stderr, which the packaged Windows build discards.
    Its warnings ("QWidget: Cannot create a window...", missing platform
    plugins) are exactly what you need when the UI fails to appear.
    """
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    qt_log = logging.getLogger("qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message):
        qt_log.log(levels.get(mode, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)


def _install_exception_hooks() -> None:
    """Log uncaught exceptions from the main thread and from any other thread.

    Without this a crash in a worker thread prints to a stderr nobody reads and
    the app simply stops doing that thing, with no trace of why.
    """

    def excepthook(exc_type, exc_value, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, tb)
            return
        logging.getLogger("pywhispr").critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, tb)
        )

    def threadhook(args):
        if issubclass(args.exc_type, SystemExit):
            return
        name = args.thread.name if args.thread is not None else "?"
        logging.getLogger("pywhispr").critical(
            "Uncaught exception in thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = excepthook
    threading.excepthook = threadhook


def _enable_faulthandler() -> None:
    """Dump a native traceback on segfault/abort into whatever stderr now is."""
    try:
        faulthandler.enable()
    except (RuntimeError, ValueError, AttributeError):
        return  # stderr has no usable file descriptor
    # SIGTRAP is how Qt/Metal/ObjC aborts surface on macOS and is not covered
    # by enable() alone. It does not exist on Windows.
    if hasattr(signal, "SIGTRAP"):
        try:
            faulthandler.register(signal.SIGTRAP, chain=True)
        except (RuntimeError, ValueError, OSError):
            pass


# -- environment report ------------------------------------------------------


def _package_version(module_name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(module_name)
    except PackageNotFoundError:
        return "not installed"
    except Exception as exc:  # metadata missing in frozen builds
        return f"unknown ({exc})"


def _onnxruntime_lines() -> list[str]:
    """What ONNX Runtime thinks it can do — the usual Windows failure point.

    ``get_available_providers()`` listing CUDA does *not* mean CUDA will work:
    onnxruntime-gpu advertises the provider it was compiled with, and only
    fails when a session is actually created without the CUDA runtime present.
    """
    try:
        import onnxruntime
    except Exception as exc:
        return [f"onnxruntime: import failed: {exc!r}"]

    version = getattr(onnxruntime, "__version__", "?")
    lines = [f"onnxruntime: {version} (device {onnxruntime.get_device()})"]
    try:
        lines.append(f"  providers advertised: {', '.join(onnxruntime.get_available_providers())}")
    except Exception as exc:
        lines.append(f"  providers advertised: query failed: {exc!r}")
    return lines


def _model_cache_lines() -> list[str]:
    """Where the model is cached and whether anything is actually there.

    A half-finished first-run download looks exactly like a hang, so the size
    on disk is worth knowing.
    """
    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
    except Exception as exc:
        return [f"model cache: unavailable ({exc!r})"]

    lines = [f"model cache: {cache} ({'exists' if cache.is_dir() else 'MISSING'})"]
    if cache.is_dir():
        for entry in sorted(cache.glob("models--*")):
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            incomplete = len(list(entry.rglob("*.incomplete")))
            note = f", {incomplete} incomplete file(s)" if incomplete else ""
            lines.append(f"  {entry.name}: {size / 1e6:.0f} MB{note}")
    return lines


def _network_env_lines() -> list[str]:
    """Proxy and TLS overrides — the reason downloads fail on managed machines."""
    interesting = (
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
        "HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT", "HF_HUB_OFFLINE",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
        "CUDA_PATH", "PYWHISPR_DEBUG",
    )
    found = {k: os.environ[k] for k in interesting if os.environ.get(k)}
    if not found:
        return ["environment: no proxy/cache/TLS overrides set"]
    return ["environment:"] + [f"  {k}={v}" for k, v in found.items()]


def environment_report() -> list[str]:
    """Everything worth knowing about this machine, as loggable lines."""
    from pywhispr.config import config_path

    frozen = getattr(sys, "frozen", False)
    lines = [
        f"PyWhispr {__version__}",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"frozen: {frozen}"
        + (f" (bundle {getattr(sys, '_MEIPASS', sys.prefix)})" if frozen else ""),
        f"platform: {platform.platform()} / {platform.machine()}",
        f"config: {config_path()} ({'exists' if config_path().exists() else 'will be created'})",
        f"log file: {log_path()}",
        f"stderr file: {stderr_path()} (frozen no-console builds only)",
        "versions: "
        + ", ".join(
            f"{name}={_package_version(name)}"
            for name in ("PySide6", "numpy", "sounddevice", "pynput", "onnx-asr", "huggingface-hub")
        ),
    ]
    lines += _onnxruntime_lines()
    lines += _model_cache_lines()
    lines += _network_env_lines()
    return lines


def log_environment() -> None:
    for line in environment_report():
        log.info("%s", line)
