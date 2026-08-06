"""Measure the post-transcription half of a dictation, idle and under CPU load.

The model is not involved: this drives the real ``TextInjector`` against a paste
target in a *separate* process, so the clipboard write, the synthesized Ctrl+V,
the low-level-hook chain it has to cross and the target's on-demand clipboard
read are all the real thing.

Three intervals are timed, which is what splits the blame:

``sendinput``  our ``SendInput`` call → the target's ``keyPressEvent`` for Ctrl+V.
               This is the WH_KEYBOARD_LL hook chain plus scheduling; nothing of
               ours runs in it.
``render``     the target's keypress → the target's ``textChanged``. This is the
               target asking OLE for the data, which for a Qt-set clipboard is a
               COM call back into *our* process.
``fixed``      the two ``QTimer`` hops in the injector (150 ms + 300 ms by
               default) and any lateness in them.

Usage (from the repo, with PySide6 + pynput importable):

    python scripts/lag_probe.py                     # idle baseline
    python scripts/lag_probe.py --load 16           # busy box
    python scripts/lag_probe.py --load 16 --no-hook # busy, our hook removed

Every process it starts is killed by PID on the way out.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

TEXT = "The quick brown fox jumps over the lazy dog. " * 8  # ~360 chars, like a real dictation


# -- the paste target (a second process) --------------------------------------


def run_target(events_path: Path) -> int:
    """A focused text box that timestamps the Ctrl+V keypress and the text arriving."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    events = open(events_path, "a", buffering=1, encoding="utf-8")

    def record(kind: str, **extra) -> None:
        events.write(json.dumps({"kind": kind, "t": time.time(), **extra}) + "\n")

    class Target(QPlainTextEdit):
        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_V and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                record("keypress")
            super().keyPressEvent(event)

    app = QApplication(sys.argv)
    target = Target()
    target.setWindowTitle("PyWhispr lag probe target")
    target.resize(520, 200)
    target.show()
    target.textChanged.connect(lambda: record("text", n=len(target.toPlainText())))

    hwnd = int(target.winId())
    user32 = ctypes.windll.user32

    SW_MINIMIZE, SW_RESTORE = 6, 9

    def keep_focused() -> None:
        # Sticky, because a paste into the wrong window would both spoil the
        # measurement and dump text into whatever the user is doing.
        if user32.GetForegroundWindow() == hwnd:
            return
        if not user32.SetForegroundWindow(hwnd):
            # Windows refuses SetForegroundWindow to a process that has not been
            # foreground recently; restoring from minimised activates anyway.
            user32.ShowWindow(hwnd, SW_MINIMIZE)
            user32.ShowWindow(hwnd, SW_RESTORE)
        target.activateWindow()
        target.setFocus()

    focus_timer = QTimer(target)
    focus_timer.timeout.connect(keep_focused)
    focus_timer.start(200)
    keep_focused()

    clear_timer = QTimer(target)
    clear_timer.timeout.connect(lambda: len(target.toPlainText()) > 20000 and target.clear())
    clear_timer.start(1000)

    QTimer.singleShot(200, lambda: record("ready", hwnd=hwnd))
    return app.exec()


# -- load generators ----------------------------------------------------------

SPINNER = "x = 0\nwhile True:\n    x = (x + 1) % 1000003\n"


def start_load(count: int) -> list[subprocess.Popen]:
    """`count` normal-priority Python spinners — a stand-in for a compile."""
    return [
        subprocess.Popen([sys.executable, "-c", SPINNER], creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(count)
    ]


def start_gil_pressure() -> threading.Event:
    """A pure-Python thread inside *our* process, contending for the GIL."""
    stop = threading.Event()

    def spin() -> None:
        x = 0
        while not stop.is_set():
            for _ in range(100_000):
                x = (x + 1) % 1000003

    threading.Thread(target=spin, daemon=True).start()
    return stop


# -- the driver ---------------------------------------------------------------


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wait_for_ready(path: Path, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in read_events(path):
            if event["kind"] == "ready":
                return int(event["hwnd"])
        time.sleep(0.1)
    raise RuntimeError("the paste target never reported ready")


def pump(seconds: float) -> None:
    """Wait with the Qt event loop running.

    ``time.sleep`` here would be a bug in the probe, not a delay: while we own
    the clipboard, Qt serves the data on demand over COM, so a main thread that
    is not pumping leaves the consumer (the clipboard history service, or the
    pasting app) holding the clipboard open — which then makes our *next*
    OleSetClipboard fail. That is a real failure mode, but it has to be
    provoked deliberately, not by the measuring rig.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def describe_window() -> str:
    """Title and pid of whatever is foreground — for when a paste is skipped."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buffer, 256)
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return f"hwnd {hwnd} pid {pid.value} {buffer.value!r}"


def wait_for_focus(hwnd: int, timeout: float = 3.0) -> bool:
    """Needs a QApplication: it pumps rather than sleeps (see :func:`pump`)."""
    user32 = ctypes.windll.user32
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if user32.GetForegroundWindow() == hwnd:
            return True
        pump(0.1)
    return False


def run_driver(args: argparse.Namespace) -> int:
    import logging

    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from pywhispr import perf
    from pywhispr.injector import TextInjector

    # The injector's own stage marks, which split the clipboard read from the
    # write and show how late the two QTimer hops were.
    os.environ[perf.ENV_VAR] = "1"
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    events_path = Path(args.events)
    events_path.write_text("", encoding="utf-8")

    target = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--target", "--events", str(events_path)]
    )
    load: list[subprocess.Popen] = []
    gil_stop: threading.Event | None = None
    listener = None
    try:
        target_hwnd = wait_for_ready(events_path)
        print(f"target hwnd {target_hwnd}, pid {target.pid}")

        if args.hook:
            # The same listener the app installs for `double-tap:<ctrl>`: a
            # system-wide WH_KEYBOARD_LL hook with a Python callback.
            from pywhispr.hotkey import DoubleTapListener

            listener = DoubleTapListener("double-tap:<ctrl>", lambda: None, lambda _held: None)
            listener.start()

        if args.load:
            load = start_load(args.load)
            print(f"started {len(load)} spinner(s): {[p.pid for p in load]}")
        if args.gil:
            gil_stop = start_gil_pressure()
        if load or gil_stop:
            time.sleep(2.0)  # let the schedulers settle

        app = QApplication(sys.argv)
        sent_at: list[float] = []

        class TimedInjector(TextInjector):
            def _send_paste_keystroke(self) -> None:
                sent_at.append(time.time())
                super()._send_paste_keystroke()

        injector = TimedInjector(args.paste_delay, args.restore_delay)
        user32 = ctypes.windll.user32
        rows = []

        for iteration in range(args.iterations):
            if not wait_for_focus(target_hwnd):
                print(f"iteration {iteration}: skipped, foreground is {describe_window()}")
                continue

            before = read_events(events_path)
            started = time.time()
            sent_at.clear()

            loop = QEventLoop()
            injector.finished.connect(loop.quit)
            perf.begin(f"insert[{iteration}]")
            QTimer.singleShot(0, lambda: injector.insert(TEXT))
            QTimer.singleShot(15000, loop.quit)  # never wedge the probe
            loop.exec()
            injector.finished.disconnect(loop.quit)
            perf.end()

            # Give a late paste time to land before the events are read.
            deadline = time.monotonic() + 5.0
            fresh: list[dict] = []
            while time.monotonic() < deadline:
                fresh = read_events(events_path)[len(before) :]
                if any(e["kind"] == "text" for e in fresh):
                    break
                pump(0.05)

            keypress = next((e["t"] for e in fresh if e["kind"] == "keypress"), None)
            arrived = next((e["t"] for e in fresh if e["kind"] == "text"), None)
            sent = sent_at[0] if sent_at else None
            rows.append(
                {
                    "sendinput_ms": (keypress - sent) * 1000 if keypress and sent else None,
                    "render_ms": (arrived - keypress) * 1000 if arrived and keypress else None,
                    "insert_to_sent_ms": (sent - started) * 1000 if sent else None,
                    "total_ms": (arrived - started) * 1000 if arrived else None,
                }
            )
            print(f"iteration {iteration}: {rows[-1]}")
            pump(args.gap)

        report(args, rows)
        return 0
    finally:
        if listener is not None:
            listener.stop()
        if gil_stop is not None:
            gil_stop.set()
        for process in load:
            process.kill()
        target.kill()


def report(args: argparse.Namespace, rows: list[dict]) -> None:
    label = f"load={args.load} hook={args.hook} gil={args.gil}"
    print(f"\n== {label}, {len(rows)} iteration(s) ==")
    for key in ("insert_to_sent_ms", "sendinput_ms", "render_ms", "total_ms"):
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            print(f"{key:<20} no samples")
            continue
        print(
            f"{key:<20} median {statistics.median(values):8.1f}  "
            f"min {min(values):8.1f}  max {max(values):8.1f}  n={len(values)}"
        )
    missing = sum(1 for row in rows if row["total_ms"] is None)
    if missing:
        print(f"{'lost pastes':<20} {missing} of {len(rows)} never arrived")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="store_true", help="run as the paste target")
    parser.add_argument("--events", default=None)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--load", type=int, default=0, help="CPU spinner processes")
    parser.add_argument("--gil", action="store_true", help="also add in-process GIL pressure")
    parser.add_argument(
        "--no-hook", dest="hook", action="store_false", help="do not install our keyboard hook"
    )
    parser.add_argument("--paste-delay", type=int, default=150)
    parser.add_argument("--restore-delay", type=int, default=300)
    parser.add_argument("--gap", type=float, default=0.7, help="seconds between iterations")
    parser.set_defaults(hook=True)
    args = parser.parse_args()

    if args.events is None:
        args.events = str(Path(os.environ.get("TEMP", ".")) / "pywhispr-lag-probe.jsonl")
    if args.target:
        return run_target(Path(args.events))
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
