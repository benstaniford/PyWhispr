"""How long another process waits for text we put on the clipboard.

``QClipboard::setText`` on Windows goes through ``OleSetClipboard``, which
publishes an ``IDataObject`` and renders the text **on demand**: the pasting app's
read is a COM call back into *our* process, served by *our* Qt main thread. So a
main thread that is busy — a loaded machine, a long-running slot — makes the
paste late even though the clipboard was set instantly and the Ctrl+V arrived on
time. Nothing needs the keyboard or the focus to show it: the reader just reads.

    python scripts/clipboard_stall_probe.py            # baseline and 4 stalls
    python scripts/clipboard_stall_probe.py --flush     # with OleFlushClipboard

Focus-free and load-free, so it is safe to run on a machine somebody is using.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TEXT = "The quick brown fox jumps over the lazy dog. " * 8
STALLS_MS = (0, 200, 500, 1500)

CF_UNICODETEXT = 13


def read_clipboard_timed() -> tuple[float, int]:
    """(milliseconds, characters) for one raw CF_UNICODETEXT read."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

    started = time.perf_counter()
    text = ""
    for _ in range(500):  # ~5s of retries; OpenClipboard fails while others hold it
        if user32.OpenClipboard(None):
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if handle:
                    pointer = kernel32.GlobalLock(handle)
                    if pointer:
                        try:
                            text = ctypes.c_wchar_p(pointer).value or ""
                        finally:
                            kernel32.GlobalUnlock(handle)
                break
            finally:
                user32.CloseClipboard()
        time.sleep(0.01)
    return (time.perf_counter() - started) * 1000, len(text)


def run_reader(control: Path, results: Path) -> int:
    """Wait for a marker, then read the clipboard and report how long it took."""
    seen = 0
    deadline = time.monotonic() + 300
    with open(results, "a", buffering=1, encoding="utf-8") as out:
        out.write(json.dumps({"kind": "ready"}) + "\n")
        while time.monotonic() < deadline:
            lines = control.read_text(encoding="utf-8").splitlines() if control.exists() else []
            if len(lines) > seen:
                request = json.loads(lines[seen])
                seen += 1
                if request.get("kind") == "stop":
                    return 0
                elapsed_ms, length = read_clipboard_timed()
                out.write(
                    json.dumps({"kind": "read", "ms": elapsed_ms, "chars": length}) + "\n"
                )
            time.sleep(0.005)
    return 0


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def wait_for_lines(path: Path, count: int, timeout: float = 30.0, pump: bool = False) -> list[dict]:
    """Wait for the reader. ``pump`` runs the Qt event loop while waiting.

    Without pumping, a delayed-render clipboard is never served and this simply
    times out — which is the whole point of the no-flush case, so the caller
    decides when the main thread is allowed to answer.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = read_lines(path)
        if len(lines) >= count:
            return lines
        if pump:
            from PySide6.QtCore import QEventLoop, QTimer

            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec()
        else:
            time.sleep(0.02)
    raise RuntimeError(f"reader produced {len(read_lines(path))} lines, not {count}")


def run_driver(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication

    scratch = Path(args.dir)
    control = scratch / "clipboard-stall-control.jsonl"
    results = scratch / "clipboard-stall-results.jsonl"
    control.write_text("", encoding="utf-8")
    results.write_text("", encoding="utf-8")

    reader = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--reader", "--dir", str(scratch)]
    )
    try:
        wait_for_lines(results, 1)
        app = QApplication(sys.argv)
        clipboard = app.clipboard()
        expected = 1

        print(f"flush={args.flush}, {len(TEXT)} characters")
        print(f"{'main thread busy':>18}   {'reader wait':>12}   chars")
        for stall_ms in STALLS_MS:
            clipboard.setText(TEXT)
            if args.flush:
                # Render now, so the reader gets bytes off the clipboard instead
                # of a COM call into a process that is not answering.
                ctypes.windll.ole32.OleFlushClipboard()
            with open(control, "a", encoding="utf-8") as out:
                out.write(json.dumps({"kind": "read"}) + "\n")
            # Deliberately not pumping: this *is* the starved main thread. Then
            # the loop is let go again, as a real one would be once scheduled.
            time.sleep(stall_ms / 1000)
            expected += 1
            line = wait_for_lines(results, expected, pump=True)[-1]
            print(f"{stall_ms:>15} ms   {line['ms']:>9.0f} ms   {line['chars']}")

        with open(control, "a", encoding="utf-8") as out:
            out.write(json.dumps({"kind": "stop"}) + "\n")
        return 0
    finally:
        reader.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", action="store_true")
    parser.add_argument("--flush", action="store_true")
    parser.add_argument("--dir", default=os.environ.get("TEMP", "."))
    args = parser.parse_args()
    if args.reader:
        scratch = Path(args.dir)
        return run_reader(
            scratch / "clipboard-stall-control.jsonl", scratch / "clipboard-stall-results.jsonl"
        )
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
