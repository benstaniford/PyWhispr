"""Command-line entry points for PyWhispr."""

from __future__ import annotations

import argparse
import logging
import sys

from pywhispr import __version__

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pywhispr",
        description="Local, hotkey-driven voice dictation. Run with no arguments to start the app.",
    )
    parser.add_argument("--version", action="version", version=f"pywhispr {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="start the dictation app (default)")

    p_transcribe = sub.add_parser("transcribe", help="transcribe a wav file and print the text")
    p_transcribe.add_argument("wav", help="path to a wav file")

    sub.add_parser("devices", help="list audio input devices")

    p_record = sub.add_parser(
        "record", help="record from the microphone for N seconds and transcribe (mic test)"
    )
    p_record.add_argument(
        "--seconds", type=float, default=5.0, help="recording duration (default: 5)"
    )

    sub.add_parser("download", help="pre-download the speech-to-text model")

    sub.add_parser(
        "diagnose", help="print environment details and test-load the model (for bug reports)"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pywhispr.logging_setup import setup_logging

    log_file = setup_logging(verbose=args.verbose or None)

    command = args.command or "run"
    if log_file is not None:
        log.debug("Logging to %s", log_file)

    if command == "run":
        from pywhispr.app import run_app

        return run_app()

    if command == "transcribe":
        return _cmd_transcribe(args.wav)

    if command == "devices":
        return _cmd_devices()

    if command == "record":
        return _cmd_record(args.seconds)

    if command == "download":
        return _cmd_download()

    if command == "diagnose":
        return _cmd_diagnose()

    return 2


def _load_backend():
    from pywhispr.config import load_config
    from pywhispr.stt import create_backend

    backend = create_backend(load_config())
    log.info("Loading model (%s)...", backend.name)
    backend.load()
    return backend


def _cmd_transcribe(wav_path: str) -> int:
    from pywhispr.stt.wav import read_wav_mono_16k

    audio = read_wav_mono_16k(wav_path)
    backend = _load_backend()
    print(backend.transcribe(audio))
    return 0


def _cmd_devices() -> int:
    import sounddevice as sd

    default_input = sd.default.device[0]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = "*" if idx == default_input else " "
            print(f"{marker} [{idx}] {dev['name']} ({dev['max_input_channels']} ch)")
    print("\n(* = default input device)")
    return 0


def _cmd_record(seconds: float) -> int:
    import time

    from pywhispr.audio import AudioRecorder
    from pywhispr.config import load_config

    cfg = load_config()
    backend = _load_backend()

    recorder = AudioRecorder(device=cfg.input_device)
    print(f"Recording for {seconds:.0f}s... speak now.", file=sys.stderr)
    recorder.start()
    time.sleep(seconds)
    audio = recorder.stop()
    print("Transcribing...", file=sys.stderr)
    print(backend.transcribe(audio))
    return 0


def _cmd_download() -> int:
    _load_backend()
    print("Model downloaded and loaded successfully.")
    return 0


def _cmd_diagnose() -> int:
    """Everything a bug report needs, in one console-visible run.

    The packaged app hides these failures behind a tray icon; running this from
    a terminal on the same machine reproduces the load with the output visible.
    """
    import time

    import numpy as np

    from pywhispr.logging_setup import environment_report, log_path

    for line in environment_report():
        print(line)

    print("\n--- microphone ---")
    try:
        import sounddevice as sd

        inputs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        print(f"{len(inputs)} input device(s); default: {sd.query_devices(kind='input')['name']}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")

    print("\n--- model load ---")
    started = time.monotonic()
    try:
        backend = _load_backend()
    except Exception as exc:
        log.exception("Model load failed")
        print(f"FAILED after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}")
        print(f"\nFull traceback in {log_path()}")
        return 1
    print(f"loaded in {time.monotonic() - started:.1f}s")

    print("\n--- transcribe (1s of silence) ---")
    started = time.monotonic()
    try:
        text = backend.transcribe(np.zeros(16000, dtype=np.float32))
    except Exception as exc:
        log.exception("Transcription failed")
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"ok in {time.monotonic() - started:.1f}s (result: {text!r})")

    print(f"\nAll checks passed. Log: {log_path()}")
    return 0
