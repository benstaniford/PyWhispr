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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    command = args.command or "run"

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
