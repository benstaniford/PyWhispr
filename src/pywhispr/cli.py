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

    p_download = sub.add_parser("download", help="pre-download the speech-to-text model")
    p_download.add_argument(
        "--quantization",
        default=None,
        help='model variant to fetch ("int8", or "" for full precision); '
        "default: whatever the config says",
    )

    sub.add_parser("enable-gpu", help="download the CUDA libraries for NVIDIA GPU acceleration")
    sub.add_parser("disable-gpu", help="remove the downloaded CUDA libraries")
    sub.add_parser(
        "enable-directml",
        help="GPU acceleration for cards CUDA cannot use (pre-Turing NVIDIA, AMD, Intel)",
    )
    sub.add_parser("disable-directml", help="remove the DirectML build of onnxruntime")
    p_verify = sub.add_parser(
        "verify-gpu", help="report whether transcription really runs on the GPU"
    )
    p_verify.add_argument(
        "--quantization",
        default=None,
        help="model variant to check with (default: whatever the config says). "
        "The check only needs a model the GPU can load, so passing the variant "
        "already downloaded avoids fetching another one.",
    )

    sub.add_parser(
        "diagnose", help="print environment details and test-load the model (for bug reports)"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pywhispr.logging_setup import setup_logging

    log_file = setup_logging(verbose=args.verbose or None)

    from pywhispr.certs import use_system_certificates

    use_system_certificates()

    # Before anything imports huggingface_hub: it reads its cache path at import
    # time, so a later override would be ignored. Every subcommand wants it — the
    # download, the GPU check and the app all have to agree on where the model is.
    from pywhispr.config import load_config
    from pywhispr.storage import apply_overrides

    config = load_config()
    apply_overrides(config)

    command = args.command or "run"

    # Before any onnxruntime import, and skipped for the commands that install or
    # remove it: activating the copy being deleted only confuses the failure.
    if command not in ("enable-directml", "disable-directml"):
        from pywhispr.directml import activate_if_enabled

        activate_if_enabled(config)
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
        return _cmd_download(args.quantization)

    if command == "diagnose":
        return _cmd_diagnose()

    if command == "enable-gpu":
        return _cmd_enable_gpu()

    if command == "disable-gpu":
        return _cmd_disable_gpu()

    if command == "enable-directml":
        return _cmd_enable_directml()

    if command == "disable-directml":
        return _cmd_disable_directml()

    if command == "verify-gpu":
        return _cmd_verify_gpu(args.quantization)

    return 2


def _load_backend(quantization: str | None = None):
    from pywhispr.config import load_config
    from pywhispr.stt import create_backend

    config = load_config()
    if quantization is not None:
        config.model_quantization = quantization
    backend = create_backend(config)
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


def _cmd_download(quantization: str | None = None) -> int:
    _load_backend(quantization)
    print("Model downloaded and loaded successfully.")
    return 0


def _cmd_enable_gpu() -> int:
    from pywhispr import cuda

    offer, why_not = cuda.can_offer()
    if not offer and not cuda.is_installed():
        print(f"Not available: {why_not}")
        return 1

    if not cuda.is_installed():
        print(f"Downloading the CUDA libraries (~{cuda.APPROXIMATE_DOWNLOAD_MB} MB)...")

        def progress(fraction: float, message: str) -> bool:
            print(f"\r{fraction * 100:5.1f}%  {message:<60}", end="", flush=True)
            return True

        try:
            cuda.download(progress)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 1
        except Exception as exc:
            log.exception("CUDA download failed")
            print(f"\nFailed: {type(exc).__name__}: {exc}")
            return 1
        print()

    print("Checking that transcription really runs on the GPU...")
    works, detail = cuda.verify()
    print(f"{'Ready' if works else 'Not working'}: {detail}")
    return 0 if works else 1


def _cmd_disable_gpu() -> int:
    from pywhispr import cuda

    print("Removed the CUDA libraries." if cuda.remove() else "Nothing to remove.")
    return 0


def _cmd_enable_directml() -> int:
    """Fetch the DirectML onnxruntime, then check it in a fresh process.

    A fresh one because this process has already resolved its providers — the same
    reason `enable-gpu` verifies in a subprocess.
    """
    from pywhispr import cuda, directml
    from pywhispr.config import load_config, save_config

    offer, why_not = directml.can_offer()
    if not offer and not directml.is_installed():
        print(f"Not available: {why_not}")
        return 1

    if not directml.is_installed():
        print(f"Downloading {directml.PACKAGE} (~{directml.APPROXIMATE_DOWNLOAD_MB} MB)...")

        def progress(fraction: float, message: str) -> bool:
            print(f"{fraction * 100:5.1f}%  {message}")
            return True

        try:
            directml.download(progress)
        except KeyboardInterrupt:
            print("Cancelled.")
            return 1
        except Exception as exc:
            log.exception("DirectML download failed")
            print(f"Failed: {type(exc).__name__}: {exc}")
            return 1

    config = load_config()
    if config.use_directml is False:
        config.use_directml = True
        save_config(config)

    print("Checking that transcription really runs on the GPU...")
    works, detail = cuda.verify()  # the same verify-gpu subprocess: it reports any provider
    print(f"{'Ready' if works else 'Not working'}: {detail}")
    if works:
        print("Restart PyWhispr to start using it.")
    return 0 if works else 1


def _cmd_disable_directml() -> int:
    from pywhispr import directml
    from pywhispr.config import load_config, save_config

    installed = directml.is_installed()
    directml.remove()
    config = load_config()
    if config.use_directml:
        config.use_directml = None
        save_config(config)
    print("Removed the DirectML onnxruntime." if installed else "Nothing to remove.")
    return 0


def _cmd_verify_gpu(quantization: str | None = None) -> int:
    """Load the model here and now, and report the providers actually in use.

    Run in a subprocess by `cuda.verify()`, so its output is a single line and its
    exit code is the answer: onnxruntime only resolves providers once per process,
    which is why this cannot be checked in the app that just installed them.

    ``quantization`` exists so the check can reuse the variant already downloaded.
    Left to itself it would see working CUDA, choose full precision and fetch 2.4 GB
    inside a step the user is only being shown as "checking…".
    """
    import numpy as np

    from pywhispr.stt.onnx_backend import session_providers

    backend = _load_backend(quantization)
    providers = session_providers(getattr(backend, "_model", None))
    accelerated = sorted(p for p in providers if p != "CPUExecutionProvider")
    if not accelerated:
        print("transcription runs on the CPU: no GPU execution provider loaded")
        return 1
    backend.transcribe(np.zeros(16000, dtype=np.float32))  # prove it can actually run
    print(f"transcription runs on the GPU via {', '.join(accelerated)}")
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
