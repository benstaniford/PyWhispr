# PyWhispr

Local, hotkey-driven voice dictation — a Superwhisper / Wispr Flow alternative that runs
entirely on your machine with open source models. Press a hotkey, speak while a small
waveform overlay shows you're being heard, press it again, and the transcribed text is
typed into whatever app has focus.

- **100% local** — audio never leaves your machine
- **Fast** — NVIDIA Parakeet TDT 0.6B v3, sub-second transcription of typical dictation
- **Cross-platform** — Apple Silicon Macs (via [MLX](https://github.com/ml-explore/mlx))
  and PCs with NVIDIA GPUs (via ONNX Runtime + CUDA), with CPU fallback

## Install

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is fetched automatically):

```sh
git clone <this repo> && cd PyWhispr
uv sync
uv run pywhispr download   # optional: pre-download the model (~600 MB, one time)
uv run pywhispr            # start the app
```

The right speech backend is selected automatically:

| Platform | Backend | Runtime |
|---|---|---|
| Apple Silicon Mac | `parakeet-mlx` | MLX (Metal GPU) |
| PC with NVIDIA GPU | `onnx-asr` | ONNX Runtime, CUDAExecutionProvider |
| Anything else | `onnx-asr` | ONNX Runtime, CPU |

## Usage

1. Start `uv run pywhispr` — a microphone icon appears in the menu bar / system tray.
2. Press the hotkey (**⌘⇧Space** on Mac, **Ctrl+Alt+Space** on Windows) to start recording.
   An overlay with a live waveform appears at the bottom of the screen.
3. Speak, then press the hotkey again. The text is pasted into the focused app and your
   previous clipboard text is restored.

Other commands: `pywhispr devices` (list microphones), `pywhispr record --seconds 5`
(mic test), `pywhispr transcribe file.wav`, `pywhispr download`.

## macOS permissions

PyWhispr works with **zero permission grants** out of the box: the global hotkey uses
the Carbon `RegisterEventHotKey` API (the same approach as Superwhisper), which needs
no Input Monitoring, and without Accessibility the transcript is left on the clipboard
with a "press ⌘V" notification instead of being pasted automatically. Microphone
access is prompted automatically on first recording.

For fully automatic pasting, grant one optional permission to the **terminal app you
launch PyWhispr from**, then fully restart that terminal:

- **Accessibility** (System Settings → Privacy & Security) — lets PyWhispr synthesize
  the ⌘V keystroke so text lands in the focused app by itself, with your previous
  clipboard restored afterwards.

You can also start/stop dictation from the tray icon menu — no hotkey needed at all.

## Windows / NVIDIA notes

- An RTX 50-series GPU (Blackwell) needs `onnxruntime-gpu` ≥ 1.22 (pinned already), an
  NVIDIA driver ≥ 570 and CUDA 12.8+ runtime.
- If CUDA isn't usable, PyWhispr logs a warning and runs on CPU (still fast — Parakeet
  is a 0.6B model).

## Configuration

`~/Library/Application Support/PyWhispr/config.toml` (macOS) or
`%APPDATA%\PyWhispr\config.toml` (Windows) — also reachable via the tray menu.

| Key | Default | Meaning |
|---|---|---|
| `hotkey` | `<cmd>+<shift>+<space>` / `<ctrl>+<alt>+<space>` | Toggle chord — easiest to change via tray menu → **Change hotkey…**, which records a keypress. Modifiers + a letter, digit, `<space>`, arrows/navigation keys or `<f1>`–`<f20>` |
| `input_device` | system default | Microphone index from `pywhispr devices` |
| `model_override` | platform default | Any compatible HuggingFace repo id |
| `max_recording_seconds` | `120` | Auto-stop guard |
| `play_sounds` | `true` | Start/stop audio cues |
| `paste_delay_ms` | `150` | Clipboard settle time before pasting |
| `clipboard_restore_delay_ms` | `300` | Wait before restoring your old clipboard |

## Corporate proxies / TLS interception

If model downloads fail with certificate errors (common behind corporate TLS
inspection), point Python at a CA bundle that includes your organisation's root
certificate:

```sh
export SSL_CERT_FILE=/path/to/ca-bundle.pem REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
```

For `uv` itself, use `export UV_SYSTEM_CERTS=1`.

## Building a macOS app bundle

```sh
packaging/make_icns.sh                     # regenerate the .icns from assets/icon.png
uv run --with pyinstaller pyinstaller packaging/pywhispr.spec --noconfirm
cp -R dist/PyWhispr.app ~/Applications/
```

The bundle is menu-bar-only (`LSUIElement`), ad-hoc signed by PyInstaller, and needs no
permissions to run (see above). Because it's built locally it never gets macOS's
quarantine attribute, so Gatekeeper won't object even on managed machines — and the
microphone prompt shows "PyWhispr" instead of your terminal's name.

## Releases

Pushing a tag like `v0.2.0` triggers GitHub Actions to build and attach to the release:

- `PyWhispr-<tag>-macos-arm64.zip` — the `.app` bundle (PyInstaller, ad-hoc signed)
- `PyWhispr-<tag>-windows-x64.msi` — per-user installer (cx_Freeze), no admin required

```sh
git tag v0.2.0 && git push origin v0.2.0
```

Bump `version` in `pyproject.toml` and `src/pywhispr/__init__.py` first. Note the
downloaded macOS zip carries the quarantine attribute, so on first launch users must
right-click → Open (the bundle is ad-hoc signed, not notarized).

## Development

```sh
uv run pytest                # unit tests (fast, no model/hardware)
uv run pytest -m model       # downloads and runs the real model on a fixture
uv run python -m pywhispr.ui.overlay   # visual overlay demo
uv run ruff check src tests
```

Architecture: `app.py` runs a small state machine (idle → recording → transcribing →
inserting) on the Qt main thread. The mic callback (PortAudio thread), hotkey listener
(pynput thread) and STT worker all communicate via queued Qt signals. STT backends
implement `stt/base.py::STTBackend` and take in-memory 16 kHz float32 numpy audio, so
they're testable without a microphone.

## License

GPL-3.0 — see [LICENSE](LICENSE).
