# CLAUDE.md

Project notes for working on PyWhispr — a local voice-dictation menu-bar app
(Parakeet STT via MLX on macOS / ONNX on Windows, PySide6 UI).

## Build & test

```sh
uv sync                                   # Python is pinned to 3.12 (see below)
QT_QPA_PLATFORM=offscreen uv run pytest   # headless Qt for CI/agents
uv run pytest -m model                    # opt-in: downloads & runs the real model
uv run ruff check src tests
```

STT backends take in-memory numpy audio, so they're testable without a mic.

## This machine is corporate-managed (Jamf MDM) — expect friction

- **TLS interception breaks downloads.** Use `UV_SYSTEM_CERTS=1` for `uv`. For
  Python/HuggingFace model downloads, build a CA bundle (certifi + system
  keychain certs) and set `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`. Once the
  model is cached, normal runs work offline.
- **Python 3.12**, not newer: `parakeet-mlx` → `librosa`/`numba` need a numpy
  the newest CPython lacks wheels for. `pyproject.toml` caps `<3.14` and pins
  `numba>=0.61` to stop bad backtracking.
- `/Applications` is not writable; install the bundle to `~/Applications`.

## Logging

`pywhispr/logging_setup.py` owns all of it. Everything goes to a rotating file
(`user_log_dir`) because the Windows build is a console-less GUI exe where
stderr is `None` and otherwise nothing survives. It also installs
`sys.excepthook` + `threading.excepthook` (worker-thread crashes used to vanish)
and a Qt message handler. `PYWHISPR_DEBUG=1` turns on debug for the packaged
app; `pywhispr diagnose` reproduces startup in a terminal with output visible.

**Rule:** failures that leave the app unusable must not quit it. A tray app that
exits is indistinguishable from a crash — report the error in the tray tooltip,
the overlay and the log, and stay running.

## Continuation joining (`join.py` + `caret.py`)

Each recording is transcribed alone, so the model capitalises and adds a full
stop every time and consecutive dictations collide. `join.py` is a pure
`(preceding, text) -> str` that adds a space and lower-cases a continuation
word; `caret.py` finds `preceding`.

- **Only ever additive.** `join_text` returns exactly `("" | " ") + text` with at
  most the first letter re-cased — never Backspace, never an edit to text
  already in the document. `app._joined` re-checks that invariant and falls back
  to the raw transcript, because losing a transcript is unrecoverable (the audio
  is gone) and an exception escaping `_on_transcribed` strands the app in
  `INSERTING` where no hotkey works.
- Lower-casing is gated on a **closed list** of function words, not a
  shape heuristic: an unlisted word keeps its capital (today's behaviour),
  whereas a loose rule produces `ben`/`monday`. One-letter words are filtered
  out of the list so `I` is safe structurally.
- **Import `HIServices`, not `ApplicationServices`** — every AX symbol is in the
  former; the umbrella drags in CoreText for nothing. pyobjc is already a
  transitive dep of pynput, so no new dependency and no spec change.
- AX reads need Accessibility (the grant auto-paste already has), so they only
  work **from the bundle** — `AXIsProcessTrusted()` is False for a CLI `python`.
  Set `AXUIElementSetMessagingTimeout`: AX is synchronous IPC on the GUI thread,
  and a wedged Electron app would otherwise freeze the tray.
- Plenty of apps expose no caret (Electron, Java, terminals). That is the
  everyday case, not an error: debug-level only, fall back to memory.
- **Never log the context text** — it is whatever window is focused, possibly a
  password field. Lengths only.
- Testing gotcha: `patch.dict(sys.modules, ...)` restores by wiping the dict, so
  anything imported for real *inside* the patch is deleted on exit. Faking
  `HIServices` while letting `CoreFoundation` import for real tore pyobjc out of
  `sys.modules` and made later tests pass for the wrong reason. Fake both.

## Debugging native crashes here

- **`lldb` attach is blocked by MDM** and **ReportCrash is disabled** (no
  `.ips` files). Don't rely on either.
- Use Python's `faulthandler` instead (in-process, no permission needed).
  `launch.py` enables it. Run the bundle binary **directly** with stderr
  redirected to a file (not via `open`, which sends stderr to the system log).
- Read the exit code: **139 = SIGSEGV** (faulthandler dumps it by default),
  **133 = SIGTRAP** (native abort in Qt/Metal/ObjC — faulthandler does NOT
  catch it by default; register it explicitly if needed).

## Hotkeys (the fiddly part)

- **Chords** (e.g. `<cmd>+<shift>+<space>`) use Carbon `RegisterEventHotKey`
  on macOS — permission-free.
- **Double-tap** (`double-tap:<alt>`) uses a Cocoa **NSEvent global monitor**
  on macOS, NOT pynput: pynput's `keyboard.Listener` builds its keycode map
  with Carbon calls off the main thread and **segfaults** alongside our Carbon
  hotkey. Double-tap needs the **Input Monitoring** permission; chords don't.
- pynput is used for both on Windows/Linux (fine there).
- Accessibility is optional: without it, text goes to the clipboard instead of
  auto-pasting.
- The app owns recording state; listeners only report events (incl. the
  double-tap hold duration for push-to-talk), so state can't desync.

## Making a release

```sh
./scripts/make-release   # bumps patch version in pyproject.toml + __init__.py,
                         # commits, tags vX.Y.Z, pushes → triggers the workflow
```

`.github/workflows/release.yml` builds the macOS `.app` (zip) and Windows MSI.
**Lesson:** parallel build jobs must NOT each create/update the release — that
races and drops assets. Build jobs upload artifacts; a single final job
publishes them together. Bump the version files for real (the MSI's in-place
upgrade depends on it).

## Packaging (`packaging/`)

- `pywhispr.spec` — PyInstaller, menu-bar-only (`LSUIElement`), ad-hoc signed.
  **`mlx.metallib` must be copied next to the relocated `libmlx.dylib`** or MLX
  crashes on load ("Failed to load the default metallib").
- `make_icns.sh` / `PyWhispr.ico` — icons; `scripts/make_icon.py` regenerates
  the base PNG (background removal + speech bubble).
