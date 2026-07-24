# PyWhispr

Local, hotkey-driven voice dictation — a Superwhisper / Wispr Flow alternative that runs
entirely on your machine with open source models. Press a hotkey, speak while a small
waveform overlay shows you're being heard, press it again, and the transcribed text is
typed into whatever app has focus.

- **100% local** — audio never leaves your machine
- **Fast** — NVIDIA Parakeet TDT 0.6B v3, sub-second transcription of typical dictation
- **Cross-platform** — Apple Silicon Macs (via [MLX](https://github.com/ml-explore/mlx))
  and PCs with NVIDIA GPUs (via ONNX Runtime + CUDA), with CPU fallback
- **Shareable** — an open [HTTP API](#network-api) lets phones, browsers and games on
  your network use the same warm model

## Install

Just downlaod the latest [release]( https://github.com/benstaniford/PyWhispr/releases) for Mac / Windows and run the installer.

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

**Exception — double-tap hotkeys:** a hotkey like double-tap-Option can't use the
permission-free API; it has to listen for raw modifier taps, which macOS gates behind
**Input Monitoring**. If you pick one via **Change hotkey…**, PyWhispr requests the
permission and shows what to grant; relaunch it afterwards. Chord hotkeys never need
this. (On Windows, double-tap works without any special permission.)

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
| `hotkey` | `<cmd>+<shift>+<space>` / `<ctrl>+<alt>+<space>` | Toggle hotkey — easiest to change via tray menu → **Change hotkey…**, which records a keypress. Either a chord (modifiers + a letter, digit, `<space>`, arrows/navigation keys or `<f1>`–`<f20>`) or a modifier double-tap like `double-tap:<alt>` |
| `input_device` | system default | Microphone index from `pywhispr devices` |
| `model_override` | platform default | Any compatible HuggingFace repo id |
| `max_recording_seconds` | `120` | Auto-stop guard |
| `play_sounds` | `true` | Start/stop audio cues |
| `paste_delay_ms` | `150` | Clipboard settle time before pasting |
| `clipboard_restore_delay_ms` | `300` | Wait before restoring your old clipboard |
| `api_enabled` | `true` | Serve the [network API](#network-api) |
| `api_host` | `0.0.0.0` | Bind address — set to `127.0.0.1` to keep it on this machine |
| `api_port` | `9149` | Listening port |
| `api_max_audio_seconds` | `300` | Longest clip accepted per request |
| `api_max_queue` | `4` | Requests in flight before new ones get `503 busy` |

## Network API

PyWhispr keeps the Parakeet model loaded the whole time it's running, so it can serve
transcriptions to other devices for free. Phones, browsers, games and scripts on your
network POST audio to `http://<your-machine>:9149` and get text back — no cloud, no API
key, no per-minute billing.

It only ever converts audio to text. It cannot start a recording on the host, read its
microphone, or type anything into it.

> **There is no authentication.** Anything that can reach port 9149 can spend your CPU
> and GPU. That's fine on a home LAN; on a shared or untrusted network set
> `api_host = "127.0.0.1"`, or `api_enabled = false` to close the port entirely.
> Expect a firewall prompt on first launch (macOS "allow incoming connections", Windows
> Defender Firewall) — the API is unreachable from other machines until you allow it.

### `GET /v1/health`

```json
{
  "status": "ready",
  "version": "0.2.4",
  "backend": "parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)",
  "sample_rate": 16000,
  "max_audio_seconds": 300,
  "max_upload_bytes": 38400000,
  "queue_depth": 0,
  "max_queue": 4,
  "pcm_formats": ["f32le", "s16le"]
}
```

`status` is `loading` while the model warms up (roughly the first 10 s, or minutes on a
first run that downloads it), then `ready`. Poll this before sending audio.

### `POST /v1/transcribe`

Three body encodings, whichever suits your client:

| `Content-Type` | Body |
|---|---|
| `audio/wav` | a complete wav file — 16-bit or 32-bit, any sample rate, mono or stereo |
| `application/octet-stream` | headerless PCM, described by the query parameters below |
| `multipart/form-data` | a form field named `audio` |

Query parameters (raw PCM only): `sample_rate` (default `16000`), `channels`
(default `1`), `format` — `f32le` (default) or `s16le`. Downmixing and resampling to
16 kHz mono happen server-side. A wav is recognised by its `RIFF` header whatever
content type you declare.

```json
{
  "text": "hello world",
  "backend": "parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)",
  "audio_seconds": 2.13,
  "processing_seconds": 0.41
}
```

Errors are `{"error": {"code": "...", "message": "..."}}` with a stable `code`:
`model_loading` and `busy` (both 503, with `Retry-After`), `bad_audio` (400),
`unsupported_media_type` (415), `payload_too_large` (413), `transcribe_failed` (500).

Requests share the single STT worker with local dictation, so a request that arrives
mid-dictation simply waits its turn. Beyond `api_max_queue` in flight you get `busy`
rather than an unbounded backlog.

**No codecs.** Wav and raw PCM only — no mp3, m4a or Opus. In particular the browser's
`MediaRecorder` produces WebM/Opus, which won't work; use the Web Audio recipe below.

### Clients

```sh
curl -s http://192.168.1.20:9149/v1/health
curl -s -X POST http://192.168.1.20:9149/v1/transcribe \
     -H 'Content-Type: audio/wav' --data-binary @clip.wav
```

CORS is open, so browsers can call it directly. Send float32 samples straight from an
`AudioContext`:

```js
// buffer: an AudioBuffer captured via getUserMedia + AudioWorklet/ScriptProcessor
const pcm = buffer.getChannelData(0);           // Float32Array
const res = await fetch(
  `http://192.168.1.20:9149/v1/transcribe?sample_rate=${buffer.sampleRate}&format=f32le`,
  {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: pcm},
);
const {text} = await res.json();
```

Unity / C#:

```csharp
var pcm = new byte[clip.samples * 4];
var samples = new float[clip.samples];
clip.GetData(samples, 0);
Buffer.BlockCopy(samples, 0, pcm, 0, pcm.Length);

var url = $"http://192.168.1.20:9149/v1/transcribe?sample_rate={clip.frequency}&format=f32le";
using var req = new UnityWebRequest(url, "POST") {
    uploadHandler = new UploadHandlerRaw(pcm) {contentType = "application/octet-stream"},
    downloadHandler = new DownloadHandlerBuffer(),
};
yield return req.SendWebRequest();
Debug.Log(req.downloadHandler.text);   // {"text": "...", ...}
```

Streaming (partial results over a WebSocket) isn't implemented; the `/v1` prefix leaves
room to add it without breaking these clients.

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
they're testable without a microphone. `api.py` is a stdlib `ThreadingHTTPServer` that
funnels every request through that same single-worker executor, so the model is only
ever used by one thread at a time.

## License

GPL-3.0 — see [LICENSE](LICENSE).
