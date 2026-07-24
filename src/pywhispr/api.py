"""Open HTTP transcription API, so other devices can use the warm local model.

The app already keeps a Parakeet model loaded for hotkey dictation; this exposes
it on the network as a plain request/response service. Audio in, text out —
nothing here touches the microphone, the clipboard, the keyboard or any widget,
so it can run entirely on its own threads.

Two constraints shape the design:

* The STT backend must never be called concurrently (one model, one GPU), so
  every request funnels through the app's single-worker executor via the
  injected ``transcribe`` callable and blocks on the result.
* It is stdlib-only on purpose. The surface is three routes; adding
  fastapi/uvicorn would mean hidden-import wrangling in both the PyInstaller
  and cx_Freeze packaging paths for no functional gain.

There is no authentication: it binds 0.0.0.0 by default and anyone on the
network can spend CPU here. See the README for the trade-off.
"""

from __future__ import annotations

import email.parser
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np

from pywhispr import __version__
from pywhispr.stt.base import SAMPLE_RATE
from pywhispr.stt.wav import pcm_to_mono_16k, read_wav_bytes_mono_16k

log = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9149

WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
PCM_CONTENT_TYPES = {"application/octet-stream", "audio/pcm", "audio/l16"}

# How long a request waits for its turn on the single STT worker before giving
# up, on top of the clip's own duration. Generous: it may be queued behind a
# local dictation that is still recording.
QUEUE_TIMEOUT_SECONDS = 120

# Socket-level timeout, so a client that opens a connection and stalls cannot
# tie up a server thread indefinitely.
REQUEST_TIMEOUT_SECONDS = 60


class ApiError(Exception):
    """An error with a fixed HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


def _decode_audio(data: bytes, content_type: str, query: dict[str, list[str]]) -> np.ndarray:
    """Decode a request body to mono float32 at 16 kHz.

    Wav is detected by its RIFF magic regardless of the declared content type,
    so a client that posts a wav as octet-stream still works.
    """
    if not data:
        raise ApiError(400, "bad_audio", "request body is empty")

    if data[:4] == b"RIFF" or content_type in WAV_CONTENT_TYPES:
        try:
            return read_wav_bytes_mono_16k(data)
        except (ValueError, EOFError) as exc:
            raise ApiError(400, "bad_audio", str(exc)) from exc

    if content_type not in PCM_CONTENT_TYPES:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"unsupported Content-Type {content_type!r}; send audio/wav, "
            "application/octet-stream (raw pcm) or multipart/form-data",
        )

    def param(name: str, default: str) -> str:
        return query.get(name, [default])[0]

    try:
        return pcm_to_mono_16k(
            data,
            sample_rate=int(param("sample_rate", str(SAMPLE_RATE))),
            channels=int(param("channels", "1")),
            fmt=param("format", "f32le"),
        )
    except ValueError as exc:
        raise ApiError(400, "bad_audio", str(exc)) from exc


def _extract_multipart(data: bytes, content_type_header: str) -> tuple[bytes, str]:
    """Pull the ``audio`` part out of a multipart body, with its content type."""
    parsed = email.parser.BytesParser().parsebytes(
        b"Content-Type: " + content_type_header.encode("latin-1") + b"\r\n\r\n" + data
    )
    if not parsed.is_multipart():
        raise ApiError(400, "bad_audio", "malformed multipart/form-data body")

    chosen = None
    for part in parsed.walk():
        if part.is_multipart():
            continue
        if part.get_param("name", header="content-disposition") == "audio":
            chosen = part
            break
        if chosen is None and part.get_filename():
            chosen = part
    if chosen is None:
        raise ApiError(400, "bad_audio", "multipart body has no 'audio' field")

    payload = chosen.get_payload(decode=True)
    if not isinstance(payload, bytes):
        raise ApiError(400, "bad_audio", "multipart 'audio' part has no binary content")
    return payload, chosen.get_content_type().lower()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = REQUEST_TIMEOUT_SECONDS
    server_version = f"PyWhispr/{__version__}"
    sys_version = ""

    # -- plumbing ----------------------------------------------------------

    @property
    def api(self) -> TranscriptionServer:
        return self.server.api  # type: ignore[attr-defined]

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        # Frozen builds have no console; keep this out of stderr.
        log.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: dict, extra_headers: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, exc: ApiError) -> None:
        headers = {"Connection": "close"}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        # The body may be unread (413/411), which would desync a kept-alive
        # connection, so always close after an error.
        self.close_connection = True
        self._send_json(exc.status, {"error": {"code": exc.code, "message": exc.message}}, headers)

    def _read_body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            raise ApiError(411, "length_required", "chunked bodies are not supported")

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ApiError(411, "length_required", "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError(400, "bad_audio", f"invalid Content-Length {raw_length!r}") from exc

        limit = self.api.max_upload_bytes
        if length > limit:
            raise ApiError(413, "payload_too_large", f"body exceeds the {limit} byte limit")
        return self.rfile.read(length)

    # -- routes ------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path == "/v1/health":
                self._send_json(200, self.api.health())
            elif path == "/":
                self._send_json(200, {**self.api.health(), "endpoints": _ENDPOINTS})
            else:
                raise ApiError(404, "not_found", f"no such endpoint: {path}")
        except ApiError as exc:
            self._send_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        path = parts.path.rstrip("/") or "/"
        try:
            if path != "/v1/transcribe":
                raise ApiError(404, "not_found", f"no such endpoint: {path}")
            self._send_json(200, self._transcribe(parse_qs(parts.query)))
        except ApiError as exc:
            self._send_error(exc)
        except Exception as exc:  # never let a handler thread kill the server
            log.exception("Unhandled error serving %s", self.path)
            self._send_error(ApiError(500, "internal_error", str(exc)))

    def _transcribe(self, query: dict[str, list[str]]) -> dict:
        content_type_header = self.headers.get("Content-Type", "application/octet-stream")
        content_type = content_type_header.split(";")[0].strip().lower()
        data = self._read_body()

        if content_type == "multipart/form-data":
            data, content_type = _extract_multipart(data, content_type_header)

        audio = _decode_audio(data, content_type, query)
        return self.api.run_transcription(audio)


_ENDPOINTS = {
    "GET /v1/health": "server status, backend and limits",
    "POST /v1/transcribe": "audio/wav, application/octet-stream (raw pcm) or multipart/form-data",
}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32  # stdlib default of 5 is thin for a LAN service

    def __init__(self, address, handler, api: TranscriptionServer):
        self.api = api
        super().__init__(address, handler)


class TranscriptionServer:
    """HTTP front end for the app's speech-to-text backend.

    ``transcribe`` and ``status`` are injected rather than an app reference, so
    this is testable without Qt or a real model.

    :param transcribe: called on a request thread with mono float32 16 kHz
        audio; must serialise access to the model internally and return text.
    :param status: returns ``{"status": "loading"|"ready"|"error", "backend": str}``.
    """

    def __init__(
        self,
        transcribe: Callable[[np.ndarray], str],
        status: Callable[[], dict],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_audio_seconds: int = 300,
        max_queue: int = 4,
    ):
        self._transcribe = transcribe
        self._status = status
        self.host = host
        self.port = port
        self.max_audio_seconds = max_audio_seconds
        self.max_queue = max(1, max_queue)
        # Enough for the duration cap as 16 kHz mono float32, doubled so a
        # stereo or 44.1 kHz upload of reasonable length still fits.
        self.max_upload_bytes = max(1 << 20, max_audio_seconds * SAMPLE_RATE * 4 * 2)

        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._inflight = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Bind and serve on a daemon thread. Returns False if the port is taken.

        A busy port must not be fatal — this is a menu-bar app whose main job is
        local dictation.
        """
        try:
            self._httpd = _Server((self.host, self.port), _Handler, self)
        except OSError as exc:
            log.error("Could not bind API to %s:%s: %s", self.host, self.port, exc)
            return False

        self.port = self._httpd.server_address[1]  # resolves port 0 in tests
        httpd = self._httpd
        self._thread = threading.Thread(
            # Short poll interval so stop() (and so quitting the app) is snappy.
            target=lambda: httpd.serve_forever(poll_interval=0.05),
            name="pywhispr-api",
            daemon=True,
        )
        self._thread.start()
        log.info("Transcription API listening on http://%s:%s", self.host, self.port)
        return True

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None
        log.info("Transcription API stopped")

    # -- request handling --------------------------------------------------

    def health(self) -> dict:
        status = self._status()
        return {
            "status": status.get("status", "unknown"),
            "version": __version__,
            "backend": status.get("backend"),
            "sample_rate": SAMPLE_RATE,
            "max_audio_seconds": self.max_audio_seconds,
            "max_upload_bytes": self.max_upload_bytes,
            "queue_depth": self._inflight,
            "max_queue": self.max_queue,
            "pcm_formats": ["f32le", "s16le"],
        }

    def run_transcription(self, audio: np.ndarray) -> dict:
        """Check readiness and capacity, then transcribe. Raises ApiError."""
        status = self._status()
        state = status.get("status")
        if state == "loading":
            raise ApiError(503, "model_loading", "the model is still loading", retry_after=5)
        if state != "ready":
            raise ApiError(503, "model_unavailable", "the model is not available", retry_after=30)

        audio_seconds = len(audio) / SAMPLE_RATE
        if audio_seconds > self.max_audio_seconds:
            raise ApiError(
                413,
                "payload_too_large",
                f"audio is {audio_seconds:.1f}s, limit is {self.max_audio_seconds}s",
            )

        with self._lock:
            if self._inflight >= self.max_queue:
                raise ApiError(503, "busy", "too many requests in flight", retry_after=2)
            self._inflight += 1
        try:
            started = time.monotonic()
            text = self._transcribe(audio)
            elapsed = time.monotonic() - started
        except FutureTimeoutError as exc:
            raise ApiError(504, "timeout", "transcription timed out") from exc
        except ApiError:
            raise
        except Exception as exc:
            log.exception("Transcription failed")
            raise ApiError(500, "transcribe_failed", str(exc)) from exc
        finally:
            with self._lock:
                self._inflight -= 1

        return {
            "text": text,
            "backend": status.get("backend"),
            "audio_seconds": round(audio_seconds, 3),
            "processing_seconds": round(elapsed, 3),
        }
