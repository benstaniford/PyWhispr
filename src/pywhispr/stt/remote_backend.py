"""Speech-to-text over the network, for machines that cannot run the model.

This is the client half of :mod:`pywhispr.api` (the server the full app hosts):
audio in, text out, over ``POST /v1/transcribe``. It satisfies the same
:class:`~pywhispr.stt.base.STTBackend` interface as the local backends, so the
whole record → transcribe → paste pipeline in ``app.py`` is unchanged — only the
transcription itself happens on another host.

Stdlib ``urllib`` on purpose, matching ``api.py``'s no-httpx/no-fastapi stance:
one POST does not justify a dependency, and every extra package is another
hidden-import to chase in the PyInstaller build. ``certs.use_system_certificates``
runs at startup, so ``https://`` servers behind corporate TLS inspection verify
against the OS trust store like everything else here.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE, STTBackend

log = logging.getLogger(__name__)

# Generous: a remote request may queue behind a local dictation still recording
# on the server, which allows up to QUEUE_TIMEOUT_SECONDS (120) before it even
# starts, plus the clip's own processing time.
REQUEST_TIMEOUT_SECONDS = 180

# Shorter: a health check is a cheap GET and should fail fast rather than hang
# startup when the server is unreachable.
HEALTH_TIMEOUT_SECONDS = 10


def _normalise(server_url: str) -> str:
    """Turn what the user typed into a usable base URL (scheme, no trailing /)."""
    url = server_url.strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    return url


class RemoteBackend(STTBackend):
    """Transcribe by POSTing audio to a PyWhispr (or compatible) server."""

    def __init__(self, server_url: str):
        self._server_url = _normalise(server_url or "")
        self._server_backend: str | None = None  # what the far end reported, for logs

    @property
    def name(self) -> str:
        return f"remote ({self._server_url or 'no server set'})"

    def load(self) -> None:
        """Check the server is reachable and note its backend. Never fatal.

        A menu-bar app whose main job is dictation must not refuse to start
        because a server is down — it may come up later, or the user may repoint
        it from the tray. So a failure here is logged and swallowed; the real
        error surfaces per-request from :meth:`transcribe`, where it can reach the
        overlay and the tray.
        """
        if not self._server_url:
            log.warning("Remote backend has no server URL configured yet")
            return
        try:
            with urllib.request.urlopen(
                f"{self._server_url}/v1/health", timeout=HEALTH_TIMEOUT_SECONDS
            ) as response:
                health = json.loads(response.read().decode())
            self._server_backend = health.get("backend")
            log.info(
                "Remote server %s: status=%s backend=%s",
                self._server_url,
                health.get("status"),
                self._server_backend,
            )
        except Exception as exc:
            # Debug-level detail, info-level fact: an unreachable server at startup
            # is an everyday case here, not a bug.
            log.info("Could not reach remote server %s: %s", self._server_url, exc)
            log.debug("Health check failed", exc_info=True)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if not self._server_url:
            raise RuntimeError("No server is set — see the tray menu's “Settings…”.")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")

        # Send the audio exactly as the pipeline holds it — mono float32 at 16 kHz —
        # as headerless little-endian PCM, which the server decodes without a codec.
        body = np.ascontiguousarray(audio, dtype="<f4").tobytes()
        url = (
            f"{self._server_url}/v1/transcribe"
            f"?format=f32le&sample_rate={SAMPLE_RATE}&channels=1"
        )
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        log.debug("Sending %d samples (%d bytes) to %s", len(audio), len(body), self._server_url)

        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._describe_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Can't reach the server at {self._server_url}: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Transcription request to {self._server_url} failed: {exc}") from exc

        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError("The server's reply had no transcript text")
        return text.strip()

    def _describe_http_error(self, exc: urllib.error.HTTPError) -> str:
        """Turn the server's JSON error into a message worth showing the user.

        ``api.py`` sends ``{"error": {"code", "message"}}`` with stable codes; the
        two the user can actually act on get a plainer sentence, the rest fall back
        to the server's own message.
        """
        code = message = None
        try:
            body = json.loads(exc.read().decode())
            error = body.get("error", {})
            code, message = error.get("code"), error.get("message")
        except Exception:
            log.debug("Could not parse the server error body", exc_info=True)

        if code == "model_loading":
            return "The server's model is still loading — try again in a moment."
        if code == "busy":
            return "The server is busy — try again in a moment."
        detail = message or exc.reason or "unknown error"
        return f"The server returned an error ({exc.code}): {detail}"
