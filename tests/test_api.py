import io
import json
import threading
import wave
from http.client import HTTPConnection
from pathlib import Path

import numpy as np
import pytest

from pywhispr.api import TranscriptionServer

FIXTURE = Path(__file__).parent / "fixtures" / "hello_world.wav"


class Recorder:
    """Stand-in for the app's transcribe callable; records what it was given."""

    def __init__(self, text="hello world"):
        self.text = text
        self.audio = None
        self.gate = None  # set to an Event to make transcription block

    def __call__(self, audio):
        self.audio = audio
        if self.gate is not None:
            self.gate.wait(timeout=5)
        return self.text


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def status():
    return {"status": "ready", "backend": "mock-backend"}


@pytest.fixture
def server(recorder, status):
    """A server on an ephemeral loopback port, so tests never collide."""
    srv = TranscriptionServer(
        transcribe=recorder,
        status=lambda: status,
        host="127.0.0.1",
        port=0,
        max_audio_seconds=5,
    )
    assert srv.start()
    yield srv
    srv.stop()


def request(server, method, path, body=None, content_type=None, headers=None):
    conn = HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        all_headers = dict(headers or {})
        if content_type:
            all_headers["Content-Type"] = content_type
        conn.request(method, path, body=body, headers=all_headers)
        resp = conn.getresponse()
        raw = resp.read()
        payload = json.loads(raw) if raw else None
        return resp.status, payload, resp
    finally:
        conn.close()


def wav_bytes(audio, rate=16000, channels=1, width=2):
    """Encode float audio in [-1, 1] as a wav file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        if width == 2:
            wf.writeframes((np.asarray(audio) * 32767).astype("<i2").tobytes())
        else:
            wf.writeframes(np.asarray(audio, dtype="<f4").tobytes())
    return buf.getvalue()


class TestHealth:
    def test_reports_ready(self, server):
        status, body, _ = request(server, "GET", "/v1/health")
        assert status == 200
        assert body["status"] == "ready"
        assert body["backend"] == "mock-backend"
        assert body["sample_rate"] == 16000
        assert body["max_audio_seconds"] == 5

    def test_root_lists_endpoints(self, server):
        status, body, _ = request(server, "GET", "/")
        assert status == 200
        assert "POST /v1/transcribe" in body["endpoints"]

    def test_unknown_path_is_404(self, server):
        status, body, _ = request(server, "GET", "/nope")
        assert status == 404
        assert body["error"]["code"] == "not_found"


class TestTranscribeWav:
    def test_round_trip(self, server, recorder):
        status, body, _ = request(
            server, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "audio/wav"
        )
        assert status == 200
        assert body["text"] == "hello world"
        assert body["backend"] == "mock-backend"
        assert body["audio_seconds"] > 0

    def test_backend_gets_mono_float32_16k(self, server, recorder):
        # Stereo 44.1 kHz in; the backend must still see mono float32 at 16 kHz.
        stereo = np.zeros(44100 * 2, dtype=np.float32)
        stereo[::2] = 0.5
        request(server, "POST", "/v1/transcribe", wav_bytes(stereo, 44100, channels=2), "audio/wav")
        assert recorder.audio.dtype == np.float32
        assert recorder.audio.ndim == 1
        assert recorder.audio.shape[0] == pytest.approx(16000, rel=0.01)

    def test_float32_wav_is_accepted(self, server, recorder):
        tone = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
        status, _, _ = request(
            server, "POST", "/v1/transcribe", wav_bytes(tone, width=4), "audio/wav"
        )
        assert status == 200
        assert recorder.audio.shape[0] == 16000

    def test_wav_sent_as_octet_stream_is_sniffed(self, server):
        status, _, _ = request(
            server, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "application/octet-stream"
        )
        assert status == 200

    def test_malformed_wav_is_400(self, server):
        status, body, _ = request(server, "POST", "/v1/transcribe", b"RIFFgarbage", "audio/wav")
        assert status == 400
        assert body["error"]["code"] == "bad_audio"


class TestTranscribePcm:
    def test_f32le(self, server, recorder):
        pcm = np.full(16000, 0.25, dtype="<f4").tobytes()
        status, _, _ = request(server, "POST", "/v1/transcribe", pcm, "application/octet-stream")
        assert status == 200
        assert recorder.audio.shape[0] == 16000
        assert recorder.audio[0] == pytest.approx(0.25)

    def test_s16le_with_resample(self, server, recorder):
        pcm = np.full(44100, 8192, dtype="<i2").tobytes()
        status, _, _ = request(
            server,
            "POST",
            "/v1/transcribe?sample_rate=44100&format=s16le",
            pcm,
            "application/octet-stream",
        )
        assert status == 200
        assert recorder.audio.shape[0] == pytest.approx(16000, rel=0.01)
        assert recorder.audio[0] == pytest.approx(0.25, abs=1e-3)

    def test_bad_format_is_400(self, server):
        status, body, _ = request(
            server,
            "POST",
            "/v1/transcribe?format=mp3",
            b"\x00" * 16,
            "application/octet-stream",
        )
        assert status == 400
        assert body["error"]["code"] == "bad_audio"

    def test_empty_body_is_400(self, server):
        status, body, _ = request(server, "POST", "/v1/transcribe", b"", "audio/wav")
        assert status == 400
        assert body["error"]["code"] == "bad_audio"


class TestTranscribeMultipart:
    def test_upload(self, server, recorder):
        boundary = "----pywhisprtest"
        body = (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="audio"; filename="clip.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + FIXTURE.read_bytes()
            + f"\r\n--{boundary}--\r\n".encode()
        )

        status, payload, _ = request(
            server, "POST", "/v1/transcribe", body, f"multipart/form-data; boundary={boundary}"
        )
        assert status == 200
        assert payload["text"] == "hello world"
        assert recorder.audio is not None

    def test_missing_audio_field_is_400(self, server):
        boundary = "----pywhisprtest"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="notes"\r\n\r\nhi\r\n'
            f"--{boundary}--\r\n"
        ).encode()
        status, payload, _ = request(
            server, "POST", "/v1/transcribe", body, f"multipart/form-data; boundary={boundary}"
        )
        assert status == 400
        assert payload["error"]["code"] == "bad_audio"


class TestLimitsAndErrors:
    def test_unsupported_content_type_is_415(self, server):
        status, body, _ = request(server, "POST", "/v1/transcribe", b"{}", "application/json")
        assert status == 415
        assert body["error"]["code"] == "unsupported_media_type"

    def test_oversize_content_length_is_413(self, server):
        conn = HTTPConnection("127.0.0.1", server.port, timeout=10)
        try:
            conn.putrequest("POST", "/v1/transcribe")
            conn.putheader("Content-Type", "audio/wav")
            conn.putheader("Content-Length", str(server.max_upload_bytes + 1))
            conn.endheaders()
            resp = conn.getresponse()
            assert resp.status == 413
            assert json.loads(resp.read())["error"]["code"] == "payload_too_large"
        finally:
            conn.close()

    def test_audio_longer_than_limit_is_413(self, server):
        # 10 s of float32 at 16 kHz: under the byte cap, over max_audio_seconds.
        pcm = np.zeros(16000 * 10, dtype="<f4").tobytes()
        status, body, _ = request(server, "POST", "/v1/transcribe", pcm, "application/octet-stream")
        assert status == 413
        assert body["error"]["code"] == "payload_too_large"

    def test_missing_content_length_is_411(self, server):
        conn = HTTPConnection("127.0.0.1", server.port, timeout=10)
        try:
            conn.putrequest("POST", "/v1/transcribe", skip_accept_encoding=True)
            conn.putheader("Content-Type", "audio/wav")
            conn.putheader("Transfer-Encoding", "chunked")
            conn.endheaders()
            conn.send(b"0\r\n\r\n")
            resp = conn.getresponse()
            assert resp.status == 411
            assert json.loads(resp.read())["error"]["code"] == "length_required"
        finally:
            conn.close()

    def test_backend_failure_is_500(self, server, recorder):
        def boom(audio):
            raise RuntimeError("model exploded")

        server._transcribe = boom
        status, body, _ = request(
            server, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "audio/wav"
        )
        assert status == 500
        assert body["error"]["code"] == "transcribe_failed"


class TestReadiness:
    def test_rejects_while_loading(self, server, status):
        status["status"] = "loading"
        code, body, resp = request(
            server, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "audio/wav"
        )
        assert code == 503
        assert body["error"]["code"] == "model_loading"
        assert resp.getheader("Retry-After") == "5"

    def test_rejects_when_model_failed(self, server, status):
        status["status"] = "error"
        code, body, _ = request(server, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "audio/wav")
        assert code == 503
        assert body["error"]["code"] == "model_unavailable"


class TestConcurrency:
    def test_queue_limit_returns_busy(self, recorder, status):
        srv = TranscriptionServer(
            transcribe=recorder,
            status=lambda: status,
            host="127.0.0.1",
            port=0,
            max_queue=1,
        )
        assert srv.start()
        recorder.gate = threading.Event()
        result = {}
        blocker = threading.Thread(
            target=lambda: result.update(
                code=request(srv, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "audio/wav")[0]
            )
        )
        try:
            blocker.start()
            for _ in range(200):  # wait for the first request to occupy the slot
                if srv._inflight == 1:
                    break
                threading.Event().wait(0.01)
            assert srv._inflight == 1

            code, body, resp = request(
                srv, "POST", "/v1/transcribe", FIXTURE.read_bytes(), "audio/wav"
            )
            assert code == 503
            assert body["error"]["code"] == "busy"
            assert resp.getheader("Retry-After") == "2"
        finally:
            recorder.gate.set()
            blocker.join(timeout=10)
            srv.stop()
        assert result["code"] == 200


class TestCors:
    def test_preflight(self, server):
        _, _, resp = request(server, "OPTIONS", "/v1/transcribe")
        assert resp.status == 204
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.getheader("Access-Control-Allow-Methods")
        assert "Content-Type" in resp.getheader("Access-Control-Allow-Headers")

    def test_responses_are_cross_origin_readable(self, server):
        _, _, resp = request(server, "GET", "/v1/health")
        assert resp.getheader("Access-Control-Allow-Origin") == "*"


class TestLifecycle:
    def test_start_returns_false_when_port_is_taken(self, server, recorder, status):
        clash = TranscriptionServer(
            transcribe=recorder, status=lambda: status, host="127.0.0.1", port=server.port
        )
        assert clash.start() is False
        clash.stop()  # must be a no-op, not an error

    def test_stop_is_idempotent(self, server):
        server.stop()
        server.stop()
