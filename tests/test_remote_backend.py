import numpy as np
import pytest

from pywhispr.api import TranscriptionServer
from pywhispr.stt.remote_backend import RemoteBackend, _normalise


class Recorder:
    """Stand-in for the server's transcribe callable; records the audio it saw."""

    def __init__(self, text="hello world"):
        self.text = text
        self.audio = None

    def __call__(self, audio):
        self.audio = audio
        return self.text


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def status():
    return {"status": "ready", "backend": "mock-backend"}


@pytest.fixture
def server(recorder, status):
    """A real server on an ephemeral loopback port; the backend talks to it."""
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


@pytest.fixture
def backend(server):
    return RemoteBackend(f"http://127.0.0.1:{server.port}")


class TestNormalise:
    def test_adds_scheme_and_strips_trailing_slash(self):
        assert _normalise("192.168.1.5:9149") == "http://192.168.1.5:9149"
        assert _normalise("http://host:9149/") == "http://host:9149"

    def test_keeps_https(self):
        assert _normalise("https://host/") == "https://host"


class TestLoad:
    def test_reads_health_and_backend(self, backend):
        backend.load()
        assert backend._server_backend == "mock-backend"

    def test_unreachable_server_does_not_raise(self):
        # Nothing is listening here: load() must swallow it and stay usable.
        RemoteBackend("http://127.0.0.1:1").load()

    def test_no_server_url_does_not_raise(self):
        RemoteBackend("").load()


class TestTranscribe:
    def test_round_trips_audio_to_text(self, backend, recorder):
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        assert backend.transcribe(audio) == "hello world"
        # The server received the same samples we sent (allowing float tolerance).
        assert recorder.audio is not None
        assert len(recorder.audio) == len(audio)
        np.testing.assert_allclose(recorder.audio, audio, atol=1e-6)

    def test_strips_whitespace(self, server, recorder):
        recorder.text = "  spaced out  "
        backend = RemoteBackend(f"http://127.0.0.1:{server.port}")
        assert backend.transcribe(np.zeros(16000, dtype=np.float32)) == "spaced out"

    def test_no_server_set_is_a_clear_error(self):
        with pytest.raises(RuntimeError, match="No server is set"):
            RemoteBackend("").transcribe(np.zeros(16000, dtype=np.float32))

    def test_unreachable_server_raises_clear_error(self):
        backend = RemoteBackend("http://127.0.0.1:1")
        with pytest.raises(RuntimeError, match="Can't reach the server"):
            backend.transcribe(np.zeros(16000, dtype=np.float32))

    def test_model_loading_gets_a_friendly_message(self, server, status):
        status["status"] = "loading"
        backend = RemoteBackend(f"http://127.0.0.1:{server.port}")
        with pytest.raises(RuntimeError, match="still loading"):
            backend.transcribe(np.zeros(16000, dtype=np.float32))

    def test_wrong_sample_rate_is_rejected(self, backend):
        with pytest.raises(ValueError, match="16000 Hz"):
            backend.transcribe(np.zeros(16000, dtype=np.float32), sample_rate=44100)
