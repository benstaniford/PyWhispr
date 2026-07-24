from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pywhispr.app import PUSH_TO_TALK_HOLD_SECONDS, PyWhisprApp, State
from pywhispr.config import Config


@pytest.fixture
def app(qtbot, qapp):
    backend = MagicMock()
    backend.name = "mock-backend"
    with (
        patch("pywhispr.app.create_backend", return_value=backend),
        patch("pywhispr.app.AudioRecorder") as recorder_cls,
        patch("pywhispr.app.TrayIcon"),
    ):
        recorder = recorder_cls.return_value
        recorder.recording = False
        recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        instance = PyWhisprApp(Config(play_sounds=False))
        instance._test_backend = backend
        instance._test_recorder = recorder
        yield instance
    instance._worker.shutdown(wait=True)


def wait_for_worker(app, qtbot):
    """Block until the single worker thread has drained its queue and Qt
    delivered the resulting queued signals."""
    app._worker.submit(lambda: None).result()
    qtbot.wait(20)


def test_starts_in_loading_and_ignores_nothing_burger(app):
    assert app.state == State.LOADING


class TestPushToTalk:
    def _ready(self, app):
        app._on_model_ready()
        assert app.state == State.IDLE

    def test_held_release_stops_recording(self, app):
        self._ready(app)
        app._on_activate()  # double-tap start
        assert app.state == State.RECORDING
        app._on_activation_key_released(PUSH_TO_TALK_HOLD_SECONDS + 0.2)  # held → stop
        assert app.state == State.TRANSCRIBING

    def test_quick_release_leaves_recording_latched(self, app):
        self._ready(app)
        app._on_activate()
        app._on_activation_key_released(0.05)  # quick tap → stay recording
        assert app.state == State.RECORDING

    def test_release_after_stop_activation_does_not_restart(self, app):
        # Double-tapping to STOP a latched recording, then holding, must not
        # start a new recording on release.
        self._ready(app)
        app._on_activate()  # start (latched)
        app._on_activation_key_released(0.05)
        assert app.state == State.RECORDING
        app._on_activate()  # second double-tap: stop
        assert app.state == State.TRANSCRIBING
        app._on_activation_key_released(PUSH_TO_TALK_HOLD_SECONDS + 0.5)
        assert app.state == State.TRANSCRIBING  # not restarted

    def test_release_after_external_stop_is_noop(self, app):
        # Max-duration guard stops the recording before the key is released.
        self._ready(app)
        app._on_activate()
        app._on_max_duration()
        assert app.state == State.TRANSCRIBING
        app._on_activation_key_released(PUSH_TO_TALK_HOLD_SECONDS + 1.0)
        assert app.state == State.TRANSCRIBING  # release ignored, no restart


def test_full_cycle(app, qtbot):
    app._test_backend.transcribe.return_value = "hello world"
    app._on_model_ready()
    assert app.state == State.IDLE

    app._on_toggle()  # start recording
    assert app.state == State.RECORDING
    app._test_recorder.start.assert_called_once()

    with patch.object(app.injector, "insert") as insert:
        app._on_toggle()  # stop → transcribe on worker
        assert app.state == State.TRANSCRIBING
        wait_for_worker(app, qtbot)
        assert app.state == State.INSERTING
        insert.assert_called_once_with("hello world")

    app.injector.finished.emit(True)
    assert app.state == State.IDLE


def test_toggle_ignored_while_transcribing(app):
    app.state = State.TRANSCRIBING
    app._on_toggle()
    assert app.state == State.TRANSCRIBING
    app._test_recorder.start.assert_not_called()


def test_empty_transcription_skips_insertion(app, qtbot):
    app._test_backend.transcribe.return_value = "   "
    app._on_model_ready()
    app._on_toggle()
    with patch.object(app.injector, "insert") as insert:
        app._on_toggle()
        wait_for_worker(app, qtbot)
        insert.assert_not_called()
    assert app.state == State.IDLE


def test_mic_error_stays_idle(app):
    app._on_model_ready()
    app._test_recorder.start.side_effect = RuntimeError("no mic")
    app._on_toggle()
    assert app.state == State.IDLE


def test_transcription_error_recovers_to_idle(app, qtbot):
    app._test_backend.transcribe.side_effect = RuntimeError("boom")
    app._on_model_ready()
    app._on_toggle()
    app._on_toggle()
    wait_for_worker(app, qtbot)
    assert app.state == State.IDLE


def test_max_duration_stops_recording(app, qtbot):
    app._test_backend.transcribe.return_value = "long dictation"
    app._on_model_ready()
    app._on_toggle()
    assert app.state == State.RECORDING
    app._on_max_duration()
    assert app.state == State.TRANSCRIBING
    wait_for_worker(app, qtbot)
