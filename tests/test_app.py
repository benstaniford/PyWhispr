from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pywhispr.app import PUSH_TO_TALK_HOLD_SECONDS, PyWhisprApp, State
from pywhispr.config import Config
from pywhispr.vocab import parse_vocabulary


@pytest.fixture
def app(qtbot, qapp):
    backend = MagicMock()
    backend.name = "mock-backend"
    with (
        patch("pywhispr.app.create_backend", return_value=backend),
        patch("pywhispr.app.AudioRecorder") as recorder_cls,
        patch("pywhispr.app.TrayIcon"),
        # Never read the developer's own vocabulary file: tests that assert on
        # exact transcripts would then depend on what is in it.
        patch("pywhispr.app.load_vocabulary", return_value=[]),
        # Nothing here should claim a real system-wide hotkey.
        patch("pywhispr.app.create_hotkey_listener"),
    ):
        recorder = recorder_cls.return_value
        recorder.recording = False
        recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        # api_enabled=False: these tests must not open a listening socket.
        # join_continuations=False: these tests are about the state machine, so
        # transcripts should reach the injector exactly as the backend said them.
        # TestContinuationJoin below turns it back on.
        instance = PyWhisprApp(
            Config(play_sounds=False, api_enabled=False, join_continuations=False)
        )
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


class TestContinuationJoin:
    """Dictating twice in a row should read as one passage."""

    def _dictate(self, app, qtbot, text):
        app._on_model_ready()
        app._on_toggle()  # start
        app._test_backend.transcribe.return_value = text
        with patch.object(app.injector, "insert") as insert:
            app._on_toggle()  # stop → transcribe
            wait_for_worker(app, qtbot)
        return insert

    def test_joins_onto_the_caret_context(self, app, qtbot):
        app.cfg.join_continuations = True
        with patch.object(app._context, "preceding_text", return_value="I went to the shop"):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with(" then I came home.")

    def test_full_stop_keeps_the_capital(self, app, qtbot):
        app.cfg.join_continuations = True
        with patch.object(app._context, "preceding_text", return_value="I went to the shop."):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with(" Then I came home.")

    def test_no_context_inserts_verbatim(self, app, qtbot):
        app.cfg.join_continuations = True
        with patch.object(app._context, "preceding_text", return_value=None):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with("Then I came home.")

    def test_a_broken_join_never_loses_the_transcript(self, app, qtbot):
        """The audio is gone by now, so a failure here must still paste the text
        and must still return the app to IDLE — a stuck INSERTING state would
        ignore every subsequent hotkey."""
        app.cfg.join_continuations = True
        with patch.object(
            app._context, "preceding_text", side_effect=RuntimeError("accessibility exploded")
        ):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with("Then I came home.")
        assert app.state == State.INSERTING
        app.injector.finished.emit(True)
        assert app.state == State.IDLE

    def test_output_violating_the_contract_is_rejected(self, app, qtbot):
        app.cfg.join_continuations = True
        with (
            patch.object(app._context, "preceding_text", return_value="context"),
            patch("pywhispr.app.join_text", return_value="something else entirely"),
        ):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with("Then I came home.")

    def test_pasted_text_is_remembered_but_clipboard_only_is_not(self, app):
        with patch.object(app._context, "remember") as remember:
            app._last_inserted = " then I came home."
            app._on_insert_finished(True)
        remember.assert_called_once_with(" then I came home.")

        with patch.object(app._context, "invalidate") as invalidate:
            app._last_inserted = " then I came home."
            app._on_insert_finished(False)
        invalidate.assert_called_once()

    def test_failed_transcription_invalidates_context(self, app):
        with patch.object(app._context, "invalidate") as invalidate:
            app._on_transcribe_failed("boom")
        invalidate.assert_called_once()


class TestVocabulary:
    """Custom terms are corrected before the transcript is joined and pasted."""

    def _dictate(self, app, qtbot, text):
        app._on_model_ready()
        app._on_toggle()  # start
        app._test_backend.transcribe.return_value = text
        with patch.object(app.injector, "insert") as insert:
            app._on_toggle()  # stop → transcribe
            wait_for_worker(app, qtbot)
        return insert

    def test_corrects_the_transcript(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at BeyondTrust.")

    def test_disabled_by_config(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        app.cfg.vocabulary_enabled = False
        insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at beyond trust.")

    def test_correction_happens_before_the_join(self, app, qtbot):
        """The join decides about the first word, so it must see the fixed one."""
        app.cfg.join_continuations = True
        app._vocab = parse_vocabulary("BeyondTrust")
        with patch.object(app._context, "preceding_text", return_value="I work at"):
            insert = self._dictate(app, qtbot, "Beyond trust, mostly.")
        insert.assert_called_once_with(" BeyondTrust, mostly.")

    def test_a_broken_vocabulary_never_loses_the_transcript(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        with patch("pywhispr.app.apply_vocabulary", side_effect=RuntimeError("boom")):
            insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at beyond trust.")
        assert app.state == State.INSERTING

    def test_output_that_ran_away_is_rejected(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        with patch("pywhispr.app.apply_vocabulary", return_value="x"):
            insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at beyond trust.")

    def test_the_api_gets_corrections_too(self, app):
        app._vocab = parse_vocabulary("BeyondTrust")
        app._test_backend.transcribe.return_value = "hello from beyond trust"
        audio = np.zeros(16000, dtype=np.float32)
        assert app._api_transcribe(audio) == "hello from BeyondTrust"

    def test_editing_reloads_without_a_restart(self, app, tmp_path, qtbot):
        path = tmp_path / "vocabulary.txt"
        app._on_model_ready()
        with (
            patch("pywhispr.ui.vocab_dialog.VocabularyDialog.edit", return_value="BeyondTrust\n"),
            patch("pywhispr.vocab.vocabulary_path", return_value=path),
        ):
            app._edit_vocabulary()
        assert path.read_text(encoding="utf-8") == "BeyondTrust\n"
        insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at BeyondTrust.")

    def test_cancelling_changes_nothing(self, app, tmp_path):
        path = tmp_path / "vocabulary.txt"
        app._on_model_ready()
        app._vocab = parse_vocabulary("BeyondTrust")
        with (
            patch("pywhispr.ui.vocab_dialog.VocabularyDialog.edit", return_value=None),
            patch("pywhispr.vocab.vocabulary_path", return_value=path),
        ):
            app._edit_vocabulary()
        assert not path.exists()
        assert [rule.wanted for rule in app._vocab] == ["BeyondTrust"]

    def test_the_listener_is_restarted_even_if_saving_fails(self, app):
        app._on_model_ready()
        with (
            patch("pywhispr.ui.vocab_dialog.VocabularyDialog.edit", return_value="BeyondTrust"),
            patch("pywhispr.vocab.save_vocabulary_text", side_effect=OSError("read-only")),
        ):
            app._edit_vocabulary()
        app.listener.stop.assert_called_once()
        app.listener.start.assert_called_once()
        app.tray.notify.assert_called_once()

    def test_ignored_while_a_dictation_is_in_flight(self, app):
        app.state = State.TRANSCRIBING
        with patch("pywhispr.ui.vocab_dialog.VocabularyDialog.edit") as edit:
            app._edit_vocabulary()
        edit.assert_not_called()


class TestModelLoadFailure:
    """A failed load must leave a running, complaining app — not a vanished one."""

    def test_does_not_quit(self, app):
        with patch("pywhispr.app.QApplication.quit") as quit_:
            app._on_model_failed("RuntimeError: no CUDA")
            quit_.assert_not_called()
        assert app.state == State.LOADING
        assert app._model_error == "RuntimeError: no CUDA"

    def test_tray_and_overlay_report_it(self, app):
        app._on_model_failed("RuntimeError: no CUDA")
        app.tray.notify.assert_called_once()
        with patch.object(app.overlay, "show_status") as show_status:
            app._on_toggle()
        show_status.assert_called_once_with("Model failed — see log")

    def test_still_loading_shows_the_loading_message(self, app):
        with patch.object(app.overlay, "show_status") as show_status:
            app._on_toggle()
        show_status.assert_called_once_with("Loading model…")


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


class TestNetworkApi:
    def test_disabled_by_config(self, app):
        assert app.api is None

    def test_enabled_by_default(self, qtbot, qapp):
        backend = MagicMock()
        backend.name = "mock-backend"
        with (
            patch("pywhispr.app.create_backend", return_value=backend),
            patch("pywhispr.app.AudioRecorder"),
            patch("pywhispr.app.TrayIcon"),
            patch("pywhispr.app.load_vocabulary", return_value=[]),
        ):
            instance = PyWhisprApp(Config(play_sounds=False, api_host="127.0.0.1", api_port=0))
        try:
            assert instance.api is not None
            assert instance.api.start()
            assert instance._api_status()["status"] == "loading"
            instance._on_model_ready()
            assert instance._api_status()["status"] == "ready"

            backend.transcribe.return_value = "remote text"
            audio = np.zeros(16000, dtype=np.float32)
            assert instance._api_transcribe(audio) == "remote text"
            backend.transcribe.assert_called_once()
        finally:
            instance.api.stop()
            instance._worker.shutdown(wait=True)

    def test_status_reports_model_failure(self, app):
        app._model_error = "download failed"
        assert app._api_status()["status"] == "error"


def test_max_duration_stops_recording(app, qtbot):
    app._test_backend.transcribe.return_value = "long dictation"
    app._on_model_ready()
    app._on_toggle()
    assert app.state == State.RECORDING
    app._on_max_duration()
    assert app.state == State.TRANSCRIBING
    wait_for_worker(app, qtbot)
