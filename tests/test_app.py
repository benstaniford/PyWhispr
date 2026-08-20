import contextlib
import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pywhispr.app import PUSH_TO_TALK_HOLD_SECONDS, PyWhisprApp, State
from pywhispr.config import Config
from pywhispr.numbers import NumberResult, Replacement
from pywhispr.plugins.api import Trigger
from pywhispr.plugins.engine import Plugin, compile_patterns
from pywhispr.scratch import compile_reset_phrases
from pywhispr.vocab import parse_vocabulary


@contextlib.contextmanager
def isolated_app(**config_kwargs):
    """A PyWhisprApp that touches nothing real. Yields (app, tray class mock).

    Its own context manager as well as a fixture, because a test that is about what
    the constructor does — which tray entries it asks for, say — needs to build one
    itself under its own patches.
    """
    backend = MagicMock()
    backend.name = "mock-backend"
    with (
        patch("pywhispr.app.create_backend", return_value=backend),
        patch("pywhispr.app.AudioRecorder") as recorder_cls,
        patch("pywhispr.app.TrayIcon") as tray_cls,
        # Never read the developer's own vocabulary file: tests that assert on
        # exact transcripts would then depend on what is in it.
        patch("pywhispr.app.load_vocabulary", return_value=[]),
        # Nor their plugins folder, for the same reason — and a plugin of theirs
        # with an act() would otherwise really run during the suite.
        patch("pywhispr.app.load_plugins", return_value=[]),
        # Nothing here should claim a real system-wide hotkey. A fresh mock per
        # call, so the dictation and recall listeners are told apart.
        patch("pywhispr.app.create_hotkey_listener", side_effect=lambda *a, **k: MagicMock()),
    ):
        recorder = recorder_cls.return_value
        recorder.recording = False
        recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        # api_enabled=False: these tests must not open a listening socket.
        # join_continuations=False: these tests are about the state machine, so
        # transcripts should reach the injector exactly as the backend said them.
        # TestContinuationJoin below turns it back on.
        # offer_gpu_setup=False: readying the model must not pop the GPU offer
        # in tests that are about something else.
        defaults = dict(
            play_sounds=False,
            api_enabled=False,
            join_continuations=False,
            offer_gpu_setup=False,
        )
        instance = PyWhisprApp(Config(**{**defaults, **config_kwargs}))
        instance._test_backend = backend
        instance._test_recorder = recorder
        try:
            yield instance, tray_cls
        finally:
            instance._worker.shutdown(wait=True)


@pytest.fixture
def app(qtbot, qapp):
    with isolated_app() as (instance, _tray_cls):
        yield instance


def wait_for_worker(app, qtbot):
    """Block until the single worker thread has drained its queue and Qt
    delivered the resulting queued signals."""
    app._worker.submit(lambda: None).result()
    qtbot.wait(20)


def test_starts_in_loading_and_ignores_nothing_burger(app):
    assert app.state == State.LOADING


class TestLiteMode:
    """The Lite build hosts no API of its own; its server field is on the settings
    page (see test_settings_dialog.py)."""

    def test_hosts_no_api(self, qapp, monkeypatch):
        monkeypatch.setattr("pywhispr.flavor.IS_LITE", True)
        with isolated_app(api_enabled=True) as (instance, _tray_cls):
            assert instance.api is None  # a client does not also host a server

    def test_a_new_server_url_rebuilds_the_backend(self, qapp, monkeypatch):
        monkeypatch.setattr("pywhispr.flavor.IS_LITE", True)
        with isolated_app() as (instance, _tray_cls):
            edited = dataclasses.replace(instance.cfg, server_url="http://elsewhere:9149")
            with (
                patch("pywhispr.app.save_config"),
                patch.object(instance, "_apply_server_url") as applied,
            ):
                instance._apply_settings(edited)
            applied.assert_called_once_with("http://elsewhere:9149")


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
        insert.assert_called_once_with("hello world", ())

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
        insert.assert_called_once_with(" then I came home.", ())

    def test_full_stop_keeps_the_capital(self, app, qtbot):
        app.cfg.join_continuations = True
        with patch.object(app._context, "preceding_text", return_value="I went to the shop."):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with(" Then I came home.", ())

    def test_no_context_inserts_verbatim(self, app, qtbot):
        app.cfg.join_continuations = True
        with patch.object(app._context, "preceding_text", return_value=None):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with("Then I came home.", ())

    def test_a_broken_join_never_loses_the_transcript(self, app, qtbot):
        """The audio is gone by now, so a failure here must still paste the text
        and must still return the app to IDLE — a stuck INSERTING state would
        ignore every subsequent hotkey."""
        app.cfg.join_continuations = True
        with patch.object(
            app._context, "preceding_text", side_effect=RuntimeError("accessibility exploded")
        ):
            insert = self._dictate(app, qtbot, "Then I came home.")
        insert.assert_called_once_with("Then I came home.", ())
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
        insert.assert_called_once_with("Then I came home.", ())

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
        insert.assert_called_once_with("I work at BeyondTrust.", ())

    def test_disabled_by_config(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        app.cfg.vocabulary_enabled = False
        insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at beyond trust.", ())

    def test_correction_happens_before_the_join(self, app, qtbot):
        """The join decides about the first word, so it must see the fixed one."""
        app.cfg.join_continuations = True
        app._vocab = parse_vocabulary("BeyondTrust")
        with patch.object(app._context, "preceding_text", return_value="I work at"):
            insert = self._dictate(app, qtbot, "Beyond trust, mostly.")
        insert.assert_called_once_with(" BeyondTrust, mostly.", ())

    def test_a_broken_vocabulary_never_loses_the_transcript(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        with patch("pywhispr.app.apply_vocabulary", side_effect=RuntimeError("boom")):
            insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at beyond trust.", ())
        assert app.state == State.INSERTING

    def test_output_that_ran_away_is_rejected(self, app, qtbot):
        app._vocab = parse_vocabulary("BeyondTrust")
        with patch("pywhispr.app.apply_vocabulary", return_value="x"):
            insert = self._dictate(app, qtbot, "I work at beyond trust.")
        insert.assert_called_once_with("I work at beyond trust.", ())

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
        insert.assert_called_once_with("I work at BeyondTrust.", ())

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

    def test_a_failed_save_says_so_rather_than_raising(self, app):
        """The editor is opened from inside the settings visit, which owns the
        listener — so this only has to survive and report."""
        app._on_model_ready()
        with (
            patch("pywhispr.ui.vocab_dialog.VocabularyDialog.edit", return_value="BeyondTrust"),
            patch("pywhispr.vocab.save_vocabulary_text", side_effect=OSError("read-only")),
        ):
            app._edit_vocabulary()
        app.tray.notify.assert_called_once()


class TestFillerRemoval:
    """"Um"s and "uh"s are gone before anything else sees the transcript."""

    def _dictate(self, app, qtbot, text):
        app._on_model_ready()
        app._on_toggle()  # start
        app._test_backend.transcribe.return_value = text
        with patch.object(app.injector, "insert") as insert:
            app._on_toggle()  # stop → transcribe
            wait_for_worker(app, qtbot)
        return insert

    def test_removes_fillers(self, app, qtbot):
        insert = self._dictate(app, qtbot, "Um, so I, uh, think so.")
        insert.assert_called_once_with("So I think so.", ())

    def test_disabled_by_config(self, app, qtbot):
        app.cfg.remove_fillers = False
        insert = self._dictate(app, qtbot, "Um, so I, uh, think so.")
        insert.assert_called_once_with("Um, so I, uh, think so.", ())

    def test_a_broken_pass_never_loses_the_transcript(self, app, qtbot):
        with patch("pywhispr.app.remove_fillers", side_effect=RuntimeError("boom")):
            insert = self._dictate(app, qtbot, "Um, so I think so.")
        insert.assert_called_once_with("Um, so I think so.", ())
        assert app.state == State.INSERTING

    def test_output_that_is_not_a_deletion_is_rejected(self, app, qtbot):
        with patch("pywhispr.app.remove_fillers", return_value="Something else entirely."):
            insert = self._dictate(app, qtbot, "Um, so I think so.")
        insert.assert_called_once_with("Um, so I think so.", ())

    def test_the_api_gets_it_too(self, app):
        app._test_backend.transcribe.return_value = "Um, hello from over there."
        audio = np.zeros(16000, dtype=np.float32)
        assert app._api_transcribe(audio) == "Hello from over there."


class TestSpokenNumbers:
    """Numbers said as words reach the paste as digits."""

    def _dictate(self, app, qtbot, text):
        app._on_model_ready()
        app._on_toggle()  # start
        app._test_backend.transcribe.return_value = text
        with patch.object(app.injector, "insert") as insert:
            app._on_toggle()  # stop → transcribe
            wait_for_worker(app, qtbot)
        return insert

    def test_converts_a_run_of_numbers(self, app, qtbot):
        insert = self._dictate(app, qtbot, "Call me on one one eight zero.")
        insert.assert_called_once_with("Call me on 1180.", ())

    def test_a_lone_number_is_left_as_a_word(self, app, qtbot):
        insert = self._dictate(app, qtbot, "I have five apples.")
        insert.assert_called_once_with("I have five apples.", ())

    def test_disabled_by_config(self, app, qtbot):
        app.cfg.numbers_to_digits = False
        insert = self._dictate(app, qtbot, "Call me on one one eight zero.")
        insert.assert_called_once_with("Call me on one one eight zero.", ())

    def test_a_broken_pass_never_loses_the_transcript(self, app, qtbot):
        with patch("pywhispr.app.to_digits", side_effect=RuntimeError("boom")):
            insert = self._dictate(app, qtbot, "Call me on one one eight zero.")
        insert.assert_called_once_with("Call me on one one eight zero.", ())
        assert app.state == State.INSERTING

    def test_output_its_spans_do_not_account_for_is_rejected(self, app, qtbot):
        broken = NumberResult("Something else entirely.", ())
        with patch("pywhispr.app.to_digits", return_value=broken):
            insert = self._dictate(app, qtbot, "Call me on one one eight zero.")
        insert.assert_called_once_with("Call me on one one eight zero.", ())

    def test_a_span_over_something_that_is_not_a_number_is_rejected(self, app, qtbot):
        # The tripwire that matters: whatever the parser does, it can only ever
        # have replaced number words.
        broken = NumberResult("I have 5 apples.", (Replacement(7, 12, "5"),))
        with patch("pywhispr.app.to_digits", return_value=broken):
            insert = self._dictate(app, qtbot, "I have apples five.")
        insert.assert_called_once_with("I have apples five.", ())

    def test_the_vocabulary_still_sees_the_words(self, app, qtbot):
        """Why numbers run after the vocabulary: an entry may spell one out."""
        app._vocab = parse_vocabulary("s three => S3")
        insert = self._dictate(app, qtbot, "Upload it to s three.")
        insert.assert_called_once_with("Upload it to S3.", ())

    def test_the_api_gets_it_too(self, app):
        app._test_backend.transcribe.return_value = "The port is nine one four nine."
        audio = np.zeros(16000, dtype=np.float32)
        assert app._api_transcribe(audio) == "The port is 9149."


class TestPlugins:
    """Where the plugin pass sits, and when a plugin's action is allowed to run."""

    def _dictate(self, app, qtbot, text):
        app._on_model_ready()
        app._on_toggle()
        app._test_backend.transcribe.return_value = text
        with patch.object(app.injector, "insert") as insert:
            app._on_toggle()
            wait_for_worker(app, qtbot)
        return insert

    @staticmethod
    def _plugin(name="test", phrase="marker", rewrite=None, act=None):
        triggers = (Trigger(phrase=phrase),)
        return Plugin(
            name=name,
            triggers=triggers,
            rewrite=rewrite,
            act=act,
            patterns=compile_patterns(triggers),
        )

    @staticmethod
    def _shout(match):
        """Claims the trigger and the word before it, replacing both with "!"."""
        if not match.words_before:
            return None
        return match.claim_from(match.words_before[-1], "!")

    def test_a_rewrite_reaches_the_injector(self, app, qtbot):
        app._plugins = [self._plugin(rewrite=self._shout)]
        insert = self._dictate(app, qtbot, "Here we go marker.")
        insert.assert_called_once_with("Here we !.", ())

    def test_runs_after_the_vocabulary(self, app, qtbot):
        """A trigger has to see the corrected spelling, or it cannot match it."""
        app._vocab = parse_vocabulary("marker")
        app._plugins = [self._plugin(rewrite=self._shout)]
        insert = self._dictate(app, qtbot, "Here we go MARKER.")
        insert.assert_called_once_with("Here we !.", ())

    def test_runs_before_the_join(self, app, qtbot):
        """The join decides about the opening word, so it must see the final one."""
        app.cfg.join_continuations = True
        app._plugins = [
            self._plugin(rewrite=lambda match: match.claim(match.start, match.end, "and"))
        ]
        with patch.object(app._context, "preceding_text", return_value="I went out"):
            insert = self._dictate(app, qtbot, "Marker then I came home.")
        # "Marker" became "and", and the join lower-cased *that* word.
        insert.assert_called_once_with(" and then I came home.", ())

    def test_a_broken_pass_never_loses_the_transcript(self, app, qtbot):
        app._plugins = [self._plugin(rewrite=self._shout)]
        with patch("pywhispr.app.apply_plugins", side_effect=RuntimeError("boom")):
            insert = self._dictate(app, qtbot, "Here we go marker.")
        insert.assert_called_once_with("Here we go marker.", ())

    def test_an_action_waits_for_the_insertion(self, app, qtbot):
        """An action that types or switches window must not race the paste."""
        app._plugins = [self._plugin(rewrite=self._shout, act=lambda match: None)]
        app._actions = MagicMock()
        self._dictate(app, qtbot, "Here we go marker.")
        app._actions.dispatch.assert_not_called()
        assert len(app._pending_actions) == 1

        app.injector.finished.emit(True)
        app._actions.dispatch.assert_called_once()
        assert app._pending_actions == ()

    def test_an_action_runs_in_clipboard_mode_too(self, app, qtbot):
        app._plugins = [self._plugin(rewrite=self._shout, act=lambda match: None)]
        app._actions = MagicMock()
        self._dictate(app, qtbot, "Here we go marker.")
        app.injector.finished.emit(False)  # copied, not pasted
        app._actions.dispatch.assert_called_once()

    def test_a_recall_does_not_fire_the_action_again(self, app, qtbot):
        app._plugins = [self._plugin(rewrite=self._shout, act=lambda match: None)]
        app._actions = MagicMock()
        self._dictate(app, qtbot, "Here we go marker.")
        app.injector.finished.emit(True)
        app._actions.dispatch.reset_mock()

        # The history picker comes back through the same signal.
        app.injector.finished.emit(True)
        app._actions.dispatch.assert_not_called()

    def test_actions_switched_off_still_rewrites(self, app, qtbot):
        app._plugins = [self._plugin(rewrite=self._shout, act=lambda match: None)]
        app._actions = None  # what cfg.plugin_actions_enabled = false builds
        insert = self._dictate(app, qtbot, "Here we go marker.")
        insert.assert_called_once_with("Here we !.", ())
        app.injector.finished.emit(True)
        assert app._pending_actions == ()

    def test_a_command_only_dictation_inserts_nothing_but_still_acts(self, app, qtbot):
        """The whole transcript was the command, punctuation included."""
        consume = lambda match: match.claim(match.window_start, match.window_end, "")  # noqa: E731
        app._plugins = [self._plugin(phrase="new paragraph", rewrite=consume, act=lambda m: None)]
        app._actions = MagicMock()
        insert = self._dictate(app, qtbot, "New paragraph.")
        insert.assert_not_called()
        app._actions.dispatch.assert_called_once()
        assert app.state == State.IDLE
        assert list(app._history) == []  # nothing worth recalling

    def test_rich_spans_reach_the_injector(self, app, qtbot):
        def rewrite(match):
            return match.claim_from(match.words_before[-1], "frown", html="<img alt='f'>")

        app._plugins = [self._plugin(rewrite=rewrite)]
        insert = self._dictate(app, qtbot, "Here we go marker.")
        text, rich = insert.call_args.args
        assert text == "Here we frown."
        assert [text[s:e] for s, e, _ in rich] == ["frown"]

    def test_rich_spans_shift_with_the_join(self, app, qtbot):
        """join_text may prepend a space, which moves every span along by one."""

        def rewrite(match):
            return match.claim_from(match.words_before[-1], "frown", html="<img alt='f'>")

        app.cfg.join_continuations = True
        app._plugins = [self._plugin(rewrite=rewrite)]
        with patch.object(app._context, "preceding_text", return_value="I said"):
            insert = self._dictate(app, qtbot, "Here we go marker.")
        text, rich = insert.call_args.args
        assert text.startswith(" ")
        assert [text[s:e] for s, e, _ in rich] == ["frown"]

    def test_spans_are_dropped_if_the_text_moved_unexpectedly(self, app):
        """Markup over the wrong characters is worse than none at all.

        Tested directly rather than through a patched join, because _joined's own
        tripwire catches a misbehaving join_text first and returns the raw text —
        so this branch is belt-and-braces that the pipeline cannot actually reach.
        Worth keeping, and worth testing at the level where it is reachable.
        """
        app._rich_spans = ((2, 5, "<b>x</b>"),)
        assert app._shifted_rich("one two", "wholly different text") == ()

    def test_spans_are_kept_when_only_a_space_was_added(self, app):
        app._rich_spans = ((0, 3, "<b>x</b>"),)
        assert app._shifted_rich("abc def", " abc def") == ((1, 4, "<b>x</b>"),)

    def test_a_failed_transcription_runs_nothing(self, app):
        app._pending_actions = ("pretend",)
        app._on_transcribe_failed("RuntimeError: no")
        assert app._pending_actions == ()

    class TestTheNetworkApi:
        """The API port is open to the LAN with no authentication."""

        def test_gets_the_rewrites(self, app):
            app._plugins = [TestPlugins._plugin(rewrite=TestPlugins._shout)]
            app._test_backend.transcribe.return_value = "Here we go marker."
            assert app._api_transcribe(np.zeros(16000, dtype=np.float32)) == "Here we !."

        # Note: there is deliberately no test here driving _api_transcribe from a
        # real thread to prove the pass runs off the GUI thread. It livelocks the
        # suite: unittest.mock is not thread-safe, and reaching the fixture's mocks
        # from a second thread wedges a *later* test whose main thread and STT
        # worker both build child mocks. The half of that constraint worth checking
        # is that the engine is reentrant, and test_plugins.py does it with no app,
        # no Qt and no mocks in the way.

        def test_never_gets_rich_spans_either(self, app):
            """Those are GUI-thread state for one dictation cycle.

            A request thread setting them would leave markup queued against a
            transcript that this machine's user never dictated.
            """
            def rewrite(match):
                return match.claim(match.start, match.end, "X", html="<b>X</b>")

            app._plugins = [TestPlugins._plugin(rewrite=rewrite)]
            app._test_backend.transcribe.return_value = "Here we go marker."
            app._api_transcribe(np.zeros(16000, dtype=np.float32))
            assert app._rich_spans == ()

        def test_never_gets_the_actions(self, app):
            """Otherwise anything that can reach the port can run local code."""
            app._plugins = [
                TestPlugins._plugin(rewrite=TestPlugins._shout, act=lambda match: None)
            ]
            app._actions = MagicMock()
            app._test_backend.transcribe.return_value = "Here we go marker."
            app._api_transcribe(np.zeros(16000, dtype=np.float32))
            assert app._pending_actions == ()
            app._actions.dispatch.assert_not_called()


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
            instance = PyWhisprApp(
                Config(
                    play_sounds=False,
                    api_host="127.0.0.1",
                    api_port=0,
                    offer_gpu_setup=False,
                )
            )
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


class TestGpuOffer:
    """The offer only appears where it would help, and only once."""

    def _ready(self, app, providers=("CPUExecutionProvider",), driver=596.08):
        app.cfg.offer_gpu_setup = True
        with (
            patch("pywhispr.stt.onnx_backend.session_providers", return_value=set(providers)),
            patch("pywhispr.cuda.nvidia_driver_version", return_value=driver),
            patch("pywhispr.cuda.is_installed", return_value=False),
            patch("sys.platform", "win32"),
            patch("pywhispr.ui.setup_window.ask_to_enable") as ask,
            patch.object(app, "_run_gpu_setup") as run,
        ):
            app._maybe_offer_gpu()
        return ask, run

    def test_offered_when_the_gpu_is_going_unused(self, app):
        ask, run = self._ready(app)
        ask.assert_called_once()

    def test_not_offered_when_already_on_the_gpu(self, app):
        ask, _ = self._ready(app, providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
        ask.assert_not_called()

    def test_not_offered_without_an_nvidia_driver(self, app):
        with patch("pywhispr.directml.can_offer", return_value=(False, "no DirectX 12 GPU")):
            ask, _ = self._ready(app, driver=None)
        ask.assert_not_called()

    def test_directml_is_offered_when_cuda_cannot_help(self, app):
        """A pre-Turing NVIDIA card, or an AMD or Intel one, has no other option."""
        with patch("pywhispr.directml.can_offer", return_value=(True, "")):
            ask, _ = self._ready(app, driver=None)
        ask.assert_called_once()
        assert ask.call_args.kwargs["kind"] == "directml"

    def test_not_offered_once_declined_for_good(self, app):
        with (
            patch("pywhispr.cuda.can_offer", return_value=(True, "")),
            patch("pywhispr.ui.setup_window.ask_to_enable") as ask,
        ):
            app.cfg.offer_gpu_setup = False
            app._maybe_offer_gpu()
        ask.assert_not_called()

    def test_never_stops_it_being_offered_again(self, app):
        with (
            patch("pywhispr.ui.setup_window.ask_to_enable", return_value=None),
            patch("pywhispr.cuda.can_offer", return_value=(True, "")),
            patch("pywhispr.app.save_config") as save,
        ):
            app._enable_gpu()
        assert app.cfg.offer_gpu_setup is False
        save.assert_called_once()

    def test_the_tray_entry_reports_when_it_cannot_help(self, app):
        with (
            patch("pywhispr.cuda.can_offer", return_value=(False, "no NVIDIA GPU was found")),
            patch("pywhispr.cuda.is_installed", return_value=False),
            patch("pywhispr.directml.can_offer", return_value=(False, "no DirectX 12 GPU")),
            patch("pywhispr.directml.is_installed", return_value=False),
            patch("pywhispr.ui.setup_window.ask_to_enable") as ask,
        ):
            app._enable_gpu(asked_by_user=True)
        ask.assert_not_called()
        app.tray.notify.assert_called_once()


class TestGpuSettingsEntry:
    """Whether the settings page is offered a GPU row at all."""

    def _settings_kwargs(self, supported, qtbot, qapp):
        with patch("pywhispr.gpu.supported", return_value=supported):
            with isolated_app() as (instance, _tray_cls):
                instance.state = State.IDLE
                with patch(
                    "pywhispr.ui.settings_dialog.SettingsDialog.edit", return_value=None
                ) as edit:
                    instance._show_settings()
                return edit.call_args.kwargs

    def test_no_row_where_no_gpu_path_could_run(self, qtbot, qapp):
        """macOS: CUDA and DirectML have no build for it and MLX is already on Metal."""
        kwargs = self._settings_kwargs(False, qtbot, qapp)
        assert kwargs["on_enable_gpu"] is None
        assert kwargs["on_disable_gpu"] is None
        assert kwargs["gpu_active"] is None

    def test_a_row_where_one_could(self, qtbot, qapp):
        kwargs = self._settings_kwargs(True, qtbot, qapp)
        assert callable(kwargs["on_enable_gpu"])
        assert callable(kwargs["on_disable_gpu"])
        assert callable(kwargs["gpu_active"])

    def test_the_label_asks_whether_it_is_installed_and_on(self, app):
        with patch("pywhispr.gpu.installed", return_value=True):
            app.cfg.use_gpu = True
            assert app._gpu_active() is True
            app.cfg.use_gpu = False
            assert app._gpu_active() is False


class TestGpuDisable:
    """The tray entry the other way round: off, but nothing deleted."""

    @pytest.fixture
    def dialogs(self):
        with (
            patch("pywhispr.ui.setup_window.ask_to_disable") as ask,
            patch("pywhispr.ui.setup_window.say_restart_needed") as told,
            patch("pywhispr.gpu.save_config") as save,
            patch("pywhispr.cuda.is_installed", return_value=True),
            patch("pywhispr.cuda.remove") as cuda_remove,
            patch("pywhispr.directml.remove") as directml_remove,
        ):
            ask.return_value = True
            yield MagicMock(
                ask=ask,
                told=told,
                save=save,
                cuda_remove=cuda_remove,
                directml_remove=directml_remove,
            )

    def test_it_switches_off_and_says_a_restart_is_needed(self, app, dialogs):
        app.state = State.IDLE
        app._disable_gpu()
        assert app.cfg.use_gpu is False
        dialogs.save.assert_called_once()
        dialogs.told.assert_called_once()

    def test_it_deletes_nothing(self, app, dialogs):
        """"Keep the files" is the whole difference from "pywhispr disable-gpu"."""
        app.state = State.IDLE
        app._disable_gpu()
        dialogs.cuda_remove.assert_not_called()
        dialogs.directml_remove.assert_not_called()

    def test_declining_the_confirmation_changes_nothing(self, app, dialogs):
        app.state = State.IDLE
        dialogs.ask.return_value = False
        app._disable_gpu()
        assert app.cfg.use_gpu is True
        dialogs.save.assert_not_called()
        dialogs.told.assert_not_called()

    def test_it_will_not_fight_a_setup_that_is_still_running(self, app, dialogs):
        app.state = State.IDLE
        app._progress_window = MagicMock(gpu_running=True)
        with patch("pywhispr.ui.foreground.show_in_front") as shown:
            app._disable_gpu()
        shown.assert_called_once_with(app._progress_window)
        dialogs.ask.assert_not_called()

    def test_it_quotes_the_size_of_what_stays_on_disk(self, app, dialogs):
        from pywhispr import cuda

        app.state = State.IDLE
        app._disable_gpu()
        assert dialogs.ask.call_args.kwargs["download_mb"] == cuda.APPROXIMATE_DOWNLOAD_MB


class TestGpuEnabledAgain:
    def test_switching_it_back_on_needs_no_download(self, app):
        """The libraries never left, so the flag is the whole job."""
        app.cfg.use_gpu = False
        with (
            patch("pywhispr.gpu.installed", return_value=True),
            patch("pywhispr.gpu.save_config"),
            patch("pywhispr.ui.setup_window.say_restart_needed") as told,
            patch("pywhispr.ui.setup_window.ask_to_enable") as ask,
            patch.object(app, "_run_gpu_setup") as setup,
        ):
            app._enable_gpu()
        assert app.cfg.use_gpu is True
        told.assert_called_once()
        ask.assert_not_called()
        setup.assert_not_called()

    def test_a_setup_asked_for_while_switched_off_switches_it_on(self, app):
        """The verification subprocess reads the config; off means it reports the CPU."""
        app.cfg.use_gpu = False
        with (
            patch("pywhispr.gpu.save_config"),
            patch.object(app, "_setup_window") as window,
        ):
            app._run_gpu_setup(kind="cuda")
        assert app.cfg.use_gpu is True
        window.assert_called_once()

    def test_a_machine_with_nothing_installed_still_gets_the_offer(self, app):
        with (
            patch("pywhispr.gpu.installed", return_value=False),
            patch("pywhispr.cuda.can_offer", return_value=(True, "")),
            patch("pywhispr.ui.setup_window.ask_to_enable", return_value=False) as ask,
        ):
            app._enable_gpu()
        ask.assert_called_once()

    def test_the_offer_stays_away_while_it_is_switched_off(self, app):
        """A hand-edited config can have use_gpu false with the offer still on."""
        app.cfg.offer_gpu_setup = True
        app.cfg.use_gpu = False
        with patch.object(app, "_enable_gpu") as enable:
            app._maybe_offer_gpu()
        enable.assert_not_called()


class TestGpuAskedBeforeAnyDownload:
    """Asked after loading, the answer comes too late to save the wasted download."""

    def _first_run(self, app, answer=True, cached=False):
        app.cfg.offer_gpu_setup = True
        app._load_model = MagicMock()
        with (
            patch("pywhispr.download.model_cached", return_value=cached),
            patch("pywhispr.cuda.can_offer", return_value=(True, "")),
            patch("pywhispr.ui.setup_window.ask_to_enable", return_value=answer) as ask,
            patch("pywhispr.app.save_config"),
            patch.object(app, "_run_gpu_setup") as setup,
            patch.object(app, "_begin_model_load") as load,
        ):
            deferred = app._offer_gpu_before_downloading()
        return ask, setup, load, deferred

    def test_accepting_holds_the_model_load_until_cuda_is_ready(self, app):
        _, setup, load, deferred = self._first_run(app, answer=True)
        assert deferred is True
        setup.assert_called_once()
        load.assert_not_called()  # otherwise int8 downloads alongside it

    def test_accepting_switches_to_full_precision_first(self, app):
        self._first_run(app, answer=True)
        assert app.cfg.model_quantization == ""  # the GPU is slower on int8

    def test_declining_loads_straight_away(self, app):
        _, setup, _, deferred = self._first_run(app, answer=False)
        assert deferred is False
        setup.assert_not_called()

    def test_not_asked_when_the_model_is_already_downloaded(self, app):
        ask, _, _, deferred = self._first_run(app, cached=True)
        ask.assert_not_called()
        assert deferred is False

    def test_a_failed_setup_falls_back_to_the_cpu_model(self, app):
        app.cfg.model_quantization = ""
        app._waiting_for_gpu_setup = True
        with (
            patch("pywhispr.app.save_config"),
            patch.object(app, "_begin_model_load") as load,
        ):
            app._on_gpu_setup_finished(worked=False)
        assert app.cfg.model_quantization is None
        load.assert_called_once()

    def test_a_working_setup_loads_without_a_restart(self, app):
        """The libraries landed before any session was built, so this process can use them."""
        app.cfg.model_quantization = ""
        app._waiting_for_gpu_setup = True
        with (
            patch("pywhispr.app.save_config"),
            patch.object(app, "_begin_model_load") as load,
        ):
            app._on_gpu_setup_finished(worked=True)
        assert app.cfg.model_quantization == ""
        load.assert_called_once()

    def test_a_tray_triggered_setup_does_not_reload_the_model(self, app):
        """The model is already loaded there; reloading would download all over again."""
        app._waiting_for_gpu_setup = False
        with (
            patch("pywhispr.app.save_config"),
            patch.object(app, "_begin_model_load") as load,
        ):
            app._on_gpu_setup_finished(worked=True)
        load.assert_not_called()

    def test_it_is_not_asked_twice_in_one_run(self, app):
        self._first_run(app, answer=False)
        with patch("pywhispr.ui.setup_window.ask_to_enable") as ask:
            app._maybe_offer_gpu()
        ask.assert_not_called()


class TestModelDownloadProgress:
    def _window(self, app):
        """Stand in for the real window, which owns threads and timers."""
        window = MagicMock()
        window.gpu_running = False
        app._progress_window = window
        return window

    def test_shown_only_when_nothing_is_cached(self, app):
        with (
            patch("pywhispr.download.model_cached", return_value=True),
            patch.object(app, "_setup_window") as window,
        ):
            app._show_model_download()
        window.assert_not_called()

        with (
            patch("pywhispr.download.model_cached", return_value=False),
            patch.object(app, "_setup_window") as window,
        ):
            app._show_model_download()
        window.assert_called_once()

    def test_the_size_shown_is_the_variant_about_to_be_fetched(self, app):
        """The variant is chosen on the worker thread, so it must be forced early."""

        class Backend:
            quantization = None

            def choose_quantization(self):
                self.quantization = "int8"

            @property
            def download_mb(self):
                return 650 if self.quantization else 2450

        app.backend = Backend()
        window = self._window(app)
        with (
            patch("pywhispr.download.model_cached", return_value=False),
            patch.object(app, "_setup_window", return_value=window),
        ):
            app._show_model_download()
        window.track_model_download.assert_called_once_with(650)

    def test_a_backend_without_variants_still_shows_progress(self, app):
        app.backend = MagicMock(spec=["download_mb", "name"], download_mb=2450)
        window = self._window(app)
        with (
            patch("pywhispr.download.model_cached", return_value=False),
            patch.object(app, "_setup_window", return_value=window),
        ):
            app._show_model_download()
        window.track_model_download.assert_called_once_with(2450)

    def test_one_window_whichever_download_starts_first(self, app):
        """A window each is what the user saw: two bars over overlapping bytes."""
        window = self._window(app)
        with (
            patch("pywhispr.download.model_cached", return_value=False),
            patch.object(app, "_setup_window", return_value=window) as factory,
        ):
            app._show_model_download()  # model first
            app._run_gpu_setup()  # then GPU, from the tray
        assert factory.call_count == 2  # the same window both times
        window.track_model_download.assert_called_once()
        window.start_gpu_setup.assert_called_once()

    def test_the_window_is_reused_not_recreated(self, app, qtbot):
        first = app._setup_window()
        qtbot.addWidget(first)
        assert app._setup_window() is first

    def test_told_when_the_model_is_ready(self, app):
        window = self._window(app)
        app._on_model_ready()
        window.finish_model.assert_called_once_with(None)

    def test_a_failed_load_says_so_in_the_window(self, app):
        window = self._window(app)
        app._on_model_failed("RuntimeError: offline")
        assert "offline" in window.finish_model.call_args.args[0]


class TestAudioDucking:
    """Other apps go quiet while recording; every exit path brings them back."""

    def _ducked(self, app):
        app.ducker = MagicMock()
        # A plain stub, not the fixture's MagicMock. Stopping a recording submits a
        # transcription, so the worker thread reaches the backend while this thread
        # is still asserting on the ducker — and unittest.mock is not thread-safe,
        # so two threads building child mocks livelock. It cost an afternoon once:
        # the suite hung at whichever test happened to be running, nowhere near the
        # cause. These tests are about the ducker, so the backend can be inert.
        app.backend = SimpleNamespace(name="ducking-stub", transcribe=lambda audio: "")
        app._on_model_ready()
        return app.ducker

    def test_recording_ducks_and_stopping_restores(self, app):
        ducker = self._ducked(app)
        app._on_toggle()  # start recording
        ducker.duck.assert_called_once()
        ducker.restore.assert_not_called()
        app._on_toggle()  # stop
        ducker.restore.assert_called_once()

    def test_mic_failure_does_not_duck(self, app):
        ducker = self._ducked(app)
        app._test_recorder.start.side_effect = OSError("no microphone")
        with patch.object(app.tray, "notify", create=True):
            app._on_toggle()
        assert app.state == State.IDLE
        ducker.duck.assert_not_called()

    def test_recorder_failure_on_stop_still_restores(self, app):
        # PortAudio can raise from stream.stop()/close() (e.g. the microphone
        # was unplugged mid-recording). The ducked volumes must come back
        # anyway — Windows remembers per-app mixer levels forever.
        ducker = self._ducked(app)
        app._on_toggle()
        app._test_recorder.stop.side_effect = OSError("stream died")
        with pytest.raises(OSError):
            app._on_toggle()
        ducker.restore.assert_called_once()

    def test_quit_restores_even_when_recorder_stop_fails(self, app):
        ducker = self._ducked(app)
        app._on_toggle()
        app._test_recorder.recording = True
        app._test_recorder.stop.side_effect = OSError("stream died")
        with pytest.raises(OSError):
            app._quit()
        ducker.restore.assert_called_once()

    def test_max_duration_stop_restores(self, app):
        ducker = self._ducked(app)
        app._on_toggle()
        app._on_max_duration()
        ducker.restore.assert_called_once()

    def test_quit_restores_even_mid_recording(self, app):
        # Windows remembers per-app mixer levels, so quitting while ducked
        # would leave the user's other apps quiet for good.
        ducker = self._ducked(app)
        app._on_toggle()
        app._quit()
        ducker.restore.assert_called_once()

    def test_ducking_is_off_by_default(self, app):
        from pywhispr.ducking import NoOpDucker

        assert isinstance(app.ducker, NoOpDucker)


class TestRemoteQuit:
    """An installer — or a newer build that has just replaced this one — can ask
    the app to stop. See pywhispr.instance."""

    def test_no_guard_means_nothing_to_release(self, app):
        """The suite builds apps directly and never goes through run_app, which is
        what keeps it from claiming a real named event or socket."""
        assert app._guard is None
        app._quit()  # must not raise

    def test_quitting_is_idempotent(self, app):
        """Reachable from the tray and from a request at the same time now."""
        with patch("pywhispr.app.QApplication.quit") as quit_:
            app._quit()
            app._quit()
        quit_.assert_called_once()

    def test_the_guard_is_released(self, qapp):
        guard = MagicMock()
        with isolated_app() as (instance_app, _tray):
            instance_app._guard = guard
            with patch("pywhispr.app.QApplication.quit"):
                instance_app._quit()
        guard.release.assert_called_once()

    def test_a_request_reaches_the_shutdown(self, qapp):
        """Emitted on the main thread deliberately: the cross-thread hop is
        exercised in test_instance.py, which has no mocks to trip over."""
        with isolated_app() as (instance_app, _tray):
            with (
                patch("pywhispr.app.QApplication.quit") as quit_,
                patch("pywhispr.app.QApplication.closeAllWindows") as close_all,
                patch("pywhispr.app.os._exit") as hard_exit,
            ):
                instance_app._quit_requested.emit()
            quit_.assert_called_once()
            close_all.assert_called_once()
            # The transcription worker's threads are joined at interpreter exit, so
            # a request during a model load would otherwise hold the process open
            # for as long as the load takes — with an installer waiting on it.
            hard_exit.assert_called_once_with(0)

    def test_the_hotkey_is_not_re_armed_while_quitting(self, app):
        """A request unwinds whatever dialog is open, and the `finally` that
        re-arms the listener then runs on the way out."""
        with patch("pywhispr.app.QApplication.quit"):
            app._quit()
        app.listener.start.reset_mock()
        app._resume_listeners()
        app.listener.start.assert_not_called()

    def test_a_download_in_flight_is_abandoned(self, app):
        """Or a killed install leaves a pip or Hugging Face download writing into
        the directory the upgrade is replacing."""
        app._progress_window = MagicMock()
        with patch("pywhispr.app.QApplication.quit"):
            app._quit()
        app._progress_window.abandon.assert_called_once()


class TestHistoryRecall:
    """A transcript that auto-pasted into the wrong window can be pasted again."""

    def _ready(self, app):
        app._on_model_ready()
        assert app.state == State.IDLE

    def _dictate(self, app, qtbot, text):
        app._test_backend.transcribe.return_value = text
        app._on_toggle()  # record
        with patch.object(app.injector, "insert"):
            app._on_toggle()  # stop → transcribe
            wait_for_worker(app, qtbot)
        app.injector.finished.emit(True)

    def test_transcripts_are_remembered(self, app, qtbot):
        self._ready(app)
        self._dictate(app, qtbot, "first one")
        self._dictate(app, qtbot, "second one")
        assert list(app._history) == ["second one", "first one"]

    def test_empty_transcripts_are_not_remembered(self, app, qtbot):
        self._ready(app)
        self._dictate(app, qtbot, "   ")
        assert list(app._history) == []

    def test_chosen_transcript_is_pasted(self, app, qtbot):
        self._ready(app)
        self._dictate(app, qtbot, "the lost sentence")
        with (
            patch(
                "pywhispr.ui.history_dialog.HistoryDialog.choose",
                return_value="the lost sentence",
            ),
            patch("pywhispr.ui.foreground.remember_foreground", return_value=1234) as remember,
            patch("pywhispr.ui.foreground.restore_foreground") as restore,
            patch.object(app.injector, "insert") as insert,
        ):
            app._show_history()
            assert app.state == State.INSERTING
            remember.assert_called_once()
            # The focus goes back to where it was *before* the paste keystroke.
            restore.assert_called_once_with(1234)
            qtbot.waitUntil(lambda: insert.called, timeout=1000)
            # No rich spans, and correctly so: the history keeps plain text, because
            # a remembered transcript is re-pasted somewhere else entirely. Anything
            # that was markup comes back as the words it degrades to.
            insert.assert_called_once_with("the lost sentence")
        app.injector.finished.emit(True)
        assert app.state == State.IDLE

    def test_cancelling_pastes_nothing(self, app, qtbot):
        self._ready(app)
        self._dictate(app, qtbot, "the lost sentence")
        with (
            patch("pywhispr.ui.history_dialog.HistoryDialog.choose", return_value=None),
            patch.object(app.injector, "insert") as insert,
        ):
            app._show_history()
        qtbot.wait(50)
        insert.assert_not_called()
        assert app.state == State.IDLE

    def test_empty_history_only_notifies(self, app):
        self._ready(app)
        with (
            patch("pywhispr.ui.history_dialog.HistoryDialog.choose") as choose,
            patch.object(app.injector, "insert") as insert,
        ):
            app._show_history()
        choose.assert_not_called()
        insert.assert_not_called()

    def test_ignored_while_recording(self, app, qtbot):
        self._ready(app)
        self._dictate(app, qtbot, "the lost sentence")
        app._on_toggle()  # recording
        with patch("pywhispr.ui.history_dialog.HistoryDialog.choose") as choose:
            app._show_history()
        choose.assert_not_called()
        assert app.state == State.RECORDING

    def test_recall_claims_no_hotkey_of_its_own(self, qtbot, qapp):
        """The picker is a tray menu item; only dictation registers a hotkey."""
        from pywhispr.app import PyWhisprApp
        from pywhispr.config import Config

        with (
            patch("pywhispr.app.create_backend"),
            patch("pywhispr.app.AudioRecorder"),
            patch("pywhispr.app.TrayIcon"),
            patch("pywhispr.app.load_vocabulary", return_value=[]),
            patch("pywhispr.app.create_hotkey_listener") as make_listener,
        ):
            instance = PyWhisprApp(
                Config(play_sounds=False, api_enabled=False, offer_gpu_setup=False)
            )
        chords = [call.args[0] for call in make_listener.call_args_list]
        # Dictation and reset register; the picker is a tray menu item.
        assert chords == [Config().hotkey, Config().reset_hotkey]
        instance._worker.shutdown(wait=True)


class TestResetHotkey:
    def _recording(self, app):
        app._on_model_ready()
        app._on_toggle()
        assert app.state == State.RECORDING

    def test_drops_the_audio_and_keeps_recording(self, app):
        self._recording(app)
        app._on_reset()
        app._test_recorder.reset.assert_called_once_with()
        assert app.state == State.RECORDING
        app._test_recorder.stop.assert_not_called()

    def test_ignored_when_not_recording(self, app):
        app._on_model_ready()
        assert app.state == State.IDLE
        app._on_reset()
        app._test_recorder.reset.assert_not_called()
        assert app.state == State.IDLE

    def test_a_failing_reset_leaves_the_recording_alone(self, app):
        self._recording(app)
        app._test_recorder.reset.side_effect = RuntimeError("boom")
        app._on_reset()
        assert app.state == State.RECORDING

    def test_no_listener_when_unset_or_clashing(self, qtbot, qapp):
        from pywhispr.app import PyWhisprApp

        for reset_hotkey in ("", Config().hotkey):
            with (
                patch("pywhispr.app.create_backend"),
                patch("pywhispr.app.AudioRecorder"),
                patch("pywhispr.app.TrayIcon"),
                patch("pywhispr.app.load_vocabulary", return_value=[]),
                patch("pywhispr.app.create_hotkey_listener"),
            ):
                instance = PyWhisprApp(
                    Config(
                        play_sounds=False,
                        api_enabled=False,
                        offer_gpu_setup=False,
                        reset_hotkey=reset_hotkey,
                    )
                )
            assert instance.reset_listener is None
            instance._worker.shutdown(wait=True)


class TestVoiceReset:
    def test_only_the_tail_is_inserted(self, app, qtbot):
        app._on_model_ready()
        with patch.object(app.injector, "insert") as insert:
            app._on_transcribed("Book the room. Clear clear, book the hall.")
        insert.assert_called_once_with("Book the hall.", ())
        assert list(app._history) == ["Book the hall."]  # not the discarded half

    def test_the_words_used_as_words_are_left_alone(self, app):
        app._on_model_ready()
        with patch.object(app.injector, "insert") as insert:
            app._on_transcribed("Please clear that surface.")
        insert.assert_called_once_with("Please clear that surface.", ())

    def test_no_phrases_configured_inserts_verbatim(self, app):
        app._on_model_ready()
        app._reset_phrases = compile_reset_phrases([])
        with patch.object(app.injector, "insert") as insert:
            app._on_transcribed("Book the room. Clear clear. Book the hall.")
        insert.assert_called_once_with("Book the room. Clear clear. Book the hall.", ())


class TestSettingsVisit:
    """The tray's one door: the hotkey is silenced for the whole visit and the
    edited config is applied when it closes."""

    def _open(self, app, returned):
        with (
            patch("pywhispr.ui.settings_dialog.SettingsDialog.edit", return_value=returned),
            patch("pywhispr.app.save_config") as save,
        ):
            app._show_settings()
        return save

    def test_the_hotkey_is_silenced_around_the_window(self, app):
        """A chord pressed inside a settings dialog must not start a recording."""
        app.state = State.IDLE
        self._open(app, None)
        app.listener.stop.assert_called_once()
        app.listener.start.assert_called_once()

    def test_the_hotkey_comes_back_even_if_the_window_explodes(self, app):
        app.state = State.IDLE
        with patch(
            "pywhispr.ui.settings_dialog.SettingsDialog.edit",
            side_effect=RuntimeError("no Qt today"),
        ):
            with pytest.raises(RuntimeError):
                app._show_settings()
        app.listener.start.assert_called_once()

    def test_it_waits_until_the_app_is_idle(self, app):
        app.state = State.RECORDING
        with patch("pywhispr.ui.settings_dialog.SettingsDialog.edit") as edit:
            app._show_settings()
        edit.assert_not_called()

    def test_cancelling_saves_nothing(self, app):
        app.state = State.IDLE
        save = self._open(app, None)
        save.assert_not_called()

    def test_saving_applies_and_writes(self, app):
        app.state = State.IDLE
        edited = dataclasses.replace(app.cfg, remove_fillers=False, max_recording_seconds=30)
        save = self._open(app, edited)
        save.assert_called_once()
        assert app.cfg.remove_fillers is False
        assert app._max_duration_timer.interval() == 30_000

    def test_gpu_changes_made_inside_the_window_survive_the_save(self, app):
        """gpu.turn_off writes to the live config while the window is open; a
        wholesale replace would put the window's older copy back over it."""
        app.state = State.IDLE
        edited = dataclasses.replace(app.cfg)  # copied before the GPU button was used
        app.cfg.use_gpu = False
        self._open(app, edited)
        assert app.cfg.use_gpu is False

    def test_a_new_hotkey_is_registered(self, app):
        app.state = State.IDLE
        edited = dataclasses.replace(app.cfg, hotkey="<ctrl>+<alt>+j")
        with patch("pywhispr.app.create_hotkey_listener") as make:
            self._open(app, edited)
        assert make.call_args.args[0] == "<ctrl>+<alt>+j"
        assert app.cfg.hotkey == "<ctrl>+<alt>+j"

    def test_a_hotkey_that_will_not_register_is_reverted(self, app):
        app.state = State.IDLE
        old = app.cfg.hotkey
        edited = dataclasses.replace(app.cfg, hotkey="<nonsense>")
        with patch("pywhispr.app.create_hotkey_listener") as make:
            make.side_effect = [RuntimeError("taken"), MagicMock()]
            self._open(app, edited)
        assert app.cfg.hotkey == old
        app.tray.notify.assert_called()

    def test_settings_that_need_a_restart_say_so(self, app):
        app.state = State.IDLE
        edited = dataclasses.replace(app.cfg, api_port=9999)
        self._open(app, edited)
        assert "Restart" in app.tray.notify.call_args.args[0]

    def test_settings_that_do_not_stay_quiet(self, app):
        app.state = State.IDLE
        edited = dataclasses.replace(app.cfg, play_sounds=False)
        self._open(app, edited)
        app.tray.notify.assert_not_called()


class TestMicrophoneChoice:
    """The chosen input device is resolved by name at every recording."""

    def test_no_choice_means_the_system_default(self, app):
        assert app._input_device() is None

    def test_a_legacy_index_is_still_honoured(self, app):
        """Configs written before names existed keep working untouched."""
        app.cfg.input_device = 3
        assert app._input_device() == 3

    def test_the_chosen_device_is_looked_up_by_name(self, app):
        app.cfg.input_device_name = "Yeti"
        app.cfg.input_device = 3  # stale index: the name wins
        with patch("pywhispr.app.find_device", return_value=1):
            assert app._input_device() == 1

    def test_a_missing_device_falls_back_to_the_default_and_says_so(self, app):
        app.cfg.input_device_name = "Yeti"
        with patch("pywhispr.app.find_device", return_value=None):
            assert app._input_device() is None
        assert "Yeti" in app.tray.notify.call_args.args[1]

    def test_it_complains_once_rather_than_every_dictation(self, app):
        app.cfg.input_device_name = "Yeti"
        with patch("pywhispr.app.find_device", return_value=None):
            app._input_device()
            app._input_device()
        assert app.tray.notify.call_count == 1

    def test_it_complains_again_after_the_device_came_back(self, app):
        app.cfg.input_device_name = "Yeti"
        with patch("pywhispr.app.find_device", side_effect=[None, 2, None]):
            app._input_device()
            app._input_device()
            app._input_device()
        assert app.tray.notify.call_count == 2

    def test_recording_opens_the_resolved_device(self, app):
        app._on_model_ready()
        app.cfg.input_device_name = "Yeti"
        with patch("pywhispr.app.find_device", return_value=4):
            app._start_recording()
        assert app.recorder.device == 4
