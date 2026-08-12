"""The plugin framework: what it matches, what it refuses, and what it splices."""

from __future__ import annotations

import logging

import pytest

from pywhispr.config import Config
from pywhispr.plugins.actions import ActionRunner
from pywhispr.plugins.api import Match, Rewrite, Trigger
from pywhispr.plugins.engine import (
    MAX_REPLACEMENT_CHARS,
    PendingAction,
    Plugin,
    apply_plugins,
    compile_patterns,
    compile_trigger,
)
from pywhispr.plugins.registry import ensure_plugins_dir, load_plugins


def make_plugin(name="test", phrase="marker", rewrite=None, act=None, **trigger_kwargs):
    """A Plugin with its patterns compiled, the way the registry would build it."""
    triggers = (Trigger(phrase=phrase, **trigger_kwargs),)
    return Plugin(
        name=name,
        triggers=triggers,
        rewrite=rewrite,
        act=act,
        patterns=compile_patterns(triggers),
    )


def claim_all(replacement="X"):
    """A rewrite that claims from its first context word to the end of the trigger."""

    def rewrite(match: Match) -> Rewrite | None:
        if not match.words_before:
            return None
        return match.claim_from(match.words_before[-1], replacement)

    return rewrite


class TestTriggerCompilation:
    def test_matches_on_a_word_boundary_only(self):
        pattern = compile_trigger(Trigger(phrase="marker"))
        assert pattern.search("say marker now")
        assert not pattern.search("a supermarkerish thing")

    def test_is_case_insensitive(self):
        pattern = compile_trigger(Trigger(phrase="marker"))
        assert pattern.search("MARKER")

    def test_allows_punctuation_between_a_phrase_words(self):
        """The model punctuates freely, so "clear, clear" is still one phrase."""
        pattern = compile_trigger(Trigger(phrase="two words"))
        assert pattern.search("say two, words please")
        assert pattern.search("say two words please")

    def test_at_segment_end_requires_a_clause_to_end(self):
        pattern = compile_trigger(Trigger(phrase="marker", at_segment_end=True))
        assert pattern.search("here it is, marker.")
        assert pattern.search("here it is marker")  # end of the transcript
        assert not pattern.search("the marker is red")

    def test_a_raw_pattern_is_used_as_given(self):
        pattern = compile_trigger(Trigger(phrase="", pattern=r"\d+ degrees"))
        assert pattern.search("about 20 degrees")

    def test_an_uncompilable_pattern_is_dropped_not_raised(self):
        assert compile_trigger(Trigger(phrase="x", pattern="(unclosed")) is None

    def test_an_empty_phrase_is_dropped(self):
        assert compile_trigger(Trigger(phrase="   ")) is None


class TestRewriting:
    def test_replaces_the_claimed_span_only(self):
        plugin = make_plugin(rewrite=claim_all("!"))
        assert apply_plugins("keep this word marker and this", [plugin]).text == (
            "keep this ! and this"
        )

    def test_returning_none_leaves_the_transcript_alone(self):
        plugin = make_plugin(rewrite=lambda match: None)
        assert apply_plugins("a marker here", [plugin]).text == "a marker here"

    def test_no_plugins_is_the_identity(self):
        assert apply_plugins("untouched", []).text == "untouched"

    def test_every_occurrence_is_rewritten(self):
        plugin = make_plugin(rewrite=claim_all("X"))
        assert apply_plugins("one marker two marker", [plugin]).text == "X X"

    def test_a_zero_width_claim_changes_nothing(self):
        plugin = make_plugin(rewrite=lambda match: match.nothing_to_change())
        result = apply_plugins("a marker here", [plugin])
        assert result.text == "a marker here"
        assert result.rewrites == 1  # claimed, but with nothing to change

    def test_counts_the_rewrites_it_made(self):
        plugin = make_plugin(rewrite=claim_all("X"))
        assert apply_plugins("one marker two marker", [plugin]).rewrites == 2


class TestClaimValidation:
    """Every one of these leaves the transcript untouched: the audio is gone."""

    @pytest.mark.parametrize(
        "claim",
        [
            Rewrite(start=-1, end=3, text="x"),  # before the beginning
            Rewrite(start=3, end=1, text="x"),  # backwards
            Rewrite(start=0, end=10_000, text="x"),  # past the end
            Rewrite(start=0, end=2, text=None),  # not a string
            Rewrite(start=0.5, end=2, text="x"),  # not an integer
        ],
    )
    def test_a_malformed_claim_is_refused(self, claim):
        plugin = make_plugin(rewrite=lambda match: claim)
        assert apply_plugins("a marker here", [plugin]).text == "a marker here"

    def test_something_that_is_not_a_rewrite_is_refused(self):
        plugin = make_plugin(rewrite=lambda match: "just a string")
        assert apply_plugins("a marker here", [plugin]).text == "a marker here"

    def test_a_claim_outside_the_window_shown_is_refused(self):
        """A trigger at the end must not be able to rewrite the beginning.

        The window is the context the plugin was handed, so this is what stops a
        plugin replacing a whole dictation from one word at the end of it.
        """
        text = "one two three four five six seven eight nine marker"
        plugin = make_plugin(rewrite=lambda match: Rewrite(0, len(text), "gone"))
        assert apply_plugins(text, [plugin]).text == text

    def test_an_oversized_replacement_is_refused(self):
        plugin = make_plugin(rewrite=claim_all("y" * (MAX_REPLACEMENT_CHARS + 1)))
        assert apply_plugins("a marker here", [plugin]).text == "a marker here"

    def test_a_replacement_at_the_limit_is_allowed(self):
        plugin = make_plugin(rewrite=claim_all("y" * MAX_REPLACEMENT_CHARS))
        assert apply_plugins("a marker here", [plugin]).rewrites == 1

    def test_a_raising_plugin_costs_only_its_own_match(self):
        def explode(match):
            raise RuntimeError("bad plugin")

        good = make_plugin(name="good", phrase="here", rewrite=claim_all("!"))
        bad = make_plugin(name="bad", phrase="marker", rewrite=explode)
        # "bad" leaves its own match alone; "good" still claims "is here".
        assert apply_plugins("a marker is here", [bad, good]).text == "a marker !"


class TestOverlap:
    def test_the_leftmost_claim_wins_and_the_loser_is_dropped(self):
        """Position decides, not load order: claims are offered left to right.

        Both of these want the words "a word"/"word marker", which overlap. The
        one whose trigger comes first in the transcript is offered first, and the
        other is dropped rather than half-applied.
        """
        early = make_plugin(name="early", phrase="word", rewrite=claim_all("EARLY"))
        late = make_plugin(name="late", phrase="marker", rewrite=claim_all("LATE"))
        for order in ([early, late], [late, early]):
            assert apply_plugins("a word marker", order).text == "EARLY marker"

    def test_load_order_breaks_a_tie_at_the_same_position(self):
        """Two plugins on the same words resolve by load order, the same way twice."""
        first = make_plugin(name="first", phrase="marker", rewrite=claim_all("FIRST"))
        second = make_plugin(name="second", phrase="marker", rewrite=claim_all("SECOND"))
        assert apply_plugins("a marker", [first, second]).text == "FIRST"
        assert apply_plugins("a marker", [second, first]).text == "SECOND"

    def test_non_overlapping_claims_all_apply(self):
        plugin = make_plugin(rewrite=claim_all("X"))
        assert apply_plugins("one marker and two marker", [plugin]).text == "X and X"


class TestContext:
    def test_words_before_are_in_reading_order_nearest_last(self):
        seen = {}

        def rewrite(match):
            seen["before"] = [word.text for word in match.words_before]
            seen["after"] = [word.text for word in match.words_after]
            return None

        apply_plugins("alpha beta marker gamma delta", [make_plugin(rewrite=rewrite)])
        assert seen["before"] == ["alpha", "beta"]
        assert seen["after"] == ["gamma", "delta"]

    def test_context_is_capped(self):
        seen = {}

        def rewrite(match):
            seen["before"] = len(match.words_before)
            return None

        apply_plugins("a b c d e f g h marker", [make_plugin(rewrite=rewrite)])
        assert seen["before"] == 4  # LOOKBEHIND_WORDS

    def test_a_trigger_with_no_context_can_still_claim_itself(self):
        plugin = make_plugin(rewrite=lambda match: Rewrite(match.start, match.end, "X"))
        assert apply_plugins("marker", [plugin]).text == "X"

    def test_trigger_text_is_what_matched(self):
        seen = {}

        def rewrite(match):
            seen["text"] = match.trigger_text
            return None

        apply_plugins("say MARKER now", [make_plugin(rewrite=rewrite)])
        assert seen["text"] == "MARKER"


class TestActions:
    def test_an_action_is_collected_not_run(self):
        ran = []
        plugin = make_plugin(rewrite=claim_all("X"), act=ran.append)
        result = apply_plugins("a marker", [plugin])
        assert ran == []  # collected for later, on another thread
        assert [pending.plugin.name for pending in result.actions] == ["test"]

    def test_a_rewrite_returning_none_earns_no_action(self):
        """The plugin's own validation is the gate: not for us means not at all."""
        plugin = make_plugin(rewrite=lambda match: None, act=lambda match: None)
        assert apply_plugins("a marker", [plugin]).actions == ()

    def test_a_plugin_with_no_rewrite_acts_on_every_match(self):
        plugin = make_plugin(act=lambda match: None)
        result = apply_plugins("marker and marker", [plugin])
        assert len(result.actions) == 2
        assert result.text == "marker and marker"  # nothing claimed, nothing changed

    def test_a_refused_claim_earns_no_action(self):
        plugin = make_plugin(
            rewrite=lambda match: Rewrite(0, 10_000, "x"), act=lambda match: None
        )
        assert apply_plugins("a marker", [plugin]).actions == ()


class TestActionRunner:
    def test_runs_the_action_on_its_own_thread(self):
        done = []
        runner = ActionRunner()
        plugin = make_plugin(act=lambda match: done.append(match.trigger_text))
        match = Match(transcript="marker", start=0, end=6)
        try:
            assert runner.dispatch([PendingAction(plugin, match)]) == 1
            runner._pool.shutdown(wait=True)
        finally:
            runner.stop()
        assert done == ["marker"]

    def test_a_failing_action_is_contained(self, caplog):
        def explode(match):
            raise RuntimeError("no")

        runner = ActionRunner()
        plugin = make_plugin(name="boom", act=explode)
        match = Match(transcript="marker", start=0, end=6)
        try:
            with caplog.at_level(logging.ERROR):
                runner.dispatch([PendingAction(plugin, match)])
                runner._pool.shutdown(wait=True)
        finally:
            runner.stop()
        assert "boom" in caplog.text

    def test_a_plugin_without_an_act_is_skipped(self):
        runner = ActionRunner()
        plugin = make_plugin(rewrite=claim_all())
        try:
            assert runner.dispatch([PendingAction(plugin, Match("marker", 0, 6))]) == 0
        finally:
            runner.stop()

    def test_dispatching_after_stop_does_not_raise(self):
        runner = ActionRunner()
        plugin = make_plugin(act=lambda match: None)
        runner.stop()
        assert runner.dispatch([PendingAction(plugin, Match("marker", 0, 6))]) == 0


class TestRegistry:
    """Loading from a folder. Never the developer's real one — tmp_path throughout."""

    @pytest.fixture(autouse=True)
    def _isolated_dir(self, tmp_path, monkeypatch):
        self.directory = tmp_path / "plugins"
        self.directory.mkdir()
        monkeypatch.setattr("pywhispr.plugins.registry.plugins_dir", lambda: self.directory)
        monkeypatch.setattr("pywhispr.plugins.registry.BUILTINS", ())

    def write(self, name: str, body: str) -> None:
        (self.directory / f"{name}.py").write_text(body, encoding="utf-8")

    def test_loads_a_plugin_from_a_file(self):
        self.write(
            "greet",
            "from pywhispr.plugins.api import Trigger\n"
            "TRIGGERS = (Trigger(phrase='hello'),)\n"
            "def rewrite(match):\n"
            "    return match.claim(match.start, match.end, 'hi')\n",
        )
        plugins = load_plugins(Config())
        assert [p.name for p in plugins] == ["greet"]
        assert apply_plugins("say hello", plugins).text == "say hi"

    def test_a_broken_file_costs_only_itself(self):
        self.write("broken", "this is not python(")
        self.write(
            "fine",
            "from pywhispr.plugins.api import Trigger\n"
            "TRIGGERS = (Trigger(phrase='hello'),)\n"
            "def act(match):\n    pass\n",
        )
        assert [p.name for p in load_plugins(Config())] == ["fine"]

    def test_a_plugin_with_no_triggers_is_ignored(self):
        self.write("empty", "def rewrite(match):\n    return None\n")
        assert load_plugins(Config()) == []

    def test_a_plugin_with_neither_hook_is_ignored(self):
        self.write(
            "inert",
            "from pywhispr.plugins.api import Trigger\nTRIGGERS = (Trigger(phrase='x'),)\n",
        )
        assert load_plugins(Config()) == []

    def test_underscore_files_are_helpers_not_plugins(self):
        self.write(
            "_shared",
            "from pywhispr.plugins.api import Trigger\n"
            "TRIGGERS = (Trigger(phrase='x'),)\n"
            "def act(match):\n    pass\n",
        )
        assert load_plugins(Config()) == []

    def test_the_master_switch_loads_nothing(self):
        self.write(
            "greet",
            "from pywhispr.plugins.api import Trigger\n"
            "TRIGGERS = (Trigger(phrase='hello'),)\n"
            "def act(match):\n    pass\n",
        )
        assert load_plugins(Config(plugins_enabled=False)) == []

    def test_a_plugin_can_be_disabled_by_name(self):
        self.write(
            "greet",
            "from pywhispr.plugins.api import Trigger\n"
            "TRIGGERS = (Trigger(phrase='hello'),)\n"
            "def act(match):\n    pass\n",
        )
        cfg = Config(plugins={"greet": {"enabled": False}})
        assert load_plugins(cfg) == []

    def test_an_unmentioned_plugin_is_on(self):
        self.write(
            "greet",
            "from pywhispr.plugins.api import Trigger\n"
            "TRIGGERS = (Trigger(phrase='hello'),)\n"
            "def act(match):\n    pass\n",
        )
        cfg = Config(plugins={"other": {"enabled": False}})
        assert [p.name for p in load_plugins(cfg)] == ["greet"]

    def test_a_missing_folder_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(
            "pywhispr.plugins.registry.plugins_dir", lambda: self.directory / "nope"
        )
        assert load_plugins(Config()) == []

    def test_the_folder_is_created_with_its_readme(self, monkeypatch):
        target = self.directory / "made"
        monkeypatch.setattr("pywhispr.plugins.registry.plugins_dir", lambda: target)
        created = ensure_plugins_dir()
        assert created.is_dir()
        assert "TRIGGERS" in (created / "README.txt").read_text(encoding="utf-8")

    def test_the_builtin_emoji_plugin_is_loadable(self, monkeypatch):
        """The real BUILTINS path, which is what ships."""
        monkeypatch.setattr("pywhispr.plugins.registry.BUILTINS", ("emoji",))
        assert [p.name for p in load_plugins(Config())] == ["emoji"]
