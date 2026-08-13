"""The built-in emoji plugin: what it converts, and what it leaves alone.

Characters are written as escapes, like the plugin itself, so this file stays
pure ASCII and no editor or terminal can quietly rewrite an assertion.
"""

from __future__ import annotations

import pytest

from pywhispr.plugins.builtin import emoji
from pywhispr.plugins.engine import Plugin, apply_plugins, compile_patterns

THUMBS_UP = "\U0001f44d"
HEART = "\U00002764"
FIRE = "\U0001f525"
ROCKET = "\U0001f680"
PARTY = "\U0001f389"
SHRUG = "\U0001f937"
MAN = "\U0001f468"
GUN = "\U0001f52b"

PLUGIN = Plugin(
    name="emoji",
    triggers=emoji.TRIGGERS,
    rewrite=emoji.rewrite,
    patterns=compile_patterns(emoji.TRIGGERS),
)


def convert(text: str) -> str:
    return apply_plugins(text, [PLUGIN]).text


class TestConversion:
    @pytest.mark.parametrize(
        ("said", "expected"),
        [
            ("Thumbs up emoji", THUMBS_UP),
            ("thumbs up emoji", THUMBS_UP),
            ("Thumbs up emoji.", THUMBS_UP),
            ("Heart emoji", HEART),
            ("Fire emoji", FIRE),
            ("Rocket emoji", ROCKET),
            ("Shrug emoji", SHRUG),
        ],
    )
    def test_a_name_before_the_trigger_becomes_the_character(self, said, expected):
        assert convert(said) == expected

    def test_keeps_the_words_around_it(self):
        assert convert("Nice work, thumbs up emoji.") == f"Nice work {THUMBS_UP}"

    def test_leaves_the_space_in_front_alone(self):
        assert convert("I agree thumbs up emoji") == f"I agree {THUMBS_UP}"

    def test_converts_more_than_one_in_a_sentence(self):
        assert convert("Rocket emoji, party emoji") == f"{ROCKET} {PARTY}"

    def test_survives_the_punctuation_the_model_adds(self):
        assert convert("Thumbs-up emoji") == THUMBS_UP
        assert convert("thumbs up, emoji") == THUMBS_UP

    def test_prefers_the_longer_name(self):
        """"plus one" is a name in its own right and must beat "one".

        Also the case that pins the alias table ahead of the function-word guard:
        both halves of "plus one" are function words, so a guard consulted first
        would refuse to look it up at all.
        """
        assert convert("Well said, plus one emoji.") == f"Well said {THUMBS_UP}"

    @pytest.mark.parametrize("said", ["I like emoji", "I want no emoji", "We are done emoji"])
    def test_ordinary_words_are_not_aliases(self, said):
        """"like", "no" and "done" are left out of the table on purpose."""
        assert convert(said) == said


class TestThePunctuationTheModelAdded:
    """The model appends a full stop to everything and commas at every pause.

    Neither belongs on an emoji, and the full stop also stops Teams and Slack
    rendering the large version, which they only do for a message that is nothing
    but emoji.
    """

    def test_the_appended_full_stop_goes(self):
        assert convert("Baby emoji.") == "\U0001f476"

    def test_an_ellipsis_goes_too(self):
        assert convert("Baby emoji…") == "\U0001f476"

    def test_the_comma_before_the_name_goes(self):
        # Said as "Hello smile emoji", transcribed as "Hello, smile emoji."
        assert convert("Hello, smile emoji.") == "Hello \U0001f642"

    def test_leaves_nothing_but_the_emoji_when_that_is_all_that_was_said(self):
        assert convert("Thumbs up emoji.") == THUMBS_UP

    def test_an_exclamation_is_the_speakers_and_stays(self):
        assert convert("Nice, party emoji!") == f"Nice {PARTY}!"

    def test_a_question_mark_stays(self):
        assert convert("Baby emoji?") == "\U0001f476?"

    def test_repeated_marks_are_not_the_models_and_stay(self):
        assert convert("Party emoji!!!") == f"{PARTY}!!!"

    def test_a_comma_that_separates_clauses_is_left_alone(self):
        """Only the trailing mark at the very end of the transcript is absorbed."""
        assert convert("I said hello, smile emoji, then left.") == (
            "I said hello \U0001f642, then left."
        )

    def test_a_sentence_break_in_front_is_left_alone(self):
        assert convert("Hello. Smile emoji.") == "Hello. \U0001f642"

    def test_no_separator_is_invented_at_the_start(self):
        assert convert("Smile emoji.") == "\U0001f642"


class TestChains:
    """"Man emoji gun emoji" — the other way people dictate these.

    A chain has an ordinary word after its first trigger, so the position guard
    cannot be the Trigger's at_segment_end: it would throw the first half away
    before the plugin ever saw it.
    """

    def test_two_links(self):
        assert convert("Man emoji gun emoji") == f"{MAN} {GUN}"

    def test_three_links(self):
        assert convert("Man emoji gun emoji fire emoji") == f"{MAN} {GUN} {FIRE}"

    def test_a_multi_word_name_can_lead_a_chain(self):
        assert convert("Thumbs up emoji rocket emoji") == f"{THUMBS_UP} {ROCKET}"

    def test_a_chain_with_a_comma_still_works(self):
        assert convert("Rocket emoji, party emoji") == f"{ROCKET} {PARTY}"

    def test_prose_between_two_mentions_is_not_a_chain(self):
        """The whole run between the triggers has to name an emoji.

        "and the water" resolves to nothing, so the first mention stays prose. The
        last one still converts, because it ends the transcript — that was true
        before chains existed.
        """
        got = convert("I use the fire emoji and the water emoji")
        assert got.startswith("I use the fire emoji and the ")

    def test_the_trigger_word_twice_over_is_not_a_chain(self):
        assert convert("Emoji emoji") == "Emoji emoji"


class TestNamesTheUnicodeDataMisses:
    """Words whose emoji the legacy Unicode names call something else entirely.

    Each of these resolved to nothing before it was given an alias — 🔫 is
    "PISTOL", 🍔 is "HAMBURGER", ⛳ is "FLAG IN HOLE".
    """

    @pytest.mark.parametrize(
        "word",
        [
            "gun",
            "sword",
            "golf",
            "swimming",
            "cycling",
            "bike",
            "piano",
            "headphones",
            "flashlight",
            "sofa",
            "backpack",
            "city",
            "cigarette",
            "chain",
            "rope",
            "pinch",
            "burger",
            "noodles",
            "soup",
            "donut",
            "corn",
        ],
    )
    def test_resolves_to_a_single_character(self, word):
        got = emoji._resolve(word)
        assert got is not None and len(got) == 1

    def test_through_the_whole_pass(self):
        assert convert("Gun emoji") == GUN


class TestSquashedTier:
    """People say compounds as one word, and the model writes them that way."""

    @pytest.mark.parametrize(
        ("said", "expected"),
        [("eyeroll", "\U0001f644"), ("thumbsup", THUMBS_UP), ("checkmark", "\U00002705")],
    )
    def test_a_spaceless_compound_resolves(self, said, expected):
        assert emoji._resolve(said) == expected

    def test_a_mis_split_phrase_resolves(self):
        """"thumb sup" is "thumbsup" once the spaces go — an exact hit, not a guess."""
        assert emoji._resolve("thumb sup") == THUMBS_UP

    def test_through_the_whole_pass(self):
        assert convert("Eyeroll emoji.") == "\U0001f644"

    def test_curated_aliases_win_over_index_names(self):
        squashed = emoji._squashed()
        assert squashed["thumbsup"] == THUMBS_UP
        assert squashed["redheart"] == HEART


class TestHomophoneTier:
    """The model writes a word that sounds right and means something else.

    "I roll" for "eyeroll" is the case that motivated all of this, and it is
    unreachable by letter-level matching: "iroll" is one edit from "troll" and
    three from "eyeroll", so the fuzzy tier alone answers confidently and wrongly.
    """

    @pytest.mark.parametrize(
        ("said", "expected"),
        [
            ("i roll", "\U0001f644"),
            ("hi five", "\U0001f64c"),
            ("plus won", THUMBS_UP),
            ("read heart", HEART),
            ("czech mark", "\U00002705"),
            ("waiving hand", "\U0001f44b"),
            ("preying hands", "\U0001f64f"),
        ],
    )
    def test_a_misheard_spelling_resolves(self, said, expected):
        assert emoji._resolve(said) == expected

    def test_the_case_from_the_bug_report(self):
        assert convert("I roll emoji") == "\U0001f644"
        assert convert("That is so annoying, I roll emoji") == "That is so annoying \U0001f644"

    def test_the_literal_spelling_always_wins(self):
        """"one hundred" must resolve as itself, never by way of "won"."""
        assert emoji._resolve("one hundred") == "\U0001f4af"

    def test_a_substituted_word_never_reaches_the_text(self):
        """The map transforms a lookup key, so "eye" cannot land in the sentence.

        This is the property that makes the map safe here and would not make it
        safe as a general vocabulary pass.
        """
        assert "eye" not in convert("I roll emoji")

    def test_variants_are_capped(self):
        many = " ".join(["i"] * 8)
        assert len(emoji._homophone_variants(many)) <= emoji.MAX_HOMOPHONE_VARIANTS

    def test_a_phrase_with_no_homophones_makes_no_variants(self):
        assert emoji._homophone_variants("thumbs up") == []

    def test_every_mapping_reaches_a_real_name(self):
        """An entry whose target appears in no emoji name is dead weight.

        Checked against the words of every alias and Unicode name rather than
        against _resolve, because a target may only be meaningful inside a phrase:
        "one" alone is a function word with no emoji, but "plus one" and "one
        hundred" are both aliases, so won -> one earns its place.
        """
        words = set()
        for key in emoji.ALIASES:
            words.update(key.split())
        for name in emoji._index():
            words.update(name.split())
        assert [w for w in emoji.HOMOPHONES.values() if w not in words] == []

    def test_no_mapping_is_circular(self):
        """A target that is itself a key would make the variant order matter."""
        assert not set(emoji.HOMOPHONES) & set(emoji.HOMOPHONES.values())


class TestFuzzyTier:
    """Last resort, and the only tier that can be wrong about a correct name."""

    @pytest.mark.parametrize(
        ("said", "expected"),
        [("partly popper", PARTY), ("rockit", ROCKET), ("banna", "🍌")],
    )
    def test_a_near_miss_resolves(self, said, expected):
        assert emoji._resolve(said) == expected

    def test_function_words_never_reach_it(self):
        """Without the shared guard, "the" is two edits from "tree"."""
        assert emoji._resolve("the") is None
        assert emoji._resolve("of the") is None

    def test_too_short_to_risk_an_edit(self):
        assert emoji._fuzzy("abc") is None  # under MIN_FUZZY_CHARS

    def test_the_trigger_word_never_reaches_it(self):
        assert emoji._fuzzy("emoji") is None

    def test_a_tie_is_no_answer(self, monkeypatch):
        """Two names equally close means neither, as vocab decides it too."""
        monkeypatch.setattr(
            emoji, "_squashed", lambda: {"aaaax": "\U0001f600", "aaaay": "\U0001f601"}
        )
        assert emoji._fuzzy("aaaaz") is None


class TestUnicodeNameTier:
    """The long tail, which comes from unicodedata rather than the alias table."""

    def test_an_exact_unicode_name_resolves(self):
        assert convert("Party popper emoji") == PARTY

    def test_a_prefix_at_a_word_boundary_resolves(self):
        # "waving hand" is "WAVING HAND SIGN" in the standard library's data.
        assert convert("Waving hand emoji") == "\U0001f44b"

    def test_a_name_containing_the_words_resolves(self):
        # "SLICE OF PIZZA"
        assert convert("Pizza emoji") == "\U0001f355"

    def test_an_alias_beats_the_index(self):
        """"heart" would find nothing in the legacy names: ~ is HEAVY BLACK HEART."""
        assert convert("Red heart emoji") == HEART


class TestLeavesAlone:
    @pytest.mark.parametrize(
        "said",
        [
            "Send me an emoji.",
            "Use the emoji picker.",
            "Emoji.",
            "I need an emoji for this.",
            "Which emoji did you mean?",
        ],
    )
    def test_words_that_name_no_emoji_are_untouched(self, said):
        assert convert(said) == said

    def test_a_sentence_about_an_emoji_is_untouched(self):
        """Not at the end of a clause, so these words are about it, not asking for it."""
        assert convert("The fire emoji is best.") == "The fire emoji is best."

    def test_the_trigger_word_alone_converts_nothing(self):
        assert convert("Emoji") == "Emoji"

    def test_function_words_never_resolve(self):
        assert convert("One of the emoji") == "One of the emoji"

    def test_a_transcript_with_no_trigger_is_identical(self):
        assert convert("Nothing to see here.") == "Nothing to see here."


class TestResolve:
    """The lookup on its own, where the tiers are easiest to see."""

    def test_normalises_case_and_separators(self):
        assert emoji._resolve("Thumbs-Up") == THUMBS_UP
        assert emoji._resolve("  thumbs   up  ") == THUMBS_UP

    def test_refuses_something_too_short_to_mean_anything(self):
        assert emoji._resolve("zz") is None  # under MIN_QUERY_CHARS

    def test_but_a_curated_alias_works_however_short(self):
        """The table outranks the length guard, or a listed term is dead code.

        "ok" is exactly that: two characters, in ALIASES, and refused by
        MIN_QUERY_CHARS until the alias lookup moved ahead of it.
        """
        assert emoji._resolve("ok") == "\U0001f44c"

    def test_refuses_function_words(self):
        assert emoji._resolve("the") is None
        assert emoji._resolve("of the") is None

    def test_unknown_words_resolve_to_nothing(self):
        assert emoji._resolve("qwertyuiop") is None

    def test_a_prefix_does_not_cross_a_word_boundary(self):
        """"fire" must not be free to match "fireworks"."""
        assert emoji._resolve("fire") == FIRE

    def test_every_alias_is_normalised_as_written(self):
        """A key that does not survive normalising could never be looked up."""
        assert all(key == emoji._normalise(key) for key in emoji.ALIASES)

    def test_every_alias_is_a_single_character(self):
        """A multi-character value would be a sequence we have not tested rendering."""
        assert all(len(value) == 1 for value in emoji.ALIASES.values())

    def test_the_index_is_built_once(self):
        emoji._index.cache_clear()
        first = emoji._index()
        assert first is emoji._index()
        assert len(first) > 1000  # the whole point: the long tail comes free
