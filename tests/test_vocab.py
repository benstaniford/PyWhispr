import pytest

from pywhispr.vocab import (
    MIN_FUZZY_LENGTH,
    TEMPLATE,
    apply_vocabulary,
    count_entries,
    load_vocabulary,
    load_vocabulary_text,
    normalise,
    parse_vocabulary,
    save_vocabulary_text,
)

VOCAB = """
# A comment, and a blank line above it.
BeyondTrust
PyWhispr
Kubernetes
Jamf
Microsoft Defender for Endpoint
pie whisper => PyWhispr
c sharp => C#
"""

RULES = parse_vocabulary(VOCAB)


def apply(text, vocab=VOCAB, **kwargs):
    return apply_vocabulary(text, parse_vocabulary(vocab), **kwargs)


# (transcript, expected) — one row per rule this feature exists to enforce.
CASES = [
    # Tier 1: the model heard it, then wrote it as ordinary English.
    ("I work at Beyond Trust.", "I work at BeyondTrust."),
    ("I work at beyond trust.", "I work at BeyondTrust."),
    ("I work at BEYOND TRUST.", "I work at BeyondTrust."),
    ("I work at Beyondtrust.", "I work at BeyondTrust."),
    ("A beyond-trust problem.", "A BeyondTrust problem."),
    # Already right: untouched, and not doubled up.
    ("I work at BeyondTrust.", "I work at BeyondTrust."),
    # Tier 2: a near miss on a long enough term.
    ("We run cubernetes here.", "We run Kubernetes here."),
    ("We run kubernets here.", "We run Kubernetes here."),
    # Too far away to be a near miss.
    ("We run docker here.", "We run docker here."),
    # Short terms are exact-only, so "jam" survives next to a listed "Jamf".
    ("I like jam on toast.", "I like jam on toast."),
    ("Pushed by jamf overnight.", "Pushed by Jamf overnight."),
    # An explicit fix, matched exactly whatever its length.
    ("Open pie whisper please.", "Open PyWhispr please."),
    ("Open Pie Whisper please.", "Open PyWhispr please."),
    # Longest phrase wins over any shorter match inside it.
    (
        "Deploy microsoft defender for endpoint today.",
        "Deploy Microsoft Defender for Endpoint today.",
    ),
    # Punctuation and line breaks are not word separators, so a phrase cannot
    # be assembled across them.
    ("Beyond, trust me.", "Beyond, trust me."),
    ("Beyond\ntrust me.", "Beyond\ntrust me."),
    # A near match must never swallow a neighbouring word: "a cubernetes" is
    # two edits from "kubernetes", and the "a" is not ours to delete.
    ("A cubernetes problem.", "A Kubernetes problem."),
    # Punctuation the model can't type comes back via an explicit fix, and the
    # spelling it produces on its own is left alone.
    ("I write c sharp.", "I write C#."),
    ("I write C# daily.", "I write C# daily."),
    # An inflected form is left alone rather than dragged back to the term.
    ("Two kubernetes clusters.", "Two Kubernetes clusters."),
    ("Lots of beyondtrusters here.", "Lots of beyondtrusters here."),
    # Sentence-initial: the wanted spelling wins over the model's capital.
    ("Beyond trust is hiring.", "BeyondTrust is hiring."),
    # Nothing to do.
    ("", ""),
    ("Just an ordinary sentence.", "Just an ordinary sentence."),
    ("...", "..."),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_vocabulary_rules(text, expected):
    assert apply(text) == expected


def test_multiple_corrections_in_one_transcript():
    assert apply("Beyond trust runs cubernetes on pie whisper.") == (
        "BeyondTrust runs Kubernetes on PyWhispr."
    )


def test_fuzzy_can_be_switched_off():
    assert apply("We run cubernetes here.", fuzzy=False) == "We run cubernetes here."
    # Exact matching still applies.
    assert apply("We run kubernetes here.", fuzzy=False) == "We run Kubernetes here."


def test_no_rules_returns_the_text_untouched():
    assert apply_vocabulary("Beyond trust.", []) == "Beyond trust."


def test_ambiguous_near_matches_are_left_alone():
    """Two terms an equal distance away: pick neither, and say nothing."""
    vocab = "Kubernetos\nKubernetes"
    assert apply("We run kubernetzs here.", vocab) == "We run kubernetzs here."
    # One clearly closer than the other is not ambiguous, so it still applies.
    assert apply("We run kubernetes here.", vocab) == "We run Kubernetes here."


def test_common_words_are_never_near_matched():
    """A term one edit from a common joining word must not eat it."""
    assert apply("Between us.", "Betweed") == "Between us."
    assert apply("Because of that.", "Becuase") == "Because of that."


def test_a_near_match_cannot_cross_a_word_boundary():
    """Word counts must agree, so an edit is always inside a word."""
    assert apply("An unrelated word.", "Unrelatedwordy") == "An unrelated word."


def test_spacing_and_punctuation_are_preserved_around_a_match():
    assert apply("  (beyond trust),  really") == "  (BeyondTrust),  really"
    assert apply("beyond trust\tand more") == "BeyondTrust\tand more"


class TestParsing:
    def test_comments_and_blanks_are_skipped(self):
        assert [rule.wanted for rule in parse_vocabulary("# note\n\n  \nOne\n")] == ["One"]

    def test_only_a_leading_hash_is_a_comment(self):
        assert [rule.wanted for rule in parse_vocabulary("C# tooling\n")] == ["C# tooling"]

    def test_terms_that_normalise_too_short_are_dropped(self):
        """"C#" reduces to "c", which would rewrite every stray letter c."""
        assert parse_vocabulary("C#\nX\n") == []
        # The explicit form gives it something real to match on.
        assert [rule.wanted for rule in parse_vocabulary("c sharp => C#")] == ["C#"]

    def test_explicit_replacement(self):
        (rule,) = parse_vocabulary("pie whisper => PyWhispr")
        assert (rule.key, rule.wanted, rule.words) == ("piewhisper", "PyWhispr", 2)
        assert not rule.fuzzy  # an explicit fix is exact by definition

    def test_fuzzy_is_gated_on_length(self):
        short, long_ = parse_vocabulary("Jamf\nKubernetes")
        assert not short.fuzzy
        assert long_.fuzzy
        assert len(short.key) < MIN_FUZZY_LENGTH <= len(long_.key)

    def test_unusable_lines_are_dropped(self):
        assert parse_vocabulary("...\n=> nothing\nsomething =>\n") == []

    def test_duplicates_keep_the_first(self):
        rules = parse_vocabulary("BeyondTrust\nbeyond trust => Something Else")
        assert [rule.wanted for rule in rules] == ["BeyondTrust"]

    def test_the_template_defines_no_rules(self):
        assert parse_vocabulary(TEMPLATE) == []

    def test_count_entries(self):
        assert count_entries(VOCAB) == (7, 0)
        assert count_entries("Good\n...\n# ignored comment\n") == (1, 1)
        assert count_entries(TEMPLATE) == (0, 0)


class TestNormalise:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("BeyondTrust", "beyondtrust"),
            ("Beyond Trust", "beyondtrust"),
            ("beyond-trust", "beyondtrust"),
            ("C#", "c"),
            ("GPT-4", "gpt4"),
            ("...", ""),
        ],
    )
    def test_normalise(self, word, expected):
        assert normalise(word) == expected


class TestStorage:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "vocabulary.txt"
        save_vocabulary_text("BeyondTrust", path)
        assert load_vocabulary_text(path) == "BeyondTrust\n"
        assert [rule.wanted for rule in load_vocabulary(path)] == ["BeyondTrust"]

    def test_missing_file_is_not_an_error(self, tmp_path):
        path = tmp_path / "nope" / "vocabulary.txt"
        assert load_vocabulary_text(path) == ""
        assert load_vocabulary(path) == []

    def test_save_creates_the_directory(self, tmp_path):
        path = tmp_path / "made" / "up" / "vocabulary.txt"
        save_vocabulary_text("Kubernetes\n", path)
        assert path.read_text(encoding="utf-8") == "Kubernetes\n"

    def test_unreadable_file_costs_the_vocabulary_not_the_dictation(self, tmp_path):
        path = tmp_path / "vocabulary.txt"
        path.mkdir()  # a directory where a file should be: read raises OSError
        assert load_vocabulary_text(path) == ""
        assert load_vocabulary(path) == []


ADVERSARIAL_TEXT = [
    "",
    " ",
    "\n",
    "!!!",
    "42",
    "🙂",
    "beyond",
    "trust",
    "beyond trust beyond trust",
    "BeyondTrustBeyondTrust",
    "a-b-c-d-e-f",
    "'",
    "beyond  trust",
    "beyond trust",
]


@pytest.mark.parametrize("text", ADVERSARIAL_TEXT)
def test_never_destroys_the_transcript(text):
    """Whatever the input, the output must still be recognisably the same text:
    every non-matching word survives, in order, with its spacing."""
    out = apply(text)
    assert out.strip() or not text.strip()
    # Nothing the vocabulary does may drop half the transcript on the floor.
    assert len(out) >= len(text) - len("beyond trust") * text.count("beyond")


@pytest.mark.parametrize("text", ADVERSARIAL_TEXT)
def test_is_idempotent(text):
    once = apply(text)
    assert apply(once) == once
