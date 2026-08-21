import pytest

from pywhispr.acronyms import (
    AcronymResult,
    Replacement,
    is_acronym_substitution,
    to_acronyms,
)


def acronyms(text: str) -> str:
    return to_acronyms(text).text


class TestCodes:
    """Letters followed by digits, hyphenated — the case this exists for.

    The inputs are what the model really writes, measured: it glues spelled
    letters into an acronym by itself, and the numbers pass has already run.
    """

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("EPM 1180", "EPM-1180"),
            ("Assign EPM 1180 to me please.", "Assign EPM-1180 to me please."),
            ("The ticket is EPM 54321.", "The ticket is EPM-54321."),
            ("Please look at ticket EPM 9149.", "Please look at ticket EPM-9149."),
            # No space at all: the model writes the code as one word.
            ("I need the value of ABC123.", "I need the value of ABC-123."),
            ("ABC123", "ABC-123"),
            # Said with pauses, so the letters arrive spaced too.
            ("P M 9149.", "PM-9149."),
            ("Look at C V E 202512345.", "Look at CVE-202512345."),
            ("A B C 123", "ABC-123"),
        ],
    )
    def test_hyphenates(self, said, wanted):
        assert acronyms(said) == wanted


class TestLetterRuns:
    """A sequence of letters becomes one token."""

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("A B C", "ABC"),
            ("E P M", "EPM"),
            ("U S A today", "USA today"),
            ("Look at C V E now.", "Look at CVE now."),
        ],
    )
    def test_joins(self, said, wanted):
        assert acronyms(said) == wanted

    @pytest.mark.parametrize(
        "said",
        [
            # Two is not evidence of a spelling: the model glues sentences, so
            # "Grade A. B is worse." really does arrive like this.
            "Grade A B is worse. C is worse.",
            "Section A B",
            # A multi-letter group only ever stands alone — these are two
            # acronyms with a space between them.
            "The AB CD test passed.",
            "My initials are BS and I work on EPM.",
            "Send it to HR.",
        ],
    )
    def test_untouched(self, said):
        assert acronyms(said) == said


class TestInterleaved:
    """Letters and digits mixed are one sequence with no whitespace.

    Almost always a no-op: this is what the model writes already. The rule is
    what stops the two-group hyphen arm reaching a code that has more to it.
    """

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("The file is A23BC234.", "The file is A23BC234."),
            ("The code is A1B2C3.", "The code is A1B2C3."),
            ("The machine name is WIN11EPM Build.", "The machine name is WIN11EPM Build."),
            ("A 23 BC 234", "A23BC234"),
        ],
    )
    def test_concatenates(self, said, wanted):
        assert acronyms(said) == wanted


class TestPunctuationEndsARun:
    """Only spacing holds a run together, which is what saves a list."""

    @pytest.mark.parametrize(
        "said",
        [
            "Do you prefer option A, B, or C?",
            "The answer is A, B, or C.",
            "John F. Kennedy Airport.",
            "Do you want plan A or plan B?",
            # Already hyphenated, so there is nothing to join: the gap is a
            # hyphen, not spacing. This is also what makes the pass idempotent.
            "Raise it under EPM-12345.",
        ],
    )
    def test_untouched(self, said):
        assert acronyms(said) == said


class TestLowerCaseIsProse:
    """Capitals are what tell a spelled letter from an ordinary word."""

    @pytest.mark.parametrize(
        "said",
        [
            "The Windows 11 build is broken.",
            "Let us meet at 5 pm in the office.",
            "It is a 12-month contract.",
            # The model writes "help" in lower case because it is a word; a
            # `help => HELP` vocabulary entry is how to reach HELP-4567.
            "I filed it under help 4567.",
            "The USB three port is dead.",
            "It is iOS 18 now.",
        ],
    )
    def test_untouched(self, said):
        assert acronyms(said) == said


class TestThresholds:
    """A code's number has more than one digit; a lone letter is not an acronym."""

    @pytest.mark.parametrize(
        "said",
        [
            "I sent it to IT 5 times.",
            "A 12 month contract.",
            "Section B 12 is next.",
            # Digits then letters is prose, however many of each.
            "at 5 PM sharp",
            "the 3 PCs are here",
        ],
    )
    def test_untouched(self, said):
        assert acronyms(said) == said


class TestNewlinesAreStructure:
    """Only horizontal spacing joins, so a line break cannot be swallowed."""

    def test_a_newline_ends_a_run(self):
        assert acronyms("EPM\n1180") == "EPM\n1180"

    def test_a_non_breaking_space_joins(self):
        # An escape, not the character: written literally, a reformat can quietly
        # turn this into an ordinary space and the test still passes.
        assert acronyms("EPM\xa01180") == "EPM-1180"


class TestTheTripwire:
    """Whatever the scanner does, it can only have taken spacing out."""

    def test_accepts_a_real_conversion(self):
        said = "Assign EPM 1180 to me."
        assert is_acronym_substitution(said, to_acronyms(said))

    def test_accepts_a_no_op(self):
        said = "Send it to HR."
        assert is_acronym_substitution(said, AcronymResult(said))

    def test_rejects_text_the_spans_do_not_account_for(self):
        assert not is_acronym_substitution("EPM 1180", AcronymResult("Something else.", ()))

    def test_rejects_a_span_over_something_that_is_not_a_code(self):
        said = "I have five apples"
        claimed = AcronymResult("I have five-apples", (Replacement(11, 18, "five-apples"),))
        assert not is_acronym_substitution(said, claimed)

    def test_rejects_a_replacement_that_lost_a_character(self):
        said = "EPM 1180"
        claimed = AcronymResult("EPM-118", (Replacement(0, 8, "EPM-118"),))
        assert not is_acronym_substitution(said, claimed)

    def test_rejects_a_replacement_that_re_cased_a_character(self):
        said = "EPM 1180"
        claimed = AcronymResult("Epm-1180", (Replacement(0, 8, "Epm-1180"),))
        assert not is_acronym_substitution(said, claimed)

    def test_rejects_a_second_hyphen(self):
        said = "CVE 202512345"
        claimed = AcronymResult("CVE-2025-12345", (Replacement(0, 13, "CVE-2025-12345"),))
        assert not is_acronym_substitution(said, claimed)

    def test_rejects_spans_out_of_order_or_overlapping(self):
        said = "A B C and D E F"
        result = AcronymResult(
            "ABC and DEF",
            (Replacement(10, 15, "DEF"), Replacement(0, 5, "ABC")),
        )
        assert not is_acronym_substitution(said, result)

    def test_rejects_a_span_outside_the_text(self):
        claimed = AcronymResult("ABC", (Replacement(0, 99, "ABC"),))
        assert not is_acronym_substitution("A B C", claimed)

    def test_rejects_a_span_that_swallowed_punctuation(self):
        said = "A, B, C"
        claimed = AcronymResult("ABC", (Replacement(0, 7, "ABC"),))
        assert not is_acronym_substitution(said, claimed)


class TestIdempotence:
    @pytest.mark.parametrize(
        "said",
        [
            "Assign EPM 1180 to me please.",
            "I need the value of ABC123.",
            "A B C",
            "Look at C V E 202512345.",
            "The answer is A, B, or C.",
            "The file is A23BC234.",
            "Raise it under EPM-12345.",
        ],
    )
    def test_a_second_pass_changes_nothing(self, said):
        once = acronyms(said)
        assert acronyms(once) == once


def test_empty_text():
    assert to_acronyms("") == AcronymResult("")
