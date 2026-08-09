import pytest

from pywhispr.config import Config
from pywhispr.scratch import compile_reset_phrases, is_suffix_of, strip_before_reset


@pytest.fixture
def pattern():
    """The shipped defaults: doubled commands."""
    return compile_reset_phrases(Config().voice_reset_phrases)


class TestFalsePositives:
    """The words used as words must survive, whatever surrounds them."""

    @pytest.mark.parametrize(
        "text",
        [
            "I can scratch that surface.",
            "We scratch scratch marks off the paint.",  # doubled, but mid-clause
            "Reset reset handling is what this ticket is about.",
            "The scratch, scratch of the pen went on all night.",  # words either side
            "Ask them to reset.",
        ],
    )
    def test_mid_sentence_never_triggers(self, pattern, text):
        assert strip_before_reset(text, pattern) == text


class TestStandaloneSegment:
    def test_own_sentence_triggers(self, pattern):
        assert strip_before_reset(
            "Book the room. Scratch scratch. Book the hall.", pattern
        ) == "Book the hall."

    def test_own_clause_triggers(self, pattern):
        assert strip_before_reset(
            "Book the room, scratch scratch, book the hall.", pattern
        ) == "Book the hall."

    def test_opening_the_transcript_triggers(self, pattern):
        assert strip_before_reset("Scratch scratch, book the hall.", pattern) == "Book the hall."

    def test_whisper_may_punctuate_between_the_halves(self, pattern):
        assert strip_before_reset("Book the room. Scratch, scratch. Yes.", pattern) == "Yes."

    def test_on_its_own_line_triggers(self, pattern):
        assert strip_before_reset("Book the room.\nreset reset\nBook the hall.", pattern) == (
            "Book the hall."
        )

    def test_nothing_after_the_phrase_discards_everything(self, pattern):
        assert strip_before_reset("Send it at four. Scratch scratch.", pattern) == ""


class TestLastOccurrenceWins:
    def test_only_text_after_the_last_trigger_survives(self, pattern):
        text = "One. Scratch scratch. Two. Reset reset. Three."
        assert strip_before_reset(text, pattern) == "Three."

    def test_an_earlier_innocent_use_does_not_move_the_cut(self, pattern):
        text = "Do not scratch that surface. Scratch scratch. Paint it."
        assert strip_before_reset(text, pattern) == "Paint it."


class TestConfiguration:
    def test_a_single_word_marker_still_needs_its_own_segment(self):
        pattern = compile_reset_phrases(["banana"])
        assert strip_before_reset("I ate a banana yesterday.", pattern) == (
            "I ate a banana yesterday."
        )
        assert strip_before_reset("Buy apples. Banana. Buy pears.", pattern) == "Buy pears."

    def test_an_undoubled_phrase_is_allowed(self):
        pattern = compile_reset_phrases(["scratch that"])
        assert strip_before_reset("Book the room. Scratch that. Book the hall.", pattern) == (
            "Book the hall."
        )

    def test_no_phrases_configured_is_a_no_op(self):
        assert compile_reset_phrases([]) is None
        assert compile_reset_phrases([" "]) is None
        assert strip_before_reset("Scratch scratch. Hello.", None) == "Scratch scratch. Hello."


class TestSuffixInvariant:
    def test_result_is_always_a_suffix(self, pattern):
        for text in ("Scratch scratch, hello.", "nothing here", "reset reset"):
            assert is_suffix_of(text, strip_before_reset(text, pattern))

    def test_is_suffix_of_rejects_an_edit(self):
        assert not is_suffix_of("book the hall", "book the room")
