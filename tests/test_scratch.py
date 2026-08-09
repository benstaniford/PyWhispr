import pytest

from pywhispr.scratch import compile_reset_phrases, is_suffix_of, strip_before_reset

PHRASES = ["scratch that", "start over"]


@pytest.fixture
def pattern():
    return compile_reset_phrases(PHRASES)


def test_keeps_only_what_follows_the_phrase(pattern):
    assert strip_before_reset("Book the room. Scratch that, book the hall.", pattern) == (
        "Book the hall."
    )


def test_keeps_only_the_last_of_several(pattern):
    text = "One. Scratch that, two. Start over, three."
    assert strip_before_reset(text, pattern) == "Three."


def test_phrase_may_be_split_by_punctuation(pattern):
    assert strip_before_reset("No. Scratch that. Yes please.", pattern) == "Yes please."


def test_nothing_after_the_phrase_discards_everything(pattern):
    assert strip_before_reset("Send it at four. Scratch that.", pattern) == ""


def test_text_without_a_phrase_is_untouched(pattern):
    text = "Scratching a record starts nothing over."
    assert strip_before_reset(text, pattern) == text


def test_word_boundaries_are_respected(pattern):
    text = "Run the startover script."
    assert strip_before_reset(text, pattern) == text


def test_no_phrases_configured_is_a_no_op():
    assert compile_reset_phrases([]) is None
    assert compile_reset_phrases([" "]) is None
    assert strip_before_reset("Scratch that, hello.", None) == "Scratch that, hello."


def test_result_is_always_a_suffix(pattern):
    for text in ("Scratch that, hello.", "nothing here", "start over"):
        assert is_suffix_of(text, strip_before_reset(text, pattern))


def test_is_suffix_of_rejects_an_edit():
    assert not is_suffix_of("book the hall", "book the room")
