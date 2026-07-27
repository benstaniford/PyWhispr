import pytest

from pywhispr.join import (
    CONTINUATION_WORDS,
    is_mid_sentence,
    join_text,
    lowercase_first_word,
    needs_leading_space,
)

# (preceding, text, expected) — one case per row of the rule tables.
CASES = [
    # No context at all: paste verbatim.
    (None, "Then I came home.", "Then I came home."),
    ("", "Then I came home.", "Then I came home."),
    # The two failures this feature exists to fix.
    ("I went to the shop.", "Then I came home.", " Then I came home."),
    ("I went to the shop", "Then I came home.", " then I came home."),
    # A full stop is deliberate, so the capital after it stands.
    ("I went to the shop.", "And then came home.", " And then came home."),
    ("He said it was fine!", "But it wasn't.", " But it wasn't."),
    ("Wait, really?", "Then we left.", " Then we left."),
    # Closers are stripped before looking for the sentence end.
    ('He said, "It\'s fine."', "Then we left.", " Then we left."),
    ("(That was done.)", "Then we left.", " Then we left."),
    # Mid-sentence after a comma or colon.
    ("I went to the shop,", "Then I came home.", " then I came home."),
    ("Bring these:", "And also milk.", " and also milk."),
    # Already separated — must not double up.
    ("I went to the shop. ", "Then I came home.", "Then I came home."),
    ("I went to the shop.\t", "Then I came home.", "Then I came home."),
    # Trailing whitespace stops the space but not the lower-casing: a
    # hand-typed "I went to the shop " really is mid-sentence.
    ("shop ", "Then I came home.", "then I came home."),
    ("shop\u00a0", "Then I came home.", "then I came home."),
    # A line break starts a fresh sentence: no space, no lower-casing.
    ("I went to the shop\n", "Then I came home.", "Then I came home."),
    ("shop\r", "Then I came home.", "Then I came home."),
    # A bare list marker starts a sentence too.
    ("Shopping list:\n- ", "And milk.", "And milk."),
    ("Steps:\n1. ", "Then stir.", "Then stir."),
    # Glue and openers keep the text tight against what precedes it.
    ("well-", "Or so.", "or so."),
    ("and/", "Or something.", "or something."),
    # The accepted gap: a word outside CONTINUATION_WORDS keeps its capital,
    # which is exactly what happens without this feature.
    ("well-", "Known issue.", "Known issue."),
    ("I went to the shop", "Walked home after.", " Walked home after."),
    ("mail to foo@", "Bar.com", "Bar.com"),
    ("The result (", "And so on.", "and so on."),
    ("He said “", "And so on.", "and so on."),
    # Trailing straight quote: closing wants a space, opening does not.
    ("the dogs'", "Then we left.", " then we left."),
    ("he said '", "Then we left.", "then we left."),
    # Punctuation-first text hugs the previous word.
    ("I went to the shop", ", didn't I?", ", didn't I?"),
    ("I went to the shop", "...anyway.", "...anyway."),
    ("the dog", "'s dinner", "'s dinner"),
    # Text that brings its own separator keeps it, and gains nothing.
    ("I went to the shop", " then I came home.", " then I came home."),
    # Proper nouns are never touched, punctuated or not.
    ("I spoke to", "Ben said hello.", " Ben said hello."),
    ("We met in", "March last year.", " March last year."),
    ("It might", "Will do it.", " Will do it."),
    # "I" survives because one-letter words can never be lower-cased.
    ("Dear Ben,", "I hope you're well.", " I hope you're well."),
    ("I think", "I'll go.", " I'll go."),
    # Acronyms and camel case fail the shape gate.
    ("I checked the", "API returned 500.", " API returned 500."),
    ("the class", "AndThen is odd.", " AndThen is odd."),
    ("we use", "IT support.", " IT support."),
    # Digits either side.
    ("In 1999", "The war ended.", " the war ended."),
    ("costs", "20 pounds.", " 20 pounds."),
]


@pytest.mark.parametrize("preceding,text,expected", CASES)
def test_join_rules(preceding, text, expected):
    assert join_text(preceding, text) == expected


def test_lowercase_can_be_switched_off():
    assert join_text("I went to the shop", "Then I came home.", lowercase_continuations=False) == (
        " Then I came home."
    )


def test_empty_text_is_returned_untouched():
    assert join_text("I went to the shop.", "") == ""


ADVERSARIAL = [
    None,
    "",
    "   ",
    "\n",
    "\r\n",
    "(",
    "-",
    "…",
    'said."',
    "- ",
    " ",
    "'",
    '"',
    "42",
    "!!!",
    "%",
    "🙂",
]


@pytest.mark.parametrize("preceding", ADVERSARIAL)
@pytest.mark.parametrize("text", ["Then I came home.", "and so on", "A", ".", " x"])
def test_never_loses_text(preceding, text):
    """The contract the caller relies on: at most one extra leading character,
    and everything from text[1] onwards byte-identical."""
    joined = join_text(preceding, text)
    assert joined.endswith(text[1:])
    assert 0 <= len(joined) - len(text) <= 1
    assert joined in (text, " " + text, text[0].lower() + text[1:], " " + text[0].lower() + text[1:])


def test_continuation_words_are_sane():
    assert all(word.islower() for word in CONTINUATION_WORDS)
    assert all(len(word) > 1 for word in CONTINUATION_WORDS)
    assert "i" not in CONTINUATION_WORDS
    # Names and months masquerading as function words.
    for homograph in ("will", "may", "march", "august", "june", "july", "mark", "grant", "bill"):
        assert homograph not in CONTINUATION_WORDS


def test_needs_leading_space_edges():
    assert not needs_leading_space("", "text")
    assert not needs_leading_space("shop", "")
    assert needs_leading_space("shop", "then")


def test_is_mid_sentence():
    assert is_mid_sentence("I went to the shop")
    assert not is_mid_sentence("I went to the shop.")
    assert not is_mid_sentence("")
    assert not is_mid_sentence("   ")


def test_lowercase_first_word_ignores_non_words():
    assert lowercase_first_word("...anyway") == "...anyway"
    assert lowercase_first_word("") == ""
    assert lowercase_first_word("42 things") == "42 things"
