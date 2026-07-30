import time

import pytest

from pywhispr.filler import (
    DEFAULT_FILLERS,
    compile_fillers,
    filler_words,
    is_deletion_only,
    remove_fillers,
)


def clean(text: str, extra=(), keep=()) -> str:
    return remove_fillers(text, filler_words(extra, keep))



class TestMidSentence:
    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("I think uh we should go.", "I think we should go."),
            ("I think, uh, we should go.", "I think we should go."),
            ("I think, um, we should go.", "I think we should go."),
            ("I think um, we should go.", "I think we should go."),
            ("I think erm we should go.", "I think we should go."),
            # A run of them goes in one piece, leaving one clean seam.
            ("I think um, uh, we should go.", "I think we should go."),
            ("I think um uh we should go.", "I think we should go."),
        ],
    )
    def test_no_double_spaces_or_stray_commas(self, said, wanted):
        assert clean(said) == wanted

    def test_a_comma_on_one_side_only_may_be_the_sentence_s_own(self):
        # Only a matched pair is taken as the filler's pauses. One comma is
        # left where it is: at worst that reads as a pause the speaker made.
        assert clean("In the end, um we should go.") == "In the end, we should go."

    def test_case_and_spelling_variants(self):
        assert clean("I think, Um, we should go.") == "I think we should go."
        assert clean("I think uhhh we should go.") == "I think we should go."

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            # A comma pair around a filler was the pause either side of it...
            ("I think, um, we should go.", "I think we should go."),
            ("I, um, think so.", "I think so."),
            ("He said, um, hello.", "He said hello."),
            ("It is, um, fine.", "It is fine."),
            ("And then, uh, I came home.", "And then I came home."),
            # The commonest hesitation site of all: after a determiner,
            # preposition, pronoun or adverb, where a comma cannot be structural.
            ("Pass me the, um, wrench.", "Pass me the wrench."),
            ("I want to, um, apologise.", "I want to apologise."),
            ("We, um, decided to leave.", "We decided to leave."),
            ("You, um, should go.", "You should go."),
            ("I just, um, wanted to say thanks.", "I just wanted to say thanks."),
            ("Send it to, um, Bob.", "Send it to Bob."),
            ("My, um, colleague disagrees.", "My colleague disagrees."),
            ("A, um, big problem.", "A big problem."),
            ("I like it, um, a lot.", "I like it a lot."),
            ("He asked if I would, um, help out.", "He asked if I would help out."),
            # ...but a comma doing work in the sentence keeps it.
            ("We need milk, um, eggs and bread.", "We need milk, eggs and bread."),
            ("You can have one, um, two or three.", "You can have one, two or three."),
            ("I like this, um, that and the other.", "I like this, that and the other."),
            ("We need plan A, um, plan B.", "We need plan A, plan B."),
            ("He said yes, um, then left.", "He said yes, then left."),
            ("If not, um, we stop.", "If not, we stop."),
            ("First, um, we prepare.", "First, we prepare."),
            ("On Monday, um, we ship.", "On Monday, we ship."),
            # A subordinate clause ends at that comma; dropping it would say
            # something else ("If you do tell me" is not "if you do, tell me").
            ("If you do, um, tell me.", "If you do, tell me."),
            ("If you can, um, come early.", "If you can, come early."),
            ("Whatever it is, um, we'll manage.", "Whatever it is, we'll manage."),
            ("While you wait, um, read this.", "While you wait, read this."),
            # ...even when a connective hides the subordinator.
            ("Even if you can, um, come early.", "Even if you can, come early."),
            ("And when you do, um, call me.", "And when you do, call me."),
            ("So if you can, um, join us.", "So if you can, join us."),
            # A comma splicing two clauses is structure, not a pause.
            ("I saw it, um, I left.", "I saw it, I left."),
            ("That's it, um, we're done.", "That's it, we're done."),
            ("We tried it, um, we failed.", "We tried it, we failed."),
            # A lone capital is a label; sentence-initially it is the article.
            ("We need plan A, um, plan B.", "We need plan A, plan B."),
            ("John F, um, Kennedy.", "John F, Kennedy."),
        ],
    )
    def test_a_comma_survives_where_it_is_doing_work(self, said, wanted):
        assert clean(said) == wanted

    def test_a_colon_is_the_sentence_s_own(self):
        assert clean("Fine um: also this.") == "Fine: also this."
        assert clean("Fine um; also this.") == "Fine; also this."


class TestSentenceStart:
    def test_capital_is_handed_to_the_next_word(self):
        assert clean("Um, so I think so.") == "So I think so."
        assert clean("Uh I think so.") == "I think so."

    def test_after_a_full_stop_too(self):
        assert clean("I went. Um, then I came back.") == "I went. Then I came back."

    def test_lowercase_input_stays_lowercase(self):
        # Nothing here knows the user's style, so only a capital is handed on.
        assert clean("um, so i think so.") == "so i think so."

    def test_a_capitalised_word_is_left_alone(self):
        assert clean("Um, Ben said so.") == "Ben said so."

    def test_after_a_newline(self):
        assert clean("First line.\num, second line.") == "First line.\nsecond line."
        assert clean("First line.\nUm, second line.") == "First line.\nSecond line."
        assert clean("First line.\r\nUm, second line.") == "First line.\r\nSecond line."

    def test_after_a_quote_or_bracket(self):
        assert clean('"Um, go home," he said.') == '"Go home," he said.'
        assert clean("(Um, maybe not.)") == "(Maybe not.)"

    def test_after_a_bullet(self):
        assert clean("- Um, bullet.") == "- Bullet."
        assert clean("1. Um, bullet.") == "1. Bullet."

    def test_when_the_filler_was_its_own_sentence(self):
        assert clean("Um... yes.") == "Yes."
        assert clean("Hello. Um... right.") == "Hello. Right."


class TestSentenceEnd:
    def test_takes_the_comma_in_front_of_it(self):
        assert clean("I think so, um.") == "I think so."
        assert clean("I think so, uh!") == "I think so!"

    def test_at_the_very_end_without_punctuation(self):
        assert clean("I think so, um") == "I think so"
        assert clean("I think so um") == "I think so"

    def test_before_a_newline(self):
        assert clean("I think so, um\nNext line.") == "I think so\nNext line."

    def test_a_filler_that_was_its_own_sentence(self):
        assert clean("Hello. Um. Right then.") == "Hello. Right then."
        assert clean("Hello. Um, uh. Right then.") == "Hello. Right then."
        assert clean("Hello. Um.") == "Hello."
        assert clean("Um. Hello.") == "Hello."

    def test_inside_a_bracket_or_quote(self):
        """The pair was only holding the filler apart, so it goes too."""
        assert clean("(um) yes.") == "yes."
        assert clean("I think (um) we should go.") == "I think we should go."
        assert clean("I think(um)we should go.") == "I think we should go."
        assert clean('He said "um".') == "He said."
        assert clean('He said "go, um".') == 'He said "go".'
        assert clean("I think so, um)") == "I think so)"

    def test_between_dashes_one_dash_stays(self):
        """The dashes are the sentence's punctuation, not the filler's."""
        assert clean("I think — um — we should go.") == "I think — we should go."
        assert clean("wait—um—no.") == "wait—no."
        # Two separate parentheticals must not be merged into one.
        assert clean("I like it — really — um — truly — yes.") == (
            "I like it — really — truly — yes."
        )

    def test_a_mark_that_was_closing_something_is_not_a_pair(self):
        """Straight quotes are both halves of their pair, so words can't be fused."""
        assert clean('He said "go" um "now".') == 'He said "go" "now".'
        assert clean("the dogs' um 'til dawn.") == "the dogs' 'til dawn."
        assert clean('He said “go” um “now”.') == 'He said “go” “now”.'


class TestWhitespaceDebris:
    """A deleted filler must not leave a blank line or a ragged edge behind."""

    def test_at_the_start_of_the_dictation(self):
        assert clean("Um\nNew line.") == "New line."
        assert clean("Um, uh.\nHello.") == "Hello."
        assert clean("   Um, hello.") == "Hello."

    def test_on_a_line_of_its_own(self):
        assert clean("Line one.\nUm.\nLine three.") == "Line one.\nLine three."
        assert clean("Line one.\r\nUm.\r\nLine three.") == "Line one.\r\nLine three."

    def test_no_stray_mark_where_the_dictation_started(self):
        assert clean("Um, um: weird.") == "Weird."
        assert clean("Um; hello.") == "Hello."

    def test_at_the_end_of_the_dictation(self):
        assert clean("I think so.\nUm.") == "I think so."
        assert clean("Trailing spaces um   ") == "Trailing spaces"


class TestNothingButFiller:
    @pytest.mark.parametrize("said", ["Um.", "Um", "  Uh...  ", "Um, uh, erm."])
    def test_yields_nothing_to_insert(self, said):
        assert clean(said) == ""

    def test_a_real_word_survives(self):
        assert clean("Um. Yes.") == "Yes."

    def test_text_with_nothing_to_remove_is_left_alone(self):
        # Blanking is for what filler removal emptied, not for text that never
        # said anything in the first place.
        assert clean("...") == "..."
        assert clean("") == ""
        assert clean("   ") == "   "


class TestLeavesTextAlone:
    @pytest.mark.parametrize(
        "said",
        [
            "I think we should go.",
            # Not fillers: "uh-huh" means yes, and these are all ordinary words.
            "Uh-huh, that's right.",
            "It's a big number.",
            "Hummus and jam.",
            "The summer term.",
            "Err on the side of caution.",
            "Ah, I see. Oh well. Hmm.",
            # Acronyms are not hesitations: ER, UM and ERM are real things.
            "Give me the ERM report.",
            "The ER department, and the UM system.",
            # The other side of that trade: shouted dictation keeps its filler.
            "UM, SO I THINK SO.",
            # Written with a space, "uh huh" still means yes and "uh oh" trouble.
            "Uh huh, sure.",
            "Uh oh, that broke.",
            "I think uh huh means yes.",
            "I like it, right, so we ship it.",
            "Turn right, then left.",
            "It's fine, right?",
            # A filler spelling inside a longer word.
            "The umbrella is uhhuh nonsense.",
            "Mermaids, Bermuda, permanent.",
        ],
    )
    def test_unchanged(self, said):
        assert clean(said) == said

    def test_a_phrase_only_matches_when_adjacent(self):
        assert clean("You surely know that.", extra=["you know"]) == "You surely know that."

    def test_a_comma_between_phrase_words_is_not_the_phrase(self):
        assert clean("You, know, that.", extra=["you know"]) == "You, know, that."


class TestConfiguration:
    def test_extra_words(self):
        assert clean("It is, you know, fine.", extra=["you know"]) == "It is fine."
        assert clean("It is, like, fine.", extra=["like"]) == "It is fine."
        assert clean("Hmm, maybe.", extra=["hmm"]) == "Maybe."

    def test_keeping_a_builtin(self):
        assert clean("I think, er, so.", keep=["er"]) == "I think, er, so."
        # The others still go.
        assert clean("I think, um, so.", keep=["er"]) == "I think so."

    def test_an_empty_list_changes_nothing(self):
        assert remove_fillers("I think, um, so.", frozenset()) == "I think, um, so."

    def test_entries_are_normalised(self):
        assert filler_words(extra=["  You  Know  "]) == filler_words(extra=["you know"])
        assert compile_fillers(["", "  ", "!!"]) == frozenset()

    def test_keep_removes_every_spelling_it_names(self):
        phrases = filler_words(keep=["Um"])
        assert ("um",) not in phrases
        assert ("uh",) in phrases


class TestIsDeletionOnly:
    def test_accepts_deletions_and_recapitalisation(self):
        assert is_deletion_only("Um, so I think.", "So I think.")
        assert is_deletion_only("anything", "")
        assert is_deletion_only("anything", "anything")

    def test_rejects_invented_or_reordered_text(self):
        assert not is_deletion_only("Um, so I think.", "So I thought.")
        assert not is_deletion_only("so I think", "I think so")
        assert not is_deletion_only("hello", "hello world")

    def test_every_default_removal_satisfies_it(self):
        for filler in DEFAULT_FILLERS:
            said = f"I think, {filler}, we should go."
            assert is_deletion_only(said, clean(said))


def test_stays_fast_on_a_long_transcript():
    """The comma gate looks backwards for the clause it is in, so it must not
    rescan the whole transcript every time it is asked."""
    said = "I think, um, we should go, and then, uh, we came back. " * 2000
    started = time.perf_counter()
    cleaned = clean(said)
    assert time.perf_counter() - started < 1.0
    assert "um" not in cleaned and "uh" not in cleaned


def test_long_transcript_keeps_its_shape():
    said = (
        "Um, so the thing is, uh, I went to the shop, um, and then, uh, "
        "I came home. Erm. It was fine, um."
    )
    assert clean(said) == (
        "So the thing is I went to the shop, and then I came home. It was fine."
    )
