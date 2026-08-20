import pytest

from pywhispr.numbers import (
    NumberResult,
    Replacement,
    is_digit_substitution,
    replaces_only_number_words,
    spoken_value,
    to_digits,
)


def digits(text: str) -> str:
    return to_digits(text).text


class TestDigitRuns:
    """Numbers read out one digit at a time — the case this exists for."""

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("one one eight zero", "1180"),
            ("call me on one one eight zero", "call me on 1180"),
            ("one two", "12"),
            ("one two three", "123"),
            ("zero one", "01"),
            ("ONE TWO", "12"),
            # A number the model already wrote as digits ends the run without
            # disturbing it.
            ("EPM 1234 one one eight zero", "EPM 1234 1180"),
        ],
    )
    def test_concatenates(self, said, wanted):
        assert digits(said) == wanted


class TestCompoundNumerals:
    """Words that compose into one number, arithmetic and all."""

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("twenty five", "25"),
            ("twenty five people", "25 people"),
            ("one hundred and eighty", "180"),
            ("five hundred and six", "506"),
            ("two thousand and twenty four", "2024"),
            ("two thousand twenty four", "2024"),
            ("two hundred and fifty thousand", "250000"),
            ("one million two hundred thousand", "1200000"),
            # "hundred" multiplies within a chunk; a bigger scale closes one.
            ("one hundred thousand two hundred", "100200"),
            ("nineteen hundred eighty four", "1984"),
        ],
    )
    def test_composes(self, said, wanted):
        assert digits(said) == wanted

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            # Two numbers that cannot compose are two groups, concatenated —
            # which is exactly how a year sounds.
            ("twenty twenty", "2020"),
            ("nineteen eighty four", "1984"),
            ("ten ten", "1010"),
        ],
    )
    def test_a_word_that_cannot_extend_starts_a_new_group(self, said, wanted):
        assert digits(said) == wanted


class TestLoneNumbers:
    """One number word is not a number sequence. The false-trigger guard."""

    @pytest.mark.parametrize(
        "said",
        [
            "I have five apples",
            "one of the reasons",
            "one or two things",
            "nine to five",
            "one on one",
            "there can be only one",
            "three point five",
        ],
    )
    def test_untouched(self, said):
        assert digits(said) == said


class TestAnd:
    """British numerals need "and"; sentences use it for something else."""

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("one hundred and eighty", "180"),
            ("five thousand and one", "5001"),
            ("three hundred and four thousand", "304000"),
            # The run ends in front of an "and" it could not earn, so the
            # conjunction and everything after it are left alone.
            ("one hundred and something", "100 and something"),
        ],
    )
    def test_absorbed_only_where_numeric(self, said, wanted):
        assert digits(said) == wanted

    @pytest.mark.parametrize(
        "said",
        ["one and two", "two and a half", "one and one and one"],
    )
    def test_a_plain_conjunction_leaves_two_lone_numbers(self, said):
        assert digits(said) == said

    def test_a_trailing_scale_word_proves_the_and_was_a_conjunction(self):
        # "three hundred and four" really is 304, so the parser takes the "and";
        # the second "hundred" is the only clue that this is a range, and it
        # arrives too late for anything but a backtrack.
        assert digits("between three hundred and four hundred") == "between 300 and 400"


class TestPunctuationBetweenNumbers:
    """The model punctuates where it heard a pause, and a number is all pauses."""

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("One, one, eight, zero.", "1180."),
            ("seven-three-two", "732"),
            ("one, two, three", "123"),
        ],
    )
    def test_absorbed_at_three_or_more_single_digits(self, said, wanted):
        assert digits(said) == wanted

    @pytest.mark.parametrize(
        "said",
        [
            # Two is a list, not a code.
            "one, two",
            "one, two, or three",
            # Three, but not digits: an approximate range, not a PIN.
            "thirty, forty, fifty percent",
            # A comma always ends a group, so this cannot become "25 of which".
            "I've got twenty, five of which are broken",
            "twenty-five",
        ],
    )
    def test_otherwise_left_alone(self, said):
        assert digits(said) == said

    @pytest.mark.parametrize(
        "said",
        ["I'll take twenty. Five are broken.", "one 2 three", "one\ntwo", "one; two; three"],
    )
    def test_sentence_punctuation_is_structure_and_ends_the_run(self, said):
        assert digits(said) == said


class TestOh:
    """"Oh" is the spoken zero of every phone number, and an interjection."""

    @pytest.mark.parametrize(
        ("said", "wanted"),
        [
            ("seven oh two", "702"),
            ("four oh four", "404"),
            # A UK mobile, whose leading zero is the part that matters.
            ("oh seven eight one two three four five", "07812345"),
        ],
    )
    def test_counts_as_zero_among_enough_numbers(self, said, wanted):
        assert digits(said) == wanted

    @pytest.mark.parametrize("said", ["five oh", "double oh seven", "Oh, one, two."])
    def test_too_few_numbers_around_it_and_it_stays_a_word(self, said):
        assert digits(said) == said

    def test_a_refused_oh_does_not_take_the_rest_of_the_run_with_it(self):
        assert digits("Oh, one hundred.") == "Oh, 100."


class TestUnparseableRuns:
    """Nonsense in, the sentence back out."""

    @pytest.mark.parametrize(
        "said",
        [
            # "hundred" cannot start a numeral, and rescanning inside the run
            # would emit "a 10000" — worse than leaving it alone.
            "a hundred and ten thousand",
            "hundred hundred",
            "a thousand times",
            # Not number words at all.
            "hundreds of people",
            "first and second",
        ],
    )
    def test_untouched(self, said):
        assert digits(said) == said

    def test_a_scale_that_can_neither_extend_nor_start_ends_the_run(self):
        assert digits("two hundred hundred") == "200 hundred"


class TestSpokenValue:
    @pytest.mark.parametrize(
        ("words", "wanted"),
        [
            (["twenty", "five"], 25),
            (["one", "hundred", "and", "eighty"], 180),
            (["nineteen"], 19),
            (["two", "thousand", "and", "twenty", "four"], 2024),
        ],
    )
    def test_one_numeral(self, words, wanted):
        assert spoken_value(words) == wanted

    @pytest.mark.parametrize(
        "words",
        [
            [],
            ["twenty", "twenty"],  # two numerals, not one
            ["hundred", "and", "six"],  # nothing to multiply
            ["one", "and", "two"],  # not a numeral's "and"
            ["seven", "oh", "two"],  # a zero is always its own group
            ["one", "hundred", "and"],  # a dangling conjunction
            ["apples"],
        ],
    )
    def test_not_one_numeral(self, words):
        assert spoken_value(words) is None


class TestTheTripwire:
    """What app._numbered checks before it trusts the pass."""

    def test_accepts_a_real_conversion(self):
        said = "One, one, eight, zero."
        assert is_digit_substitution(said, to_digits(said))

    def test_accepts_a_no_op(self):
        said = "I have five apples"
        assert is_digit_substitution(said, to_digits(said))

    def test_rejects_text_the_spans_do_not_account_for(self):
        said = "one one eight zero"
        claimed = NumberResult("something else", (Replacement(0, 18, "1180"),))
        assert not is_digit_substitution(said, claimed)

    def test_rejects_a_span_over_something_that_is_not_a_number(self):
        said = "I have five apples"
        claimed = NumberResult("I have five 0", (Replacement(12, 18, "0"),))
        assert not is_digit_substitution(said, claimed)

    def test_rejects_a_replacement_that_is_not_digits(self):
        said = "one two"
        claimed = NumberResult("twelve", (Replacement(0, 7, "twelve"),))
        assert not is_digit_substitution(said, claimed)

    def test_rejects_spans_out_of_order_or_overlapping(self):
        said = "one two three"
        claimed = NumberResult(
            "12hree", (Replacement(0, 7, "12"), Replacement(4, 7, "23"))
        )
        assert not is_digit_substitution(said, claimed)

    def test_rejects_a_span_outside_the_text(self):
        said = "one two"
        claimed = NumberResult("12", (Replacement(0, 99, "12"),))
        assert not is_digit_substitution(said, claimed)

    @pytest.mark.parametrize(
        ("span", "wanted"),
        [
            ("one one eight zero", True),
            ("One, one, eight, zero", True),
            ("one hundred and eighty", True),
            ("seven-three-two", True),
            ("one apple", False),
            ("one. two", False),
        ],
    )
    def test_what_a_span_may_cover(self, span, wanted):
        assert replaces_only_number_words(span, 0, len(span)) is wanted


class TestIdempotence:
    @pytest.mark.parametrize(
        "said",
        [
            "one one eight zero",
            "One, one, eight, zero.",
            "between three hundred and four hundred",
            "twenty twenty",
            "I have five apples",
        ],
    )
    def test_a_second_pass_changes_nothing(self, said):
        once = digits(said)
        assert digits(once) == once


def test_empty_text():
    assert to_digits("") == NumberResult("")
