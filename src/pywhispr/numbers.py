"""Spoken numbers to digits: "one one eight zero" -> "1180".

Parakeet writes number words as words, so a dictated PIN, phone number, port,
ticket or year arrives as a sentence and has to be retyped — which is the thing
dictation exists to avoid. Like :mod:`pywhispr.vocab` this is therefore a pass
over the finished transcript, and a pure function of a string.

Two mechanisms, and the split between them is the whole design:

* **Within a group**, words compose arithmetically, so "twenty five" is 25 and
  "one hundred and eighty" is 180.
* **Across groups**, digit strings are concatenated, so "one one eight zero" is
  1180 — and "twenty twenty" is 2020, which is what a year sounds like.

"point" is neither: it is a *separator*, like a comma, that puts a decimal point
in the output. Both mechanisms then apply either side of it, which is what makes
"twenty six point two" and "two six point two" both 26.2, and "one point two
five" 1.25.

A **lone** number word is never touched. That is the false-trigger guard, and it
is structural rather than a heuristic, the same way ``scratch.py``'s doubled
phrase is: "I have five apples" and "one of the reasons" come out untouched
because one number word is not a number *sequence*. Everything else here leans
the same way — a wrong substitution in the middle of a sentence is worse than a
number left as words.

Deliberately not handled: ordinals ("first"), fractions, "double oh seven",
currency, and "a" as one. Each is its own decision, and "a" is the tempting one
to get wrong: it would turn "a million thanks" into "1000000 thanks".
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

log = logging.getLogger(__name__)

UNITS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
HUNDRED = "hundred"
BIG = {"thousand": 1000, "million": 10**6, "billion": 10**9, "trillion": 10**12}
# "oh" is the spoken zero of every phone number; "nought" is its formal cousin.
ZERO_WORDS = frozenset({"zero", "oh", "nought", "naught"})
# The one word that can sit inside a numeral without being a number itself.
CONJUNCTION = "and"
# The decimal point. Deliberately *not* a number word: it names no value, it
# cannot begin a run, and a run that ends at one leaves it where it was.
POINT = "point"
# The two words a run may carry that are not numbers.
INFIX_WORDS = frozenset({CONJUNCTION, POINT})

# Every word this pass recognises as a value. Note what is absent: "a" and "an",
# "half", "quarter", "dozen", "nil", the ordinals, and the plurals "hundreds"
# and "thousands" — so "hundreds of people" never begins a run at all.
NUMBER_WORDS = frozenset({*UNITS, *TEENS, *TENS, *BIG, *ZERO_WORDS, HUNDRED})

# The lone-number guard: this many number words before anything is converted.
MIN_RUN_WORDS = 2
# A run whose groups were separated by punctuation needs this many of them, and
# every one a single digit. A PIN, phone number or code is a digit run; a spoken
# list of two ("one, two, or three") is not, and that is the false positive this
# kills. See _convertible.
MIN_PUNCTUATED_GROUPS = 3
# "oh" is an interjection as often as it is a zero, so a run containing one needs
# more numbers around it, and a run *led* by one needs more still.
MIN_OH_GROUPS = 3
MIN_LEADING_OH_GROUPS = 4

# A word here is letters only — the same construction as join._FIRST_WORD, and
# deliberately not vocab._WORD, which folds an internal hyphen into one token
# ("twenty-five" would arrive as a single unrecognised word) and also matches
# bare digits.
_WORD = re.compile(r"[^\W\d_]+")
# Horizontal spacing only; a newline is structure, never silently swallowed.
_SPACES = " \t "  # escaped so a reformat cannot lose the non-breaking space
# Two number words with nothing but spacing between them stay in the same group.
_PLAIN_GAP = re.compile(f"[{_SPACES}]+")
# A comma or a dash keeps the *run* going but always ends the group, so
# "twenty, five of which are broken" cannot become "25 of which are broken".
_PUNCTUATED_GAP = re.compile(f"[{_SPACES}]*[,\\-–—][{_SPACES}]*")

_ANY_NUMBER_WORD = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_ANY_INFIX_WORD = "|".join(sorted(INFIX_WORDS, key=len, reverse=True))
_ANY_GAP = f"(?:[{_SPACES}]+|[{_SPACES}]*[,\\-–—][{_SPACES}]*)"
# What a replaced span is allowed to have been: number words separated by the
# gaps above, with an infix word ("and", "point") allowed only *between* two of
# them. This is the invariant the caller checks, and it is why a bug in the
# parser can only ever cost digits, never words — a span cannot begin or end
# with an infix word, so a stray "and" or "point" is never swallowed at an edge.
_ONLY_NUMBER_WORDS = re.compile(
    rf"(?:{_ANY_NUMBER_WORD})"
    rf"(?:{_ANY_GAP}(?:(?:{_ANY_INFIX_WORD}){_ANY_GAP})?(?:{_ANY_NUMBER_WORD}))*",
    re.IGNORECASE,
)
# What a replacement is allowed to be: digits, with at most one decimal point
# and a digit either side of it.
_DIGIT_OUTPUT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


@dataclass(frozen=True)
class Replacement:
    """One converted run, in the coordinates of the text it came from."""

    start: int
    end: int
    digits: str


@dataclass(frozen=True)
class NumberResult:
    """The rewritten text, and every span the pass claimed to have replaced.

    The spans are what makes the invariant provable — see is_digit_substitution.
    """

    text: str
    spans: tuple[Replacement, ...] = ()


# How a token joins the one before it: nothing a run can cross, plain spacing
# (which keeps a group going), or a comma or dash (which keeps the run going but
# always ends the group).
_BROKEN, _PLAIN, _PUNCTUATED = 0, 1, 2


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    word: str
    link: int  # how this token joins the previous one


class _Group:
    """One numeral being accumulated, in the classic total/chunk shape.

    ``place`` is the magnitude of the last slot filled, and it is the trick that
    separates "twenty five" from "twenty twenty": a teen fills the tens *and*
    the units, so nothing may follow it; a tens word leaves the units open.
    ``min_big`` has to be tracked apart from ``place`` because "hundred"
    multiplies within a chunk rather than closing one — collapse the two and
    either "two thousand three hundred" or "one hundred thousand" breaks.
    """

    def __init__(self) -> None:
        self.total = 0
        self.chunk = 0
        self.place: float = math.inf
        self.hundred_used = False
        self.min_big: float = math.inf

    @property
    def value(self) -> int:
        return self.total + self.chunk

    @property
    def scale_applied(self) -> bool:
        return self.hundred_used or self.min_big != math.inf

    def extend(self, word: str) -> bool:
        """Take `word` into this group, or answer False and change nothing."""
        small = UNITS.get(word, TEENS.get(word, TENS.get(word)))
        if small is not None:
            if small >= self.place:
                return False
            self.chunk += small
            self.place = 10 if word in TENS else 1
            return True
        if word == HUNDRED:
            if not 0 < self.chunk < 100 or self.hundred_used:
                return False
            self.chunk *= 100
            self.hundred_used = True
            self.place = 100
            return True
        scale = BIG.get(word)
        if scale is not None:
            if self.chunk == 0 or scale >= self.min_big:
                return False
            self.total += self.chunk * scale
            self.chunk = 0
            self.min_big = scale
            self.hundred_used = False
            self.place = scale
            return True
        return False


def _is_scale(word: str) -> bool:
    return word == HUNDRED or word in BIG


def spoken_value(words: Sequence[str]) -> int | None:
    """The value of `words` read as a single numeral, or None if they aren't one.

    A test seam for the state machine, and the answer to "would these have been
    one group?". "and" is accepted where a numeral really uses it.
    """
    if not words or _is_scale(words[0].casefold()):
        return None
    group = _Group()
    pending_and = False
    for raw in words:
        word = raw.casefold()
        if word == CONJUNCTION:
            if pending_and or not group.scale_applied:
                return None
            pending_and = True
            continue
        if word in ZERO_WORDS or not group.extend(word):
            return None
        pending_and = False
    return None if pending_and else group.value


def _tokens(text: str) -> list[_Token]:
    """Every letters-only word of `text`, with how each joins the one before.

    A gap that is neither plain spacing nor a comma or dash is `_BROKEN`, which
    ends any run — so a stray digit ("one 2 three"), a newline or a full stop
    ("twenty. Five are broken") all keep their sentences intact.
    """
    out: list[_Token] = []
    previous_end = -1
    for match in _WORD.finditer(text):
        if previous_end < 0:
            link = _BROKEN
        else:
            gap = text[previous_end : match.start()]
            if _PLAIN_GAP.fullmatch(gap):
                link = _PLAIN
            elif _PUNCTUATED_GAP.fullmatch(gap):
                link = _PUNCTUATED
            else:
                link = _BROKEN
        out.append(_Token(match.start(), match.end(), match.group().casefold(), link))
        previous_end = match.end()
    return out


def _extent(tokens: list[_Token], start: int) -> int:
    """One past the last token of the run beginning at `start`.

    A run is number words joined by separators, and may carry an infix word
    ("and", "point") inside it — but never as its last word, which would put the
    conjunction of the next clause, or a sentence's own "point", inside the span.
    """
    end = start + 1
    while end < len(tokens) and tokens[end].link != _BROKEN:
        word = tokens[end].word
        if word not in NUMBER_WORDS and word not in INFIX_WORDS:
            break
        end += 1
    while end > start and tokens[end - 1].word in INFIX_WORDS:
        end -= 1
    return end


@dataclass
class _Run:
    """What a scan of one run consumed, and everything the guards need to see."""

    groups: list[int]
    words: int  # number words consumed; the infix words do not count
    last: int  # index of the last token consumed
    punctuated: bool  # was any group boundary a comma or a dash?
    point_at: int | None = None  # the group the decimal point sits in front of

    @property
    def digits(self) -> str:
        parts = [str(value) for value in self.groups]
        if self.point_at is None:
            return "".join(parts)
        return f"{''.join(parts[: self.point_at])}.{''.join(parts[self.point_at :])}"


def _scan(tokens: list[_Token], start: int, end: int) -> _Run | None:
    """Parse the run `tokens[start:end]` into groups, or None if there are none.

    Stops early wherever the words stop being one number sequence, and the
    caller keeps only what was consumed. Two stops are worth naming:

    * A **scale word that cannot extend** its group backtracks. If the group had
      absorbed an "and", the run ends *before* that "and" — which is what tells
      "three hundred and four" (304) from "three hundred and four hundred", a
      range whose second "hundred" is the only clue that the "and" was a
      conjunction all along.
    * An **"and" is only ever pending**: it is absorbed once the word after it
      extends the group, and otherwise the run ends in front of it. So a
      conjunction can never widen a span it did not earn.
    * A **"point"** closes the group and marks where the decimal point goes,
      but only once, only between two number words (see _fraction_follows) and
      only where no comma has already broken the run. Inside the fraction a
      comma or a scale word ends the run rather than joining it, which is what
      leaves "one point five million" as 1.5 million instead of 1.5000000.
    """
    groups: list[int] = []
    group: _Group | None = None
    words = 0
    last = -1
    punctuated = False
    pending_and = False
    point_at: int | None = None
    # Where to rewind to if a scale word later proves the absorbed "and" was a
    # conjunction: the groups, value and word count as they were in front of it.
    rewind: tuple[int, int, int] | None = None

    def close() -> None:
        nonlocal group
        if group is not None:
            groups.append(group.value)
            group = None

    for index in range(start, end):
        token = tokens[index]
        word = token.word
        if point_at is not None and (token.link == _PUNCTUATED or _is_scale(word)):
            # A fraction is a digit sequence and nothing else: the scale belongs
            # to the number as a whole ("one point five million"), and a comma
            # after the fraction is the sentence's, not the number's.
            break
        if word == POINT:
            if pending_and or punctuated or point_at is not None or token.link != _PLAIN:
                break
            if not _fraction_follows(tokens, index, end):
                break
            close()
            point_at = len(groups)
            rewind = None
            continue
        if word == CONJUNCTION:
            # Not after a comma, and not before a scale has been applied: both
            # would be an ordinary conjunction ("one and two", "one hundred, and
            # eighty"), and neither is ours to swallow.
            if pending_and or token.link == _PUNCTUATED or group is None or not group.scale_applied:
                break
            pending_and = True
            rewind = (len(groups), group.value, words)
            continue
        if token.link == _PUNCTUATED:
            if pending_and:
                break
            if group is not None or groups:
                punctuated = True
            close()
        if word in ZERO_WORDS:
            # A zero neither joins a numeral nor admits one: it is always its own
            # digit, so "seven oh two" is three groups and never seventy-two.
            if pending_and:
                break
            close()
            groups.append(0)
            words += 1
            last = index
            rewind = None
            continue
        if group is None:
            group = _Group()
        if not group.extend(word):
            if _is_scale(word):
                if rewind is not None:
                    kept, value, kept_words = rewind
                    del groups[kept:]
                    groups.append(value)
                    group = None
                    words = kept_words
                    last = _last_before_and(tokens, start, index)
                break
            close()
            group = _Group()
            group.extend(word)
            rewind = None
        pending_and = False
        words += 1
        last = index
    close()
    if not groups or last < start:
        return None
    return _Run(groups, words, last, punctuated, point_at)


def _fraction_follows(tokens: list[_Token], index: int, end: int) -> bool:
    """Is the "point" at `index` followed by the digits of a fraction?

    This is the false-trigger guard for the decimal point, and it is the same
    shape as every other one here: structural, not a heuristic. "point" is an
    ordinary English word, so it only counts where a number word sits on each
    side of it with nothing but spacing between — "the two point plan" and "one
    point I want to make" have no number after it and are left alone. A scale
    word is not a fraction ("one point five million"), and the number in front
    is guaranteed by a run only ever starting at a number word.
    """
    after = index + 1
    if after >= end:
        return False
    follower = tokens[after]
    return (
        follower.link == _PLAIN
        and follower.word in NUMBER_WORDS
        and not _is_scale(follower.word)
    )


def _last_before_and(tokens: list[_Token], start: int, index: int) -> int:
    """The last number word in front of the "and" that precedes `index`."""
    for back in range(index - 1, start - 1, -1):
        if tokens[back].word == CONJUNCTION:
            return back - 1
    return index - 1


def _convertible(run: _Run, tokens: list[_Token], start: int) -> bool:
    """Do the guards allow this run to become digits?

    Every one of these leans the same way — a number left as words is a smaller
    annoyance than a sentence rewritten around a word that was never a number.
    """
    if run.words < MIN_RUN_WORDS:
        return False  # one number word is not a number sequence
    if run.point_at is not None and not 0 < run.point_at < len(run.groups):
        return False  # a decimal point with nothing on one side of it
    single_digits = all(0 <= value <= 9 for value in run.groups)
    if run.punctuated and (len(run.groups) < MIN_PUNCTUATED_GROUPS or not single_digits):
        return False
    said = [tokens[i].word for i in range(start, run.last + 1)]
    if "oh" in said:
        if len(run.groups) < MIN_OH_GROUPS:
            return False
        if said[0] == "oh" and (len(run.groups) < MIN_LEADING_OH_GROUPS or not single_digits):
            return False
    return True


def to_digits(text: str) -> NumberResult:
    """Write every convertible run of spoken numbers in `text` as digits.

    Like apply_vocabulary, the result is built by concatenating untouched slices
    of `text` around the spans that were replaced, so nothing outside a run can
    be disturbed. The span runs from the first word of the run to the last word
    *consumed*, which is what makes an absorbed comma or "and" vanish while the
    full stop after the number stays where the model put it.
    """
    if not text:
        return NumberResult(text)
    tokens = _tokens(text)
    out: list[str] = []
    spans: list[Replacement] = []
    cursor = 0
    index = 0
    while index < len(tokens):
        word = tokens[index].word
        if word not in NUMBER_WORDS:
            index += 1
            continue
        end = _extent(tokens, index)
        if _is_scale(word):
            # The head of this run is unparseable on its own ("a hundred and ten
            # thousand"). Skipping the whole run leaves the sentence alone;
            # rescanning inside it would emit "a 10000", which is far worse.
            index = end
            continue
        run = _scan(tokens, index, end)
        if run is None or not _convertible(run, tokens, index):
            index += 1
            continue
        start_char, end_char = tokens[index].start, tokens[run.last].end
        out.append(text[cursor:start_char])
        out.append(run.digits)
        spans.append(Replacement(start_char, end_char, run.digits))
        cursor = end_char
        index = run.last + 1
    out.append(text[cursor:])
    if spans:
        # Counts only. The transcript is whatever the user just said.
        log.debug("Numbers pass converted %d run(s)", len(spans))
    return NumberResult("".join(out), tuple(spans))


def replaces_only_number_words(original: str, start: int, end: int) -> bool:
    """Is `original[start:end]` nothing but number words, separators and "and"?"""
    return bool(_ONLY_NUMBER_WORDS.fullmatch(original[start:end]))


def is_digit_substitution(original: str, result: NumberResult) -> bool:
    """The caller's tripwire: does `result` follow from `original` by its spans?

    A length ratio like the vocabulary's cannot be used here — "one one eight
    zero" legitimately becomes a quarter of its length — and this is stronger
    anyway. Every claimed span must sit inside the original, in order, hold
    nothing but number words, and produce nothing but digits; and re-splicing
    them has to reproduce the result exactly. So however the parser misbehaves,
    the only thing it can have replaced is a number.
    """
    rebuilt: list[str] = []
    cursor = 0
    for span in result.spans:
        if not cursor <= span.start < span.end <= len(original):
            return False
        if not _DIGIT_OUTPUT.fullmatch(span.digits):
            return False
        if not replaces_only_number_words(original, span.start, span.end):
            return False
        rebuilt.append(original[cursor : span.start])
        rebuilt.append(span.digits)
        cursor = span.end
    rebuilt.append(original[cursor:])
    return "".join(rebuilt) == result.text
