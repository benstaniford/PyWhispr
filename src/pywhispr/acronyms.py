"""Spoken letters and codes: "E P M one one eight zero" -> "EPM-1180".

A dictated ticket reference is the everyday case this exists for. Parakeet does
most of the work already — it glues spelled letters into an acronym by itself,
and writes "A twenty three B C two three four" as ``A23BC234`` — so like
:mod:`pywhispr.vocab` and :mod:`pywhispr.numbers` this is a pass over the
finished transcript that closes the gaps it leaves, and a pure function of a
string.

Two jobs, one scan over *groups* — a run of capitals or a run of digits,
separated by nothing or by spacing:

* **Letters join.** "C V E" becomes CVE, for the occasions the model leaves the
  letters spaced (it does when they are said with pauses).
* **Letters followed by digits gain a hyphen.** "EPM 1180" becomes EPM-1180,
  which is the thing that has to be retyped otherwise.

Interleaved letters and digits are concatenated, which is almost always what the
model wrote in the first place, so that arm is mostly a no-op.

Every guard here is structural rather than a heuristic, and every one is the
answer to a transcript that was measured rather than imagined:

* **Capitals only.** Spelled letters arrive as capitals and ordinary words do
  not, which is the whole of what leaves "Windows 11", "at 5 pm" and "a 12-month
  contract" alone.
* **Spacing between groups, never punctuation.** This is what saves "A, B, or C"
  and "John F. Kennedy" — the gap rule, not a word list.
* **A lone pair of letters is not a spelling.** The model glues sentences
  together ("Grade A. B is worse." really does arrive as "Grade A B is worse."),
  so a run of letters on its own needs three of them; two is enough only when a
  digit run follows and corroborates it. Two thresholds, the same shape as
  ``numbers.py``'s two for "oh".
* **A multi-letter group may only ever stand alone.** "The AB CD test" is two
  acronyms with a space between them, not a sequence of letters.

Deliberately not handled: a lowercase acronym, so "HELP 4567" needs a
`help => HELP` vocabulary entry — the model writes "help" in lower case because
it is an ordinary word, and reading a capitalised word as an acronym is what
would turn "Windows 11" into "Windows-11". Nor a run of digits the model already
split up ("X Y Z 9 8 7"), because spaced numbers are ``numbers.py``'s job and by
the time this runs they are one group. And never more than one hyphen, so "CVE
202512345" does not become CVE-2025-12345.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# A run of letters on its own has to be this long to be a spelling rather than
# two words the model ran together. A digit run corroborates the letters, so
# fewer are needed in front of one.
MIN_LETTERS_ALONE = 3
MIN_LETTERS_BEFORE_DIGITS = 2
# "I sent it to IT 5 times" is a sentence; a code's number has more to it.
MIN_DIGITS = 2

# Horizontal spacing only, like numbers.py: a newline is structure and is never
# silently swallowed.
# Written as an escape, not as the character: numbers.py's own _SPACES says it
# carries a non-breaking space and does not — a reformat took it years ago and
# the literal reads identically either way, which is how it went unnoticed.
_SPACES = " \t\xa0"
# A group is capitals or digits. Lower case is what tells a spelled letter from
# an ordinary word, so it can never appear in one.
_GROUP = re.compile(r"[A-Z]+|[0-9]+")
_GAP = re.compile(f"[{_SPACES}]*")
# What may not touch a candidate at either end: a letter of either case or a
# digit, so "Windows 11" is never read as the capital W and "A23BC234." keeps
# its full stop.
_ADJACENT = re.compile(r"[0-9A-Za-z]")
# What a replaced span is allowed to have been: those groups and that spacing,
# and nothing else. Note there is no hyphen here, which is what makes the
# proof in is_acronym_substitution sound.
_ONLY_GROUPS = re.compile(rf"(?:[A-Z]+|[0-9]+)(?:[{_SPACES}]*(?:[A-Z]+|[0-9]+))*")
# What a replacement is allowed to be: one sequence, and at most one hyphen with
# digits after it.
_ACRONYM_OUTPUT = re.compile(r"[A-Z0-9]+(?:-[0-9]+)?")


@dataclass(frozen=True)
class Replacement:
    """One joined run, in the coordinates of the text it came from."""

    start: int
    end: int
    written: str


@dataclass(frozen=True)
class AcronymResult:
    """The rewritten text, and every span the pass claimed to have replaced.

    Spans rather than the text alone for the same reason ``numbers.py`` does it:
    the caller cannot check this pass with a length ratio, because joining
    letters legitimately shortens the text, and a span lets the tripwire be a
    proof instead. See :func:`is_acronym_substitution`.
    """

    text: str
    spans: tuple[Replacement, ...] = ()


def _candidate(text: str, start: int) -> tuple[list[str], int]:
    """The groups of the run beginning at `start`, and one past its last one.

    A run continues while the next group is separated from this one by spacing
    or by nothing at all. Punctuation ends it, which is what leaves a list of
    letters ("A, B, or C") and a middle initial ("John F. Kennedy") alone.
    """
    groups: list[str] = []
    position = start
    while True:
        match = _GROUP.match(text, position)
        if match is None:
            break
        groups.append(match.group())
        position = match.end()
        gap = _GAP.match(text, position)
        assert gap is not None  # the pattern matches the empty string
        if _GROUP.match(text, gap.end()) is None:
            break
        position = gap.end()
    return groups, position


def _written(groups: list[str]) -> str | None:
    """The one token `groups` should become, or None to leave them as they are.

    Adjacent letter groups are one acronym, so merging them first leaves a
    strict alternation of letters and digits to decide about.
    """
    merged: list[tuple[str, str]] = []
    letters_in: list[int] = []  # how many groups each merged group came from
    single: list[bool] = []  # was each merged group's first one a lone letter?
    for group in groups:
        kind = "digits" if group[0].isdigit() else "letters"
        if merged and merged[-1][0] == kind == "letters":
            # A sequence of letters is single letters throughout: "AB CD" is two
            # acronyms with a space between them and none of our business.
            if not single[-1] or len(group) > 1:
                return None
            merged[-1] = (kind, merged[-1][1] + group)
            letters_in[-1] += 1
        else:
            merged.append((kind, group))
            letters_in.append(1)
            single.append(len(group) == 1)

    if len(merged) == 1:
        kind, group = merged[0]
        if kind == "letters" and letters_in[0] >= MIN_LETTERS_ALONE:
            return group
        return None
    if len(merged) == 2:
        (first_kind, first), (second_kind, second) = merged
        if first_kind == "letters" and second_kind == "digits":
            if len(first) >= MIN_LETTERS_BEFORE_DIGITS and len(second) >= MIN_DIGITS:
                return f"{first}-{second}"
        # Digits then letters is prose: "at 5 PM", "the 3 PCs".
        return None
    # Interleaved, which is a code however it was said.
    return "".join(group for _, group in merged)


def to_acronyms(text: str) -> AcronymResult:
    """Join every run of spoken letters and codes in `text`."""
    if not text:
        return AcronymResult(text)
    out: list[str] = []
    spans: list[Replacement] = []
    cursor = 0
    position = 0
    while position < len(text):
        match = _GROUP.search(text, position)
        if match is None:
            break
        start = match.start()
        if start and _ADJACENT.match(text[start - 1]):
            # Mid-word, so this is an ordinary word's capital, not a group.
            position = match.end()
            continue
        groups, end = _candidate(text, start)
        if end < len(text) and _ADJACENT.match(text[end]):
            position = end
            continue
        written = _written(groups)
        if written is None or written == text[start:end]:
            position = end
            continue
        out.append(text[cursor:start])
        out.append(written)
        spans.append(Replacement(start, end, written))
        cursor = position = end
    out.append(text[cursor:])
    if spans:
        # Counts only. The transcript is whatever the user just said.
        log.debug("Acronym pass joined %d run(s)", len(spans))
    return AcronymResult("".join(out), tuple(spans))


def is_acronym_substitution(original: str, result: AcronymResult) -> bool:
    """The caller's tripwire: does `result` follow from `original` by its spans?

    Shaped like ``numbers.is_digit_substitution`` — every span must sit inside
    the original, in order, hold nothing but groups and spacing, and produce one
    sequence; and re-splicing them has to reproduce the result exactly.

    It can prove one thing that pass cannot, and that check is the load-bearing
    one: the replacement must be its own source with the spacing taken out and
    at most one hyphen put in. So however the scanner misbehaves, it cannot lose,
    add or re-case a single character. A hyphen cannot be one the user dictated,
    because _ONLY_GROUPS does not admit one.
    """
    rebuilt: list[str] = []
    cursor = 0
    for span in result.spans:
        if not cursor <= span.start < span.end <= len(original):
            return False
        if not _ACRONYM_OUTPUT.fullmatch(span.written):
            return False
        source = original[span.start : span.end]
        if not _ONLY_GROUPS.fullmatch(source):
            return False
        if span.written.replace("-", "", 1) != "".join(source.split()):
            return False
        rebuilt.append(original[cursor : span.start])
        rebuilt.append(span.written)
        cursor = span.end
    rebuilt.append(original[cursor:])
    return "".join(rebuilt) == result.text
