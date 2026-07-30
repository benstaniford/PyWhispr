"""Filler removal: drop the "um"s and "uh"s out of a finished transcript.

"Um, so I, uh, think so." becomes "So I think so." — the hesitations go, and so
does the punctuation and spacing that was only holding them apart, which is most
of the work. A pure function over strings, like :mod:`pywhispr.join`.

Two invariants:

- **Deletion only.** :func:`is_deletion_only` states it and the caller checks it.
  The audio is gone by the time this runs, so a bug here must cost the user their
  "um"s at worst, never the dictation.
- **A closed list**, not a "looks like filler" rule. A wrongly deleted word costs
  a re-dictation; a surviving "um" costs a keystroke.

Conversational filler ("you know", "I mean") is out of scope: the difference
between "you know that I left" and "it's you know complicated" is meaning, not
spelling. `extra_filler_words` is the out for anyone who wants one gone anyway.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from pywhispr.join import CLOSERS, NO_SPACE_STARTERS, OPENERS, SENTENCE_ENDERS

log = logging.getLogger(__name__)

# Noises, not words. "err" (to err), "ah", "oh", "eh", "hmm" and "mm" are left
# out: ordinary words, or things people type deliberately.
DEFAULT_FILLERS: tuple[str, ...] = (
    "um",
    "umm",
    "ummm",
    "uhm",
    "uhmm",
    "uh",
    "uhh",
    "uhhh",
    "erm",
    "ermm",
    "er",
    "ahem",
)

# "uh huh" means yes and "uh oh" means trouble, so neither half goes.
PROTECTED_FOLLOWERS = frozenset({"huh", "oh"})

# Words a hesitation follows, where a comma in front of the filler was its pause
# rather than the sentence's punctuation. Positive and short by design: in a list
# ("milk, um, eggs") the comma has work to do. join.CONTINUATION_WORDS is the
# wrong list — it includes "this", "one" and "yes", which end list items.
PAUSE_WORDS = frozenset(
    """
    i we you they he she it my your his her its our their the a an
    of to at in on for with from by
    just really very quite actually sort kind
    think thinks thought thinking know knows knew say says said saying mean
    means meant guess suppose reckon feel feels felt hope wonder wondered
    remember recall
    is was are were be been am being do does did doing have has had having
    can could would should shall must might will
    and but or so because since although though however therefore then when
    while if that which what where whether than as
    """.split()
)

# "If you do, um, tell me" keeps its comma: "If you do tell me" means something
# else. A clause opening with one of these ends at that comma.
SUBORDINATORS = frozenset(
    """
    if when whenever unless until till whether while whilst although though
    because since after before once whatever whichever whoever wherever
    """.split()
)
# Skipped when looking for what a clause opens with, or "And when you do, um,
# call me" loses its comma to the same meaning change.
_COORDINATORS = frozenset("and but or so nor yet even still only then".split())

# Object pronoun before, subject pronoun after: the comma spliced two clauses
# ("I saw it, um, I left"). Contractions normalise as "we're" -> "were".
_OBJECT_PRONOUNS = frozenset("it her him them us me you".split())
_SUBJECT_PRONOUNS = frozenset(
    """
    i we you they he she it that this there
    im ive id youre youve theyre theyve hes shes its thats theres were weve well
    """.split()
)

_WORD = re.compile(r"\w+(?:['’\-]\w+)*")
# Horizontal spacing only: a newline is structure, never silently swallowed.
_SPACES = " 	 "  # escaped so a reformat cannot lose the non-breaking space
_SPACE = f"[{_SPACES}]"
_SEPARATOR = re.compile(f"{_SPACE}+")
# A run ("um, uh") is removed in one piece, leaving one seam instead of three.
_RUN_GAP = re.compile(f"{_SPACE}*,?{_SPACE}*")
# What goes with the filler on its right. No colon or semicolon: that punctuation
# is the sentence's own.
_TRAILING = re.compile(f"{_SPACE}*(?:,{_SPACE}*)?")
_OWN_SENTENCE_END = re.compile(f"[{re.escape(SENTENCE_ENDERS)}]+{_SPACE}*")
_BLANK_LINE = re.compile(f"{_SPACE}*\r?\n")
# Punctuation that hugs the word before it, so there is no seam on the right.
_HUGS_LEFT = SENTENCE_ENDERS + NO_SPACE_STARTERS + CLOSERS + "\"'" + "\r\n"
_SENTENCE_OPENERS = SENTENCE_ENDERS + OPENERS + "\"'"
_LINE_OPENER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*$")
_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "“": "”",
    "‘": "’",
    '"': '"',
    "'": "'",
    "—": "—",
    "–": "–",
}
# Dashes are the sentence's punctuation, so "I think — um — we go" keeps one.
_DASHES = "—–"

Phrases = frozenset[tuple[str, ...]]


def normalise(word: str) -> str:
    """Case and punctuation go, so "Beyond-Trust" and "beyondtrust" agree."""
    return "".join(ch for ch in word.casefold() if ch.isalnum())


def _token_key(token: str) -> str:
    """The matchable form of a transcript word, or "" for never-a-filler.

    An all-capitals word is an acronym: ER, UM and ERM are a department, a system
    and a report, and case-folding them into the filler list deletes real words.
    """
    if len(token) > 1 and token.isupper():
        return ""
    return normalise(token)


def compile_fillers(words: Iterable[str]) -> Phrases:
    """Compile filler terms into normalised word tuples, ignoring empty entries.

    A phrase entry ("you know") matches those words adjacent and separated by
    plain spacing; its run-together spelling is registered too.
    """
    phrases = set()
    for entry in words:
        parts = tuple(normalise(word) for word in _WORD.findall(entry.replace("-", " ")))
        if not parts or not all(parts):
            continue
        phrases.add(parts)
        if len(parts) > 1:
            phrases.add(("".join(parts),))
    return frozenset(phrases)


def filler_words(extra: Iterable[str] = (), keep: Iterable[str] = ()) -> Phrases:
    """The built-in hesitations plus `extra`, minus anything in `keep`."""
    return frozenset(compile_fillers((*DEFAULT_FILLERS, *extra)) - compile_fillers(keep))


def is_deletion_only(original: str, result: str) -> bool:
    """Is `result` `original` with things deleted, and at most re-capitalised?

    Alphanumerics only and case-insensitive, so spacing, punctuation and a
    promoted capital may differ while a substituted word may not.
    """
    haystack = iter(ch for ch in original.casefold() if ch.isalnum())
    return all(ch in haystack for ch in result.casefold() if ch.isalnum())


_CLAUSE_ENDS = ".!?…,;:\r\n\"'“‘”’"


def _clause_opener(text: str, end: int) -> str:
    """The word the clause ending at `end` opens with, normalised.

    Scans back to the boundary rather than splitting the prefix, which made a
    long transcript quadratic.
    """
    start = end
    while start > 0 and text[start - 1] not in _CLAUSE_ENDS:
        start -= 1
    for match in _WORD.finditer(text, start, end):
        word = normalise(match.group())
        if word not in _COORDINATORS:
            return word
    return ""


def _paired_mark(before: str, after: str) -> str | None:
    """The paired mark bracketing a filler, if that is what it is.

    A straight quote or dash is both halves of its own pair, so in
    `He said "go" um "now".` the mark in front is *closing* the quotation and
    pairing the two fuses "go" and "now". Quotes therefore only count where an
    opening quote can stand; dashes don't nest, so they always pair.
    """
    mark = before[-1:]
    closer = _PAIRS.get(mark)
    if closer is None or after[:1] != closer:
        return None
    if closer == mark and mark not in _DASHES:
        opening = before[:-1]
        if opening and not (opening[-1].isspace() or opening[-1] in OPENERS):
            return None
    return mark


def _opens_sentence(text: str, index: int) -> bool:
    """Would a word at `index` start a sentence, and so want a capital?"""
    i = index
    while i > 0 and text[i - 1].isspace():
        if text[i - 1] in "\r\n":
            return True
        i -= 1
    if i == 0 or text[i - 1] in _SENTENCE_OPENERS:
        return True
    return bool(_LINE_OPENER.match(text, text.rfind("\n", 0, i) + 1, i))


def remove_fillers(text: str, phrases: Phrases) -> str:
    """Delete every filler in `text`, tidying the seam it leaves behind.

    Built by concatenating untouched slices, so nothing outside a deleted span
    can be disturbed. Returns "" when nothing said is left.
    """
    if not text or not phrases:
        return text

    max_words = max(len(phrase) for phrase in phrases)
    spans = [match.span() for match in _WORD.finditer(text)]
    keys = [_token_key(text[start:end]) for start, end in spans]

    def spaced(start: int, stop: int) -> bool:
        return all(
            _SEPARATOR.fullmatch(text[spans[n - 1][1] : spans[n][0]])
            for n in range(start + 1, stop)
        )

    def match_phrase(start: int) -> int | None:
        """Token index just past the longest filler at `start`, if any."""
        for length in range(min(max_words, len(spans) - start), 0, -1):
            stop = start + length
            if not spaced(start, stop) or tuple(keys[start:stop]) not in phrases:
                continue
            if (
                stop < len(spans)
                and keys[stop] in PROTECTED_FOLLOWERS
                and _SEPARATOR.fullmatch(text[spans[stop - 1][1] : spans[stop][0]])
            ):
                return None
            return stop
        return None

    def match_run(start: int) -> int | None:
        """As match_phrase, but swallowing a run of fillers whole."""
        stop = match_phrase(start)
        if stop is None:
            return None
        while stop < len(spans):
            if not _RUN_GAP.fullmatch(text[spans[stop - 1][1] : spans[stop][0]]):
                break
            extended = match_phrase(stop)
            if extended is None:
                break
            stop = extended
        return stop

    def was_a_pause(i: int, stop: int) -> bool:
        """Was the comma before this filler the filler's pause, or the sentence's?

        Only a surplus comma can result from getting this wrong, never a lost
        separator — a comma that reads as the pause it was.
        """
        if i == 0:
            return True
        start, end = spans[i - 1]
        previous = keys[i - 1]
        if previous not in PAUSE_WORDS:
            return False
        if (
            end - start == 1
            and text[start].isupper()
            and text[start] != "I"
            and not _opens_sentence(text, start)
        ):
            return False  # a label, not the article: "plan A, um, plan B"
        if previous in _OBJECT_PRONOUNS and stop < len(spans) and keys[stop] in _SUBJECT_PRONOUNS:
            return False  # a comma splice: "I saw it, um, I left"
        return _clause_opener(text, end) not in SUBORDINATORS

    out: list[str] = []
    emitted_last = ""
    emitted_blank = True

    def push(chunk: str) -> None:
        nonlocal emitted_last, emitted_blank
        if chunk:
            out.append(chunk)
            emitted_last = chunk[-1]
            emitted_blank = emitted_blank and not chunk.strip()

    def tidy(cursor: int) -> int:
        """Clear up the seam a deletion left where it sat."""
        nonlocal emitted_last, emitted_blank
        if emitted_blank:
            # The filler opened the dictation: its spacing, line break and the
            # mark it was holding go too, or the paste starts with a blank line.
            out.clear()
            emitted_last, emitted_blank = "", True
            while cursor < len(text) and (text[cursor].isspace() or text[cursor] in ",;:"):
                cursor += 1
        elif emitted_last in "\r\n" and (blank := _BLANK_LINE.match(text, cursor)):
            cursor = blank.end()  # the filler was a line of its own
        elif emitted_last in _SPACES:
            while cursor < len(text) and text[cursor] in _SPACES:
                cursor += 1
        if emitted_last.isalnum() and cursor < len(text) and text[cursor].isalnum():
            push(" ")  # whatever the branches did, two words never fuse
        return cursor

    cursor = 0
    removed = 0
    i = 0
    while i < len(spans):
        stop = match_run(i)
        if stop is None:
            i += 1
            continue

        start_char, end_char = spans[i][0], spans[stop - 1][1]
        opens = _opens_sentence(text, start_char)
        trailing = _TRAILING.match(text, end_char).group()
        after_at = end_char + len(trailing)
        after = text[after_at : after_at + 1]
        lead = text[cursor:start_char]
        bare = lead.rstrip(_SPACES)
        mark = _paired_mark(bare, after)

        if opens and (own_end := _OWN_SENTENCE_END.match(text, after_at)):
            # "Hello. Um. Right." — the full stop was the filler's, so it goes
            # too rather than doubling up.
            cursor = own_end.end()
            push(lead.rstrip() if cursor >= len(text) else lead)
        elif mark is not None and mark in _DASHES:
            push(lead)
            cursor = after_at + 1  # one dash stays
        elif mark is not None:
            push(bare[:-1].rstrip(_SPACES))
            cursor = after_at + 1
        elif not after or after[0] in _HUGS_LEFT:
            # Nothing follows in this sentence, so tidy the left instead:
            # "fine, um." -> "fine."
            if bare.endswith((",", ";", ":")) and not opens:
                bare = bare[:-1].rstrip(_SPACES)
            push(bare)
            cursor = after_at if not after else end_char
        else:
            if "," in trailing and bare.endswith(",") and was_a_pause(i, stop):
                lead = bare[:-1].rstrip(_SPACES) + " "
            push(lead)
            cursor = after_at

        cursor = tidy(cursor)
        if opens and cursor < len(text) and text[start_char].isupper() and text[cursor].islower():
            push(text[cursor].upper())  # the filler was carrying the capital
            cursor += 1

        removed += 1
        i = stop

    push(text[cursor:])
    result = "".join(out)
    if removed and not any(ch.isalnum() for ch in result):
        result = ""  # all filler: insert nothing, not a stray full stop
    if removed:
        log.debug("Removed %d filler(s)", removed)  # counts only, never the text
    return result
