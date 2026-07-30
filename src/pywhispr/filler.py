"""Filler removal: drop the "um"s and "uh"s out of a finished transcript.

Parakeet is faithful — it writes down every "um", "uh" and "erm" you said,
because that is what it heard. Read back, dictation full of them looks like a
transcript of a meeting rather than something you wrote, so this pass takes them
out along with the punctuation and spacing they leave behind: "Um, so I, uh,
think so." becomes "So I think so."

Like :mod:`pywhispr.join` and :mod:`pywhispr.vocab` this is a pure function over
strings, testable without a model, a microphone or a UI, and like them it is
built to fail safe:

- **Deletion only.** Every alphanumeric character in the result appears, in
  order, in the input — the one exception being the first letter of a word
  promoted to a capital where a sentence-opening filler was removed.
  :func:`is_deletion_only` states that as a check the caller can run, because
  by the time this runs the audio is gone and a bug must not eat the transcript.
- **A closed list of sounds nobody types on purpose.** Not "words that look like
  filler": *like*, *right*, *so* and *well* are all real words far more often
  than they are noise, and a wrongly deleted word is worse than a kept "um" —
  the user can see the first only by re-reading, and has to re-dictate to fix
  it. The list is extensible from the config for anyone who wants *you know*
  gone too.

  A tier for those conversational fillers was written and dropped: no closed list
  and no punctuation rule separates "you know that I left" from "it's you know
  complicated" reliably, because the difference is meaning. Wispr Flow does it
  with a fine-tuned LLM cleanup pass in the cloud; that is the tool for the job,
  and it is not a word list. `extra_filler_words` remains the out for anyone who
  wants a specific phrase gone unconditionally.

Deleting the word is the easy half; the **seam** is the work. A filler is held
apart from its sentence by spacing, by a comma or two, sometimes by a dash or a
bracket, and if it opened the sentence it is carrying the capital. All of that
has to go with it, or the result reads worse than the "um" did. A recording that
was nothing but filler yields the empty string, so the caller inserts nothing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from pywhispr.join import CLOSERS, NO_SPACE_STARTERS, OPENERS, SENTENCE_ENDERS

log = logging.getLogger(__name__)

# Hesitation sounds and their common spellings. Deliberately narrow: every entry
# here is a noise, not a word, so deleting it can't change what a sentence says.
# "err" (to err), "ah", "oh", "eh", "hmm" and "mm" are all left out — they are
# either ordinary words or things people type deliberately. Add them yourself
# with extra_filler_words if you never mean them.
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

# A filler is not a filler when it is the first half of one of these: "uh huh"
# means yes and "uh oh" means trouble. (The hyphenated spellings are safe
# already — _WORD keeps "uh-huh" as one token.)
PROTECTED_FOLLOWERS = frozenset({"huh", "oh"})

# Words a hesitation follows, where a comma in front of the filler was a pause
# rather than punctuation: "I think, um, we should go". Deliberately positive and
# short — CONTINUATION_WORDS is the wrong list here, because it answers a
# different question and includes words that end a list item ("this", "one",
# "yes"), where the comma has work to do. Determiners, prepositions and subject
# pronouns are the commonest hesitation site of all ("Pass me the, um, wrench")
# and can barely be followed by a structural comma, so they are safe to list.
# "not" is deliberately absent: "Believe it or not, we went."
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

# A clause that opens with one of these keeps its comma even after a PAUSE_WORD:
# "If you do, um, tell me" needs it, because "If you do tell me" says something
# else. The auxiliary ends the subordinate clause there rather than trailing off.
SUBORDINATORS = frozenset(
    """
    if when whenever unless until till whether while whilst although though
    because since after before once whatever whichever whoever wherever
    """.split()
)
# Skipped when looking for the word a clause opens with: dictated speech starts
# clauses with these all the time, and "And when you do, um, call me" is the same
# meaning change as "If you do" — the subordinator is simply one word further in.
_COORDINATORS = frozenset("and but or so nor yet even still only then".split())

# An object pronoun before the filler and a subject pronoun after it means the
# comma was splicing two clauses ("I saw it, um, I left"), not marking a pause.
_OBJECT_PRONOUNS = frozenset("it her him them us me you".split())
# Contractions included, normalised the way _token_key sees them ("we're" →
# "were"), because "That's it, um, we're done." is the same comma splice.
_SUBJECT_PRONOUNS = frozenset(
    """
    i we you they he she it that this there
    im ive id youre youve theyre theyve hes shes its thats theres were weve well
    """.split()
)

# A word, matching pywhispr.vocab: letters, digits and the internal punctuation
# that holds one word together, so "uh-huh" is one token.
_WORD = re.compile(r"\w+(?:['’\-]\w+)*")
# Horizontal spacing only, including the non-breaking space: a newline is
# structure, not spacing, and is never silently swallowed.
_SPACES = " \t\u00a0"  # the non-breaking space is escaped so a reformat cannot lose it
_SPACE = f"[{_SPACES}]"
# Words of one filler phrase ("you know") may only be separated by plain spacing.
_SEPARATOR = re.compile(f"{_SPACE}+")
# Between fillers of a run ("um, uh") a comma is allowed too, so the whole run
# goes in one deletion and leaves one seam instead of three.
_RUN_GAP = re.compile(f"{_SPACE}*,?{_SPACE}*")
# What a deleted filler takes with it on its right: the spacing, plus the comma
# that was only there to hold it apart from the words around it. A colon or
# semicolon is *not* included — that punctuation is the sentence's own.
_TRAILING = re.compile(f"{_SPACE}*(?:,{_SPACE}*)?")
# A filler that was its own utterance takes its full stop with it too.
_OWN_SENTENCE_END = re.compile(f"[{re.escape(SENTENCE_ENDERS)}]+{_SPACE}*")
# The line a deleted filler leaves empty behind it.
_BLANK_LINE = re.compile(f"{_SPACE}*\r?\n")
# Punctuation that belongs tight against the word before it, so a filler in
# front of it has no seam on its right to tidy — only one on its left.
_HUGS_LEFT = SENTENCE_ENDERS + NO_SPACE_STARTERS + CLOSERS + "\"'" + "\r\n"
# After these a capital is due, exactly as after a full stop.
_SENTENCE_OPENERS = SENTENCE_ENDERS + OPENERS + "\"'"
# A line holding nothing but a bullet or a number, which pywhispr.join also
# treats as the start of a sentence.
_LINE_OPENER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*$")
# Punctuation that comes in pairs. When a filler is all that stood between the
# two, the pair was bracketing the filler and goes with it: "(um)" → "".
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
# A dash pair is different: the dashes are the sentence's punctuation, not the
# filler's, so "I think — um — we go" keeps one dash rather than losing both.
_DASHES = "—–"

Phrases = frozenset[tuple[str, ...]]


def normalise(word: str) -> str:
    """Reduce a word to what "the same word" means here: case and punctuation go."""
    return "".join(ch for ch in word.casefold() if ch.isalnum())


def _token_key(token: str) -> str:
    """The matchable form of a word in the transcript, or "" for never-a-filler.

    An all-capitals word is an acronym, not a hesitation: ER, UM and ERM are
    departments, systems and reports. Case-folding them into the filler list
    would delete real words — the one thing this module must not do. The shape
    test is the same one :func:`pywhispr.join.lowercase_first_word` uses.
    """
    if len(token) > 1 and token.isupper():
        return ""
    return normalise(token)


def compile_fillers(words: Iterable[str]) -> Phrases:
    """Compile filler terms into normalised word tuples, ignoring empty entries.

    An entry may be a phrase ("you know"); it then matches those words adjacent
    and separated by plain spacing. The run-together spelling is registered too,
    so "you know" also catches the "you-know" the model sometimes writes.
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
    """Hesitation sounds, removed wherever they appear: built-ins plus `extra`.

    `extra` joins this tier rather than the punctuated one, because a term the
    user typed in themselves is a term they want gone.
    """
    return frozenset(compile_fillers((*DEFAULT_FILLERS, *extra)) - compile_fillers(keep))


def is_deletion_only(original: str, result: str) -> bool:
    """Is `result` `original` with things deleted (and at most re-capitalised)?

    Compares alphanumeric characters only, case-insensitively, so spacing,
    punctuation and the capital promoted onto a new opening word are all allowed
    to differ while a substituted or invented word is not. The caller uses this
    as a tripwire: the audio is gone by the time this pass runs.
    """
    haystack = iter(ch for ch in original.casefold() if ch.isalnum())
    return all(ch in haystack for ch in result.casefold() if ch.isalnum())


# Where a clause begins, looking backwards. Quotes count: `She said "if you can,
# um, come"` opens its clause inside the quotation, not before it.
_CLAUSE_ENDS = ".!?…,;:\r\n\"'“‘”’"


def _clause_opener(text: str, end: int) -> str:
    """The word the clause ending at `end` opens with, normalised.

    Scans back to the clause boundary rather than splitting the whole prefix:
    this is asked once per comma'd filler, and a long transcript would otherwise
    be quadratic.
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
    """The paired punctuation mark bracketing a filler, if that is what it is.

    A straight quote or a dash is both halves of its own pair, so ending up next
    to one says nothing about which half it is: in `He said "go" um "now".` the
    mark before the filler is *closing* the previous quotation, and treating the
    two as a pair fuses "go" and "now" into one word. A quote only counts as
    opening where an opening quote can stand — the inverse of
    :func:`pywhispr.join._closes_a_quote`. Dashes don't nest, so a dash either
    side of a filler is always the one pair.
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
    """Would a word at `index` start a sentence, and so want a capital?

    Takes an index rather than the preceding text: it is asked once per candidate
    token, and slicing the prefix each time made a long transcript quadratic.
    """
    i = index
    while i > 0 and text[i - 1].isspace():
        if text[i - 1] in "\r\n":
            return True  # a line break starts a sentence
        i -= 1
    if i == 0 or text[i - 1] in _SENTENCE_OPENERS:
        return True
    return bool(_LINE_OPENER.match(text, text.rfind("\n", 0, i) + 1, i))


def remove_fillers(text: str, phrases: Phrases) -> str:
    """Delete every filler in `text`, tidying the seam it leaves behind.

    The result is built by concatenating untouched slices of `text`, so nothing
    outside a deleted span (and the punctuation and spacing that were holding it
    apart) can be disturbed. Returns "" when nothing said is left.
    """
    if not text or not phrases:
        return text

    max_words = max(len(phrase) for phrase in phrases)
    spans = [match.span() for match in _WORD.finditer(text)]
    keys = [_token_key(text[start:end]) for start, end in spans]

    def spaced(start: int, stop: int) -> bool:
        """Are tokens start..stop separated by nothing but plain spacing?"""
        return all(
            _SEPARATOR.fullmatch(text[spans[n - 1][1] : spans[n][0]])
            for n in range(start + 1, stop)
        )

    def match_phrase(start: int) -> int | None:
        """Token index just past the longest filler phrase at `start`, if any."""
        for length in range(min(max_words, len(spans) - start), 0, -1):
            stop = start + length
            window = tuple(keys[start:stop])
            if not spaced(start, stop):
                continue
            if window not in phrases:
                continue
            if (
                stop < len(spans)
                and keys[stop] in PROTECTED_FOLLOWERS
                and _SEPARATOR.fullmatch(text[spans[stop - 1][1] : spans[stop][0]])
            ):
                return None  # "uh huh": not a hesitation at all
            return stop
        return None

    def match_run(start: int) -> int | None:
        """As match_phrase, but swallowing a run of fillers ("um, uh") whole."""
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
        """Was the comma before this filler the sentence's, or the filler's pause?

        A comma either side of a filler is usually the pair of pauses around it —
        "I think, um, we should go" wants "I think we should go", not "I think,
        we should go". But in a list ("milk, um, eggs") the comma is the
        sentence's own and must survive. Nothing local settles it, so this is a
        small **positive** list of words a hesitation follows — verbs of speech
        and thought, auxiliaries, connectives — and every other word keeps its
        comma. Lopsided the same way as the join and the vocabulary: a surplus
        comma reads as the pause it was, a missing one loses the structure.
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
            # A label or an initial rather than the article: "plan A, um, plan B".
            # Sentence-initially it can only be the article ("A, um, big problem").
            return False
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
            # The filler opened the dictation, so its spacing, its line break and
            # the comma or colon it was holding go with it — otherwise the paste
            # starts with a blank line or a stray mark.
            out.clear()
            emitted_last, emitted_blank = "", True
            while cursor < len(text) and (text[cursor].isspace() or text[cursor] in ",;:"):
                cursor += 1
        elif emitted_last in "\r\n" and (blank := _BLANK_LINE.match(text, cursor)):
            cursor = blank.end()  # the filler was a line of its own
        elif emitted_last in _SPACES:
            while cursor < len(text) and text[cursor] in _SPACES:
                cursor += 1  # one space, not two
        if emitted_last.isalnum() and cursor < len(text) and text[cursor].isalnum():
            push(" ")  # whatever else happens, two words never fuse into one
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
        after = text[after_at : after_at + 1]  # one character is all any branch needs
        lead = text[cursor:start_char]
        bare = lead.rstrip(_SPACES)
        mark = _paired_mark(bare, after)

        if opens and (own_end := _OWN_SENTENCE_END.match(text, after_at)):
            # A whole sentence of filler: "Hello. Um. Right." — the full stop was
            # the filler's, so it goes too rather than doubling up.
            cursor = own_end.end()
            push(lead.rstrip() if cursor >= len(text) else lead)
        elif mark is not None and mark in _DASHES:
            # "I think — um — we go": the dashes are the sentence's, so one of
            # the two stays and the parenthetical it opened is left intact.
            push(lead)
            cursor = after_at + 1
        elif mark is not None:
            # "(um)", "he said \"um\"": the pair was only holding the filler.
            push(bare[:-1].rstrip(_SPACES))
            cursor = after_at + 1
        elif not after or after[0] in _HUGS_LEFT:
            # Nothing follows it in this sentence, so there is no seam on its
            # right to tidy: take the spacing (and the comma that was holding it
            # apart) on its left instead — "fine, um." → "fine."
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
            # The filler was carrying the sentence's capital: hand it on.
            push(text[cursor].upper())
            cursor += 1

        removed += 1
        i = stop

    push(text[cursor:])
    result = "".join(out)
    if removed and not any(ch.isalnum() for ch in result):
        # All filler and punctuation: the user said nothing, so insert nothing
        # rather than a stray full stop.
        result = ""
    # Counts only — the transcript is whatever the user just said.
    if removed:
        log.debug("Removed %d filler(s)", removed)
    return result
