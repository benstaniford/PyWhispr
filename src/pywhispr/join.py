"""Join a fresh transcript onto whatever text already precedes the caret.

Every recording is transcribed on its own, so the model capitalises the first
word and appends a full stop every time. Pasted verbatim, two dictations in a
row mesh badly: "I went to the shop.Then I came home."

This module decides two things and nothing else: whether to prepend a single
space, and whether to lower-case the opening word. It never touches text that
is already in the document — ``join_text`` returns exactly ``("" | " ") + text``
with at most the first letter re-cased, which is what lets the caller verify the
result and fall back to the raw transcript if anything here misbehaves.

A full stop in the preceding text is taken as deliberate, so the capital that
follows it is left alone. Only an unfinished sentence licenses lower-casing.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

SENTENCE_ENDERS = ".!?…"
CLOSERS = ")]}”’»"
OPENERS = "([{“‘«"
# Trimmed before looking for a sentence end, so `said."` and `(done.)` both
# read as finished. Straight quotes are included here but not in CLOSERS,
# because they are ambiguous when deciding about a space and are not when
# deciding whether a sentence ended.
_END_TRIM = CLOSERS + "\"'"
GLUE = "-–—/\\_@#=+~"  # a tail like "well-" or "foo@" wants no space after it
NO_SPACE_STARTERS = ".,;:!?…)]}%’'"

# A line holding nothing but a bullet or a number: the next words start a
# sentence, however the previous line ended.
_LIST_MARKER = re.compile(r"^\s*([-*•]|\d+[.)])\s*$")

# Leading word, apostrophes included so "It's" and "I'm" come back whole.
_FIRST_WORD = re.compile(r"^[^\W\d_]+(?:['’][^\W\d_]+)*")


def _vocab(*groups: str) -> frozenset[str]:
    """Flatten the word groups, dropping every one-letter word.

    "I" must never be lower-cased, and no other single letter is worth the
    risk, so the length filter makes that structural instead of something we
    have to remember not to add.
    """
    return frozenset(word for group in groups for word in group.split() if len(word) > 1)


# Only a word from this list can ever be lower-cased. A closed list rather than
# "lower-case anything that looks like Xyz" because the failure modes are
# lopsided: a word we have not listed keeps its capital, which is exactly what
# happens today, whereas a loose rule eventually produces "ben", "monday" and
# "google" — errors the user can see and has to retype.
#
# Deliberately absent: will, may, march, august, june, july, mark, grant, bill,
# frank, rose, hope, art, chip, van, don, sue, ray, dawn — all of them names or
# months as often as they are function words.
CONTINUATION_WORDS = _vocab(
    # connectives
    """and but or nor yet so because since although though while whilst whereas
    unless until till if then than as when whenever where wherever after before
    once also besides moreover furthermore however nevertheless nonetheless
    therefore thus hence meanwhile otherwise instead anyway still either
    neither both""",
    # pronouns, determiners, relatives ("i" and "a" are dropped by _vocab)
    """which who whom whose that what whatever whichever he she it they we you
    him her them us his hers its their theirs our ours your yours my mine this
    these those there here one some any all each every everyone everything
    someone something anyone anything nobody nothing none another other others
    such same the an""",
    # prepositions
    """of in on at to for with without within into onto from by about above
    below under over through throughout across against along among around
    behind beside between beyond despite during except inside outside near off
    per toward towards upon via versus""",
    # auxiliaries and other high-frequency openers
    """is are was were be been being am do does did doing done have has had
    having can could would should shall must might need ought going getting
    said says saying seems looks means makes gives gets want wants think thinks
    thought know knows knew like likes just really actually basically obviously
    probably maybe perhaps definitely certainly especially particularly
    generally usually often sometimes always never already almost even only
    quite rather very much more most less least few several many enough again
    back together ago okay yes not plus""",
)


def _closes_a_quote(preceding: str) -> bool:
    """Is the trailing straight quote closing something, or opening it?

    "the dogs' " wants a space after it; "he said \"" does not.
    """
    if len(preceding) < 2:
        return False
    return preceding[-2].isalnum() or preceding[-2] in ".,!?"


def needs_leading_space(preceding: str, text: str) -> bool:
    if not preceding or not text:
        return False
    # The new text brings its own separator, or is punctuation that belongs
    # tight against what came before. Note we never strip it: if it starts with
    # whitespace, that whitespace is the separator.
    if text[0] in NO_SPACE_STARTERS or text[0].isspace():
        return False
    tail = preceding[-1]
    if tail.isspace():
        return False  # already separated — this is what stops double spaces
    if tail in OPENERS or tail in GLUE:
        return False
    if tail in "\"'":
        return _closes_a_quote(preceding)
    return True


def is_mid_sentence(preceding: str) -> bool:
    """True when the caret sits in a sentence nobody has finished yet."""
    if preceding.endswith(("\n", "\r")):
        return False  # a line break starts a sentence
    stripped = preceding.rstrip()
    if not stripped:
        return False
    if _LIST_MARKER.match(stripped.splitlines()[-1]):
        return False
    core = stripped.rstrip(_END_TRIM)
    return not core or core[-1] not in SENTENCE_ENDERS


def lowercase_first_word(text: str) -> str:
    """Lower-case the opening word, but only where that is safe.

    Two gates, both required: the word must be shaped like an ordinary
    capitalised word (so ``AND``, ``IT`` and ``AndThen`` are left alone), and it
    must be one of the CONTINUATION_WORDS.
    """
    match = _FIRST_WORD.match(text)
    if match is None:
        return text
    word = match.group()
    if not (word[0].isupper() and not word.isupper() and word[1:].islower()):
        return text
    if word.lower() not in CONTINUATION_WORDS:
        return text
    return text[0].lower() + text[1:]


def join_text(preceding: str | None, text: str, *, lowercase_continuations: bool = True) -> str:
    """Adapt ``text`` to the context it is about to be pasted into.

    ``preceding`` is the text immediately before the caret; ``None`` means we
    could not find out, and an empty string means the caret really is at the
    start of the field. Both leave ``text`` alone.
    """
    if not text or preceding is None:
        return text

    space = needs_leading_space(preceding, text)
    lowered = False
    if lowercase_continuations and is_mid_sentence(preceding):
        adjusted = lowercase_first_word(text)
        lowered = adjusted != text
        text = adjusted

    # Length only: the context can be anything the user happens to have
    # focused, so it must never reach the log.
    log.debug("Join: %d chars of context, space=%s, lowered=%s", len(preceding), space, lowered)
    return " " + text if space else text
