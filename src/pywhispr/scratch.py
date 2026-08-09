"""Spoken restart: "scratch that" and what it throws away.

Transcription only happens once the recording has stopped, so there is no way to
notice the phrase while the user is still talking — the reset *hotkey* is the
live version of this. What is left is the cheap trick: find the phrase in the
finished transcript and keep only what follows the **last** one, because saying
it twice means starting over twice.

A pure function over strings, like :mod:`pywhispr.join` and
:mod:`pywhispr.filler`. The result is always a suffix of the input (at most its
first letter re-cased), and the caller checks that — the audio is gone by the
time this runs.

Off unless the user lists phrases: "scratch that idea" is a sentence someone
means, and there is nothing to re-dictate from.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Punctuation the discarded half left hanging in front of the surviving text.
_LEADING_PUNCTUATION = re.compile(r"^[\s,.;:!?…\-—–]+")


def compile_reset_phrases(phrases: list[str]) -> re.Pattern[str] | None:
    """One case-insensitive pattern for all `phrases`, or None if there are none.

    Words within a phrase may be separated by any spacing or punctuation in the
    transcript ("scratch that." then a new sentence), and the whole phrase has
    to sit on word boundaries so "startover" in a file name is safe.
    """
    parts = [
        r"\W+".join(re.escape(word) for word in phrase.split())
        for phrase in phrases
        if phrase.strip()
    ]
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def strip_before_reset(text: str, pattern: re.Pattern[str] | None) -> str:
    """Keep only what `text` says after its last reset phrase.

    Returns "" when the phrase was the last thing said — the user asked for
    nothing, so nothing is what gets inserted.
    """
    if not text or pattern is None:
        return text
    last = None
    for match in pattern.finditer(text):
        last = match  # saying it twice means starting over twice
    if last is None:
        return text
    tail = _LEADING_PUNCTUATION.sub("", text[last.end() :])
    if not tail.strip():
        log.debug("Voice reset discarded the whole transcript (%d characters)", len(text))
        return ""
    if tail[0].islower():
        tail = tail[0].upper() + tail[1:]  # it opens the dictation now
    log.debug("Voice reset kept %d characters of %d", len(tail), len(text))
    return tail


def is_suffix_of(original: str, result: str) -> bool:
    """Is `result` the tail of `original`, bar its first letter's case?

    The caller's tripwire: this pass may only ever delete a prefix.
    """
    return len(result) <= len(original) and original.casefold().endswith(result[1:].casefold())
