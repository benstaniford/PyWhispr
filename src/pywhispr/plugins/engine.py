"""Matching triggers, validating what a plugin claims, and splicing the result.

The one rule that makes plugins safe to have at all: **the framework does the
splicing, never the plugin.** A plugin returns a span and a replacement string;
this module verifies the span and then builds the output by concatenating
untouched slices of the transcript with replacement strings, exactly as
:func:`pywhispr.vocab.apply_vocabulary` does. Consequences:

- Text outside a claimed span cannot be disturbed, whatever the plugin does.
- A plugin cannot return a whole new transcript, so it cannot lose one.
- A claim is confined to the window of context the plugin was shown, so a
  trigger at the end cannot rewrite the beginning.
- Anything invalid — a bad span, a non-string, an exception, a claim that
  overlaps one already accepted — skips that one match and leaves the text
  alone. The transcript survives every failure mode here.

That matters more than it sounds: by the time any of this runs the audio is gone,
so a bug that mangles text is unrecoverable, and one that merely fails to fire
costs the user a keystroke.

:func:`apply_plugins` is a pure function of (text, plugins), like
:mod:`pywhispr.join` and :mod:`pywhispr.filler`, so the whole matching layer is
testable with no model, microphone, plugin folder or UI.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pywhispr.plugins.api import (
    LOOKAHEAD_WORDS,
    LOOKBEHIND_WORDS,
    Match,
    Rewrite,
    Trigger,
    Word,
)
from pywhispr.scratch import SEGMENT_BOUNDARY

log = logging.getLogger(__name__)

# A word, for context purposes: letters, digits and the internal punctuation that
# holds one word together ("o'clock", "well-known", "GPT-4"). The same notion as
# vocab._WORD and filler._WORD, kept local for the same reason they are.
_WORD = re.compile(r"\w+(?:['’\-]\w+)*")

# The most a single replacement may be. A rewrite turns spoken words into a short
# string — an emoji, a symbol, a date — so this is generous by two orders of
# magnitude and exists only to stop a looping plugin from producing a megabyte.
MAX_REPLACEMENT_CHARS = 4096

# On the local path a rewrite sits between transcription and the paste, so it has a
# hard budget in principle and only a warning in practice: enforcing it would
# need a subprocess per plugin, which is a far bigger machine than the problem.
SLOW_REWRITE_MS = 50


@dataclass(frozen=True)
class Plugin:
    """A loaded plugin, as the engine sees it.

    ``rewrite`` and ``act`` are the two phases described in
    :mod:`pywhispr.plugins.api`; a plugin may implement either or both, and one
    with neither is inert. ``source`` is "builtin" or the file it came from, for
    log lines that have to identify a plugin without quoting the user's words.
    """

    name: str
    triggers: tuple[Trigger, ...]
    rewrite: Callable[[Match], Rewrite | None] | None = None
    act: Callable[[Match], None] | None = None
    source: str = "builtin"
    patterns: tuple[re.Pattern[str], ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class PendingAction:
    """A plugin whose trigger fired, waiting to be run once the text is in place."""

    plugin: Plugin
    match: Match


@dataclass(frozen=True)
class PluginResult:
    text: str
    actions: tuple[PendingAction, ...] = ()
    rewrites: int = 0


def compile_trigger(trigger: Trigger) -> re.Pattern[str] | None:
    """One case-insensitive pattern for `trigger`, or None if it is unusable.

    A literal phrase is joined with ``\\W+`` between its words, like
    :func:`pywhispr.scratch.compile_reset_phrases`: the model is free to write
    "thumbs up, emoji" for what was said in one breath, and a phrase whose words
    must be adjacent would miss it.

    ``at_segment_end`` adds a lookahead for punctuation, a line break or the end
    of the transcript, so the trigger only counts where a clause ends.
    """
    if trigger.pattern is not None:
        try:
            return re.compile(trigger.pattern, re.IGNORECASE)
        except re.error as exc:
            log.error("Trigger pattern rejected: %s", exc)
            return None

    words = trigger.phrase.split()
    if not words:
        return None
    body = r"\W+".join(re.escape(word) for word in words)
    suffix = rf"(?=\s*(?:{SEGMENT_BOUNDARY}|$))" if trigger.at_segment_end else ""
    return re.compile(rf"\b(?:{body})\b{suffix}", re.IGNORECASE)


def _is_word_char(char: str) -> bool:
    """Would `char` be part of a word? Matches what _WORD considers one."""
    return char.isalnum() or char == "_"


def compile_patterns(triggers: tuple[Trigger, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile every usable trigger, dropping the ones that will not compile."""
    compiled = [compile_trigger(trigger) for trigger in triggers]
    return tuple(pattern for pattern in compiled if pattern is not None)


def _context(text: str, start: int, end: int) -> Match:
    """Build the Match for a trigger occupying ``text[start:end]``.

    The window a claim is confined to is the extent of the words handed over,
    always including the trigger itself, so a plugin given no context at all can
    still claim its own trigger.

    Where there is no context word on a side, the window stretches over the
    punctuation and spacing on that side instead. Without it a plugin whose whole
    job is to consume what was said — "new paragraph", where the words are the
    command — could not reach the full stop the model put after it, and would
    leave a lone "." behind. Only on a side with no words, so a claim can never
    reach past context the plugin was actually shown.
    """
    before = [
        Word(text[span[0] : span[1]], span[0], span[1])
        for span in (m.span() for m in _WORD.finditer(text, 0, start))
    ][-LOOKBEHIND_WORDS:]
    after = [
        Word(text[span[0] : span[1]], span[0], span[1])
        for span in (m.span() for m in _WORD.finditer(text, end))
    ][:LOOKAHEAD_WORDS]

    if before:
        window_start = min(before[0].start, start)
    else:
        window_start = start
        while window_start > 0 and not _is_word_char(text[window_start - 1]):
            window_start -= 1
    if after:
        window_end = max(after[-1].end, end)
    else:
        window_end = end
        while window_end < len(text) and not _is_word_char(text[window_end]):
            window_end += 1

    return Match(
        transcript=text,
        start=start,
        end=end,
        words_before=tuple(before),
        words_after=tuple(after),
        window_start=window_start,
        window_end=window_end,
    )


def _candidates(text: str, plugins: list[Plugin]) -> list[tuple[int, int, Plugin, Match]]:
    """Every trigger firing anywhere in `text`, in the order they will be offered.

    Sorted by position, then by load order, so two plugins claiming the same words
    resolve the same way every time — the earlier-loaded one wins. Built-ins load
    first, so a user's own plugin cannot be silently shadowed by one of ours
    appearing later.
    """
    found: list[tuple[int, int, Plugin, Match]] = []
    seen: set[tuple[int, int, int]] = set()
    for order, plugin in enumerate(plugins):
        for pattern in plugin.patterns:
            for hit in pattern.finditer(text):
                start, end = hit.span()
                if end == start:
                    continue  # a pattern that matches nothing would fire forever
                if (order, start, end) in seen:
                    continue  # two of this plugin's own triggers on the same words
                seen.add((order, start, end))
                found.append((start, order, plugin, _context(text, start, end)))
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def _validated(claim: object, match: Match, plugin: Plugin) -> Rewrite | None:
    """The claim if it is one this engine will act on, else None with a reason logged.

    Never quotes the text — a transcript is whatever window had focus, and the
    plugin name plus a length is enough to debug from.
    """
    if claim is None:
        return None
    if not isinstance(claim, Rewrite):
        log.error("Plugin %r returned %s, not a Rewrite", plugin.name, type(claim).__name__)
        return None
    if not isinstance(claim.text, str):
        log.error("Plugin %r claimed a %s replacement", plugin.name, type(claim.text).__name__)
        return None
    if not isinstance(claim.start, int) or not isinstance(claim.end, int):
        log.error("Plugin %r claimed a non-integer span", plugin.name)
        return None
    if not 0 <= claim.start <= claim.end <= len(match.transcript):
        log.error(
            "Plugin %r claimed [%d:%d] of %d characters; ignoring it",
            plugin.name,
            claim.start,
            claim.end,
            len(match.transcript),
        )
        return None
    if claim.start < match.window_start or claim.end > match.window_end:
        # A plugin may only rewrite what it was shown. Without this, a trigger at
        # the end of a long dictation could replace the whole thing.
        log.error(
            "Plugin %r claimed [%d:%d], outside the [%d:%d] it was given",
            plugin.name,
            claim.start,
            claim.end,
            match.window_start,
            match.window_end,
        )
        return None
    if len(claim.text) > MAX_REPLACEMENT_CHARS:
        log.error(
            "Plugin %r produced %d characters, over the %d limit",
            plugin.name,
            len(claim.text),
            MAX_REPLACEMENT_CHARS,
        )
        return None
    return claim


def _overlaps(claim: Rewrite, accepted: list[Rewrite]) -> bool:
    """Does `claim` collide with one already accepted? Zero-width never does."""
    return any(
        claim.start < other.end and other.start < claim.end
        for other in accepted
        if other.end > other.start
    )


def _rewrite_of(plugin: Plugin, match: Match) -> Rewrite | None:
    """Call a plugin's rewrite, containing anything it does wrong."""
    started = time.perf_counter()
    try:
        claim = plugin.rewrite(match) if plugin.rewrite is not None else None
    except Exception:
        log.exception("Plugin %r failed while rewriting; leaving the text alone", plugin.name)
        return None
    elapsed_ms = (time.perf_counter() - started) * 1000
    if elapsed_ms > SLOW_REWRITE_MS:
        log.warning(
            "Plugin %r took %.0fms to rewrite — a local dictation waits on that "
            "before it can paste, so heavy work belongs in act()",
            plugin.name,
            elapsed_ms,
        )
    return _validated(claim, match, plugin)


def apply_plugins(text: str, plugins: list[Plugin]) -> PluginResult:
    """Run every plugin over `text`, returning the new text and what should act.

    Actions are collected rather than run: they belong after the text has been
    inserted, on a thread that is not this one. A plugin with a ``rewrite`` only
    earns its action when that rewrite claimed something — its own validation is
    the gate — while a plugin with no ``rewrite`` acts on every trigger match.
    """
    if not text or not plugins:
        return PluginResult(text=text)

    accepted: list[Rewrite] = []
    actions: list[PendingAction] = []
    for _start, _order, plugin, match in _candidates(text, plugins):
        if plugin.rewrite is None:
            if plugin.act is not None:
                actions.append(PendingAction(plugin, match))
            continue

        claim = _rewrite_of(plugin, match)
        if claim is None:
            continue  # not for this plugin after all, or it misbehaved
        if _overlaps(claim, accepted):
            log.debug("Plugin %r overlaps an earlier claim; skipping this match", plugin.name)
            continue
        accepted.append(claim)
        if plugin.act is not None:
            actions.append(PendingAction(plugin, match))

    if not accepted:
        return PluginResult(text=text, actions=tuple(actions))

    out: list[str] = []
    cursor = 0
    for claim in sorted(accepted, key=lambda item: item.start):
        out.append(text[cursor : claim.start])
        out.append(claim.text)
        cursor = claim.end
    out.append(text[cursor:])

    # Counts and lengths only: the transcript is whatever the user just said.
    log.debug(
        "Plugins rewrote %d span(s) and queued %d action(s)", len(accepted), len(actions)
    )
    return PluginResult(text="".join(out), actions=tuple(actions), rewrites=len(accepted))
