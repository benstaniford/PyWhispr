"""What a plugin is handed and what it may hand back.

Deliberately tiny and free of Qt, :mod:`pywhispr.config` and anything heavy: this
is the module every plugin imports, including plugins written by the user, so it
has to stay cheap and stable.

A plugin declares one or more :class:`Trigger` phrases and implements either or
both of two functions, which are two very different jobs:

``rewrite(match) -> Rewrite | None``
    Change the text. It says which span of the transcript it claims and what should
    replace it — it never returns a whole new transcript, because the *framework*
    does the splicing (see :mod:`pywhispr.plugins.engine`) and text outside a
    claimed span therefore cannot be disturbed.

    **Runs on whichever thread is handling the transcript**, and there is more than
    one: the GUI thread for a local dictation, or an HTTP request thread for a
    network transcription (``app._api_transcribe``). The API server runs up to
    ``api_max_queue`` requests at once, so a rewrite may also be running in another
    thread — including alongside a local dictation. So it must be **reentrant** and
    must **not touch Qt or any UI**, and it must be quick, because on the local
    path it sits between transcription and the paste. A pure function of its
    ``Match`` satisfies all three; anything else is asking for trouble, and heavy
    or stateful work belongs in ``act``.

``act(match) -> None``
    Do something. Runs on its own thread once the text has been inserted, so it
    may take its time, and an exception in it costs nothing but a log line.

Returning ``None`` from :func:`rewrite` is how a plugin says "those words were not
meant for me". That is where the false-trigger guard lives, and it has to: only
the plugin knows whether "send me an emoji" is a command (it is not). A plugin
with a ``rewrite`` only acts when that rewrite claimed something; a plugin with no
``rewrite`` at all acts on every trigger match.
"""

from __future__ import annotations

from dataclasses import dataclass

# How many words either side of the trigger a plugin is shown — and, because a
# plugin may only rewrite what it was shown, how far a claim can reach. Four
# matches vocab.MAX_PHRASE_WORDS: past that, a spoken phrase is a sentence.
LOOKBEHIND_WORDS = 4
LOOKAHEAD_WORDS = 4


@dataclass(frozen=True)
class Trigger:
    """The words that wake a plugin up.

    ``phrase`` is literal text, not a pattern: the framework compiles it with word
    boundaries and case-insensitivity, and allows any spacing or punctuation
    between its words, because the model is free to write "thumbs up, emoji". A
    plugin that genuinely needs a regex can pass ``pattern`` instead, and is then
    on its own for boundaries.

    ``at_segment_end`` additionally requires the phrase to be the end of a clause
    or of the transcript — punctuation or nothing after it. It is the same
    structural guard :mod:`pywhispr.scratch` uses on the spoken reset phrase, and
    the same trade-off: it costs the odd mid-sentence command and buys immunity
    from every sentence that merely mentions the trigger word.
    """

    phrase: str
    at_segment_end: bool = False
    pattern: str | None = None


@dataclass(frozen=True)
class Word:
    """One word of context, with where it sits in the transcript."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Match:
    """A trigger firing: the transcript, where the trigger sits, and its context.

    ``words_before`` is in reading order, so the word nearest the trigger is last
    — which is the end a plugin walks back from. ``window_start``/``window_end``
    bound what this match is allowed to rewrite: the extent of the context the
    plugin was given. A claim outside it is rejected, so a plugin cannot reach
    across a transcript it was never shown.
    """

    transcript: str
    start: int
    end: int
    words_before: tuple[Word, ...] = ()
    words_after: tuple[Word, ...] = ()
    window_start: int = 0
    window_end: int = 0

    @property
    def trigger_text(self) -> str:
        return self.transcript[self.start : self.end]

    def claim(self, start: int, end: int, text: str, html: str | None = None) -> Rewrite:
        """Convenience for the usual shape: replace a span with a string."""
        return Rewrite(start=start, end=end, text=text, html=html)

    def claim_from(self, word: Word, text: str, html: str | None = None) -> Rewrite:
        """Replace everything from `word` up to the end of the trigger with `text`.

        The common case for a trailing-keyword plugin: "thumbs up emoji" becomes
        one character, and the space in front of "thumbs" is left alone because
        the claim starts where the word does.
        """
        return Rewrite(start=word.start, end=self.end, text=text, html=html)

    def nothing_to_change(self) -> Rewrite:
        """Claim the match without altering any text.

        For an action plugin that still wants its ``act`` gated on its own
        validation: a zero-width claim changes nothing and overlaps nothing.
        """
        return Rewrite(start=self.start, end=self.start, text="")


@dataclass(frozen=True)
class Rewrite:
    """A span of the transcript and what should replace it.

    Half-open, like a slice: ``transcript[start:end]`` becomes ``text``. A
    zero-width span is legal and means "mine, but change nothing".

    ``html`` is optional markup for things plain text cannot say — a Teams custom
    emoji is an ``<img>`` referencing a tenant asset, not a character. It is a
    *second rendering* of ``text``, never a substitute for it: the transcript stays
    plain throughout the pipeline, so the join, the history and the network API are
    unaffected, and the markup is used only when the paste target accepts HTML. So
    ``text`` must still be something the user would be content to receive — the
    emoji's name, say — because that is what they get anywhere HTML does not reach.
    """

    start: int
    end: int
    text: str
    html: str | None = None


__all__ = [
    "LOOKAHEAD_WORDS",
    "LOOKBEHIND_WORDS",
    "Match",
    "Rewrite",
    "Trigger",
    "Word",
]
