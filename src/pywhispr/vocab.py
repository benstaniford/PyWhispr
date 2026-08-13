"""Custom vocabulary: fix the spelling of words the model has never met.

Parakeet cannot be biased towards a word list at decode time — ``generate()``
takes a mel spectrogram and nothing else, and the ONNX path is the same — so
this is a correction pass over the finished transcript rather than a hint to
the model. That bounds what it can do: if the audio evidence is gone, the word
is gone. What it does catch is the everyday case, where the model heard the
word perfectly well and then wrote it the way an ordinary English speaker
would: "beyond trust", "pie whisper", "cubernetes".

Two tiers, safest first:

1. **Exact**, once case and separators are stripped. "Beyond Trust",
   "beyondtrust" and "BEYOND TRUST" all become "BeyondTrust". This can never
   substitute a word for a different one, so every term gets it.
2. **Near**, within a small edit distance, and only for terms long enough that
   a near-miss is unlikely to be an ordinary word. "Jamf" is one edit from
   "jam", so short terms get tier 1 only.

Like :mod:`pywhispr.join` this is a pure function over strings, so every rule
is testable without a model, a microphone or a UI.

The file format is one term per line, in the spelling you want written::

    BeyondTrust
    Kubernetes
    # A specific mishearing the model makes every single time:
    pie whisper => PyWhispr

Only a line *starting* with ``#`` is a comment, so ``C#`` is a usable term.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir

from pywhispr.config import APP_NAME
from pywhispr.join import CONTINUATION_WORDS

log = logging.getLogger(__name__)

REPLACEMENT_ARROW = "=>"
COMMENT_PREFIX = "#"

# A term has to survive normalising into at least this much to be matchable at
# all. "C#" normalises to "c", which would rewrite every stray letter c in the
# transcript; such a term has to be written as an explicit `c sharp => C#`.
MIN_KEY_LENGTH = 2
# Below this many characters (normalised) a term is matched exactly only. At
# four characters an allowance of one edit already reaches a lot of ordinary
# English, and a wrong substitution is far more annoying than a missed one.
MIN_FUZZY_LENGTH = 5
# Edit budget by normalised term length, longest bracket first. Deliberately
# stingy: two edits on a fifteen-letter term is a typo, two on a six-letter one
# is a different word.
FUZZY_BUDGET = ((12, 2), (MIN_FUZZY_LENGTH, 1))
# Most words a single term may span. Also caps how far the matcher looks ahead,
# so a long transcript stays linear in practice.
MAX_PHRASE_WORDS = 4

# A word for matching purposes: letters, digits, and the internal punctuation
# that holds one word together ("o'clock", "well-known", "GPT-4").
_WORD = re.compile(r"\w+(?:['’\-]\w+)*")
# Words in a phrase may only be separated by plain spacing. A comma or a line
# break between them means they are not one term, whatever they normalise to.
_SEPARATOR = re.compile(r"[ \t ]+")


def normalise(word: str) -> str:
    """Reduce a word or phrase to what "the same word" means here.

    Case, spaces, hyphens and punctuation all go, because they are exactly what
    the model gets wrong about an unfamiliar name.
    """
    return "".join(ch for ch in word.casefold() if ch.isalnum())


@dataclass(frozen=True)
class Rule:
    """One vocabulary entry, compiled for matching."""

    wanted: str  # the spelling to write
    key: str  # normalised form to recognise
    words: int  # how many words the key was written as
    fuzzy: bool  # may this rule match approximately?


def _fuzzy_budget(length: int) -> int:
    for minimum, budget in FUZZY_BUDGET:
        if length >= minimum:
            return budget
    return 0


def parse_vocabulary(text: str) -> list[Rule]:
    """Compile the vocabulary file's text into rules, skipping bad lines.

    Problems are logged by line number only. The entries are the user's own
    private nouns — colleagues, customers, unreleased product names — and the
    log is the thing they send us when something breaks.
    """
    rules: list[Rule] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(COMMENT_PREFIX):
            continue

        if REPLACEMENT_ARROW in line:
            heard, _, wanted = line.partition(REPLACEMENT_ARROW)
            heard, wanted = heard.strip(), wanted.strip()
            # An explicit fix is exact by definition: the user has told us the
            # precise mishearing, so guessing around it adds only risk.
            fuzzy = False
        else:
            heard = wanted = line
            fuzzy = len(normalise(line)) >= MIN_FUZZY_LENGTH

        key = normalise(heard)
        if len(key) < MIN_KEY_LENGTH or not wanted:
            log.debug("Vocabulary line %d ignored: nothing to match on", lineno)
            continue
        if key in seen:
            log.debug("Vocabulary line %d ignored: duplicate of an earlier entry", lineno)
            continue
        seen.add(key)
        rules.append(Rule(wanted=wanted, key=key, words=len(_WORD.findall(heard)) or 1, fuzzy=fuzzy))
    return rules


def count_entries(text: str) -> tuple[int, int]:
    """(usable terms, ignored lines) — for the editor's live status line."""
    content = [
        line
        for line in (raw.strip() for raw in text.splitlines())
        if line and not line.startswith(COMMENT_PREFIX)
    ]
    terms = len(parse_vocabulary(text))
    return terms, len(content) - terms


def _windows(text: str, spans: list[tuple[int, int]], start: int, limit: int):
    """Yield (last token index, window text) for phrases beginning at `start`.

    Shortest first; the caller decides which end to prefer. Stops at the first
    separator that isn't plain spacing, since a phrase cannot span a comma or a
    newline.
    """
    end = start
    while end < len(spans) and end - start < limit:
        if end > start:
            gap = text[spans[end - 1][1] : spans[end][0]]
            if not _SEPARATOR.fullmatch(gap):
                return
        yield end, text[spans[start][0] : spans[end][1]]
        end += 1


def edit_distance_within(a: str, b: str, budget: int) -> bool:
    """Levenshtein distance ≤ budget, abandoning the moment it cannot be.

    Public because the emoji plugin's fuzzy tier needs exactly this, and reaching
    into another module's private helper is worse than saying it is shared.
    """
    if abs(len(a) - len(b)) > budget:
        return False
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, 1):
        current = [i]
        for j, ch_b in enumerate(b, 1):
            current.append(
                previous[j - 1]
                if ch_a == ch_b
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        if min(current) > budget:
            return False
        previous = current
    return previous[-1] <= budget


def _distance_if_near(key: str, words: int, rule: Rule) -> int | None:
    """How far `key` is from this rule's term, if that is close enough to act on."""
    # Same number of words on both sides, so an edit is always a misspelling
    # within a word and never a whole neighbouring word being swallowed: "a
    # beyond-trust problem" is one edit from "beyondtrust" and must not lose
    # its "a". Phrases the model split or joined are the exact tier's job.
    if rule.words != words:
        return None
    if rule.key == key:
        return None  # already correct; the exact pass has had its turn
    # "parakeets" is an inflection of a listed "Parakeet", spelled exactly the
    # way the user asked for. Rewriting it back to the singular is a bug.
    if key.startswith(rule.key):
        return None
    for distance in range(1, _fuzzy_budget(len(rule.key)) + 1):
        if edit_distance_within(key, rule.key, distance):
            return distance
    return None


def _best_fuzzy(key: str, words: int, rules: list[Rule]) -> str | None:
    """The one near-enough term for this window, or None if there isn't exactly one.

    A tie between two terms counts as no match: choosing either would be a coin
    toss, and the untouched transcript is the better of the two answers.
    """
    if not key or key in CONTINUATION_WORDS:
        # Ordinary joining words are never somebody's product name. This is the
        # common-word guard in full: a closed list of function words, not a
        # dictionary, so an unusual English word one edit from a listed term
        # can still be rewritten. That is what vocabulary_fuzzy = false is for.
        return None
    best: str | None = None
    best_distance = 0
    ambiguous = False
    for rule in rules:
        distance = _distance_if_near(key, words, rule)
        if distance is None:
            continue
        if best is None or distance < best_distance:
            best, best_distance, ambiguous = rule.wanted, distance, False
        elif distance == best_distance and rule.wanted != best:
            ambiguous = True
    return None if ambiguous else best


def apply_vocabulary(text: str, rules: list[Rule], *, fuzzy: bool = True) -> str:
    """Rewrite every recognised term in `text` to its wanted spelling.

    Matching is left to right and non-overlapping, longest phrase first so
    "Microsoft Defender" wins over a term for "Defender" alone. Exact matches
    are resolved before near ones at the same position, so an entry that is
    already spelled right is never dragged onto a similar-looking neighbour.

    The result is built by concatenating untouched slices of `text` with
    replacement strings, so text outside a matched span cannot be disturbed.
    """
    if not text or not rules:
        return text

    exact = {rule.key: rule.wanted for rule in rules}
    fuzzy_rules = [rule for rule in rules if rule.fuzzy] if fuzzy else []
    limit = min(MAX_PHRASE_WORDS, max(rule.words for rule in rules) + 1)

    spans = [m.span() for m in _WORD.finditer(text)]
    out: list[str] = []
    cursor = 0
    replacements = 0
    i = 0
    while i < len(spans):
        candidates = list(_windows(text, spans, i, limit))
        match: tuple[int, str] | None = None
        for end, window in reversed(candidates):  # longest phrase first
            wanted = exact.get(normalise(window))
            if wanted is not None:
                match = (end, wanted)
                break
        if match is None and fuzzy_rules:
            for end, window in reversed(candidates):
                wanted = _best_fuzzy(normalise(window), end - i + 1, fuzzy_rules)
                if wanted is not None:
                    match = (end, wanted)
                    break
        if match is None:
            i += 1
            continue

        end, wanted = match
        start_char, end_char = spans[i][0], spans[end][1]
        if text[start_char:end_char] != wanted:  # already correct: leave it be
            out.append(text[cursor:start_char])
            out.append(wanted)
            cursor = end_char
            replacements += 1
        i = end + 1

    out.append(text[cursor:])
    # Counts only — the transcript is whatever the user just said.
    if replacements:
        log.debug("Vocabulary applied %d correction(s)", replacements)
    return "".join(out)


# -- storage -----------------------------------------------------------------

# Shown in the editor the first time, so the format teaches itself. Comments
# only: an untouched template compiles to no rules at all.
TEMPLATE = """\
# One word or phrase per line, spelled the way you want it written.
# PyWhispr fixes the transcript's spacing and capitalisation to match, and
# will also correct a near miss on longer words.
#
#   BeyondTrust
#   Kubernetes
#   Jamf
#
# When the model mishears a word the same way every time, spell out the fix:
#
#   pie whisper => PyWhispr
#
# Lines starting with # are ignored.

"""


def vocabulary_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / "vocabulary.txt"


def load_vocabulary_text(path: Path | None = None) -> str:
    """The file's raw text, or "" if there isn't one (or it can't be read).

    Raw text is the source of truth rather than the parsed rules, so the user's
    comments, ordering and blank lines survive a round trip through the editor.
    """
    path = path or vocabulary_path()
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        log.exception("Could not read the vocabulary file at %s", path)
        return ""


def save_vocabulary_text(text: str, path: Path | None = None) -> None:
    path = path or vocabulary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def load_vocabulary(path: Path | None = None) -> list[Rule]:
    """Compiled rules from the vocabulary file; an empty list on any problem.

    Never raises: a broken vocabulary file must cost the user their custom
    spellings, not their dictation.
    """
    try:
        rules = parse_vocabulary(load_vocabulary_text(path))
    except Exception:
        log.exception("Could not parse the vocabulary file; continuing without it")
        return []
    if rules:
        log.info("Loaded %d vocabulary term(s)", len(rules))
    return rules
