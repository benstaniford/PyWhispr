"""Say "thumbs up emoji" and get the character.

The words before the trigger are looked up as an emoji name, longest phrase
first, and the whole lot — words and trigger — becomes one character. Nothing
resolves, nothing happens: "send me an emoji" has no emoji name in front of it,
so :func:`rewrite` returns ``None`` and the transcript is left exactly as it was.
That is the guard, and it belongs here rather than in the engine because only
this module knows what counts as an emoji name.

Two tiers, and the order matters:

1. :data:`ALIASES` — what people actually say. Hand-written, and needed for two
   reasons the Unicode data cannot fix. The standard library carries the
   *original* Unicode 6 names, so ❤ is ``HEAVY BLACK HEART`` and "red heart"
   finds nothing at all; and where a name does exist it is often not the one
   meant — "smiling face" resolves by prefix to ``SMILING FACE WITH HALO``, which
   is a different sentiment entirely. The aliases also pin the everyday cases
   against ``unicodedata`` drifting between Python versions.
2. The Unicode names themselves, indexed on first use. This is the long tail, and
   it is free: no data file, no dependency, ~3,000 names in a couple of
   milliseconds. Exact name, then a prefix at a word boundary ("waving hand" →
   ``WAVING HAND SIGN``), then a name containing all the words asked for ("pizza"
   → ``SLICE OF PIZZA``). The shortest name wins, being the least qualified.

Characters are written as escapes rather than literals so that this file is pure
ASCII: it has to survive editors, terminals and code review on three platforms,
and ``\\U0001F44D`` cannot be mangled by any of them.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from pywhispr.join import CONTINUATION_WORDS
from pywhispr.plugins.api import Match, Rewrite, Trigger, Word
from pywhispr.scratch import SEGMENT_BOUNDARY

NAME = "emoji"

TRIGGER_WORD = "emoji"

# Every occurrence is offered and :func:`_is_a_request` decides which of them are
# asking for an emoji rather than talking about one. That decision cannot be the
# Trigger's ``at_segment_end``, tempting as it looks: a chain — "man emoji gun
# emoji" — has an ordinary word after its first trigger, so the segment rule threw
# the whole first half away before this module ever saw it.
TRIGGERS = (Trigger(phrase=TRIGGER_WORD),)

# Most words the name may span, counted back from the trigger. Four, like
# vocab.MAX_PHRASE_WORDS: past that a spoken phrase is a sentence.
MAX_PHRASE_WORDS = 4

# Below this, a query is too short to mean anything specific. Guards the index
# tiers, where two characters would prefix-match half of Unicode.
MIN_QUERY_CHARS = 3

# Where the emoji live. Deliberately not 1F000-1F2FF: mahjong tiles, playing
# cards and enclosed letters are all named, all matchable and never what anyone
# dictating meant.
RANGES = (
    (0x1F300, 0x1FAFF),  # pictographs, faces, hands, food, animals, symbols
    (0x231A, 0x23FF),  # watch, hourglass, media controls
    (0x2600, 0x27BF),  # miscellaneous symbols and dingbats
    (0x2B00, 0x2BFF),  # stars and arrows
)

_PUNCTUATION = re.compile(r"[^\w\s]|_")
_SPACING = re.compile(r"\s+")

# Punctuation the model attached that the emoji should not carry.
#
# Every transcript arrives with a full stop appended whether or not the sentence
# wanted one, and with a comma wherever the model heard the pause before the name
# — so "Hello smile emoji" comes back as "Hello, <emoji>." Neither mark is how
# anyone writes an emoji, and the full stop costs something concrete: Teams,
# Slack and the rest only render the large version when the message is *nothing
# but* emoji.
#
# Only these two marks. "!" and "?" are left alone, because the model does not
# add those on its own — they are the speaker's, and an emoji is allowed to end
# an exclamation.
_ABSORB_BEFORE = ","
_ABSORB_AFTER = ".…"
_SPACES = " \t"

# A request ends its clause: punctuation, a line break, or nothing at all after
# it. Built from the same boundary list scratch.py matches the spoken reset phrase
# against, rather than a second copy of it.
_CLAUSE_END = re.compile(rf"\s*(?:{SEGMENT_BOUNDARY}|$)")

# What people say, against what they mean. Keys are normalised: lower case, words
# separated by single spaces.
#
# Deliberately absent: "like", "no" and "done". Each is an ordinary word that
# lands in front of the word "emoji" in sentences *about* emoji — "I like emoji",
# "I want no emoji" — and each already has an unambiguous way to ask for it
# ("thumbs up", "cross", "check"). Same lopsided trade-off as everywhere else
# here: a missed conversion costs a keystroke, a wrong one costs a retype.
ALIASES: dict[str, str] = {
    # faces
    "smile": "\U0001f642",
    "smiley": "\U0001f642",
    "smiling": "\U0001f642",
    "smiling face": "\U0001f642",
    "happy": "\U0001f642",
    "happy face": "\U0001f642",
    "grin": "\U0001f600",
    "big smile": "\U0001f600",
    "laugh": "\U0001f602",
    "laughing": "\U0001f602",
    "crying laughing": "\U0001f602",
    "tears of joy": "\U0001f602",
    "wink": "\U0001f609",
    "winking": "\U0001f609",
    "tongue out": "\U0001f61b",
    "sad": "\U0001f622",
    "sad face": "\U0001f622",
    "cry": "\U0001f622",
    "crying": "\U0001f622",
    "sobbing": "\U0001f62d",
    "shocked": "\U0001f632",
    "surprised": "\U0001f632",
    "worried": "\U0001f61f",
    "angry": "\U0001f620",
    "cross face": "\U0001f620",
    "cool": "\U0001f60e",
    "sunglasses": "\U0001f60e",
    "thinking": "\U0001f914",
    "confused": "\U0001f615",
    "blushing": "\U0001f60a",
    "sleeping": "\U0001f634",
    "sick": "\U0001f637",
    "wearing a mask": "\U0001f637",
    "eye roll": "\U0001f644",
    "rolling eyes": "\U0001f644",
    "facepalm": "\U0001f926",
    "shrug": "\U0001f937",
    "party face": "\U0001f973",
    "star struck": "\U0001f929",
    "pleading": "\U0001f97a",
    "wow": "\U0001f62e",
    # hands and people
    "thumbs up": "\U0001f44d",
    "thumb up": "\U0001f44d",
    "plus one": "\U0001f44d",
    "thumbs down": "\U0001f44e",
    "thumb down": "\U0001f44e",
    "minus one": "\U0001f44e",
    "ok": "\U0001f44c",
    "ok hand": "\U0001f44c",
    "clap": "\U0001f44f",
    "clapping": "\U0001f44f",
    "applause": "\U0001f44f",
    "wave": "\U0001f44b",
    "waving": "\U0001f44b",
    "hello": "\U0001f44b",
    "pray": "\U0001f64f",
    "praying": "\U0001f64f",
    "please": "\U0001f64f",
    "thanks": "\U0001f64f",
    "thank you": "\U0001f64f",
    "folded hands": "\U0001f64f",
    "high five": "\U0001f64c",
    "muscle": "\U0001f4aa",
    "flex": "\U0001f4aa",
    "strong": "\U0001f4aa",
    "fingers crossed": "\U0001f91e",
    "handshake": "\U0001f91d",
    "salute": "\U0001fae1",
    "point right": "\U0001f449",
    "point left": "\U0001f448",
    "point up": "\U0001f446",
    "point down": "\U0001f447",
    # reactions and marks
    "heart": "\U00002764",
    "red heart": "\U00002764",
    "love": "\U00002764",
    "broken heart": "\U0001f494",
    "fire": "\U0001f525",
    "hot": "\U0001f525",
    "hundred": "\U0001f4af",
    "one hundred": "\U0001f4af",
    "hundred percent": "\U0001f4af",
    "party": "\U0001f389",
    "celebrate": "\U0001f389",
    "tada": "\U0001f389",
    "congratulations": "\U0001f389",
    "check": "\U00002705",
    "check mark": "\U00002705",
    "tick": "\U00002705",
    "cross": "\U0000274c",
    "wrong": "\U0000274c",
    "warning": "\U000026a0",
    "question mark": "\U00002753",
    "exclamation mark": "\U00002757",
    "star": "\U00002b50",
    "sparkle": "\U00002728",
    "sparkles": "\U00002728",
    "idea": "\U0001f4a1",
    "light bulb": "\U0001f4a1",
    "boom": "\U0001f4a5",
    "explosion": "\U0001f4a5",
    "lightning": "\U000026a1",
    "zap": "\U000026a1",
    "sleep": "\U0001f4a4",
    "eyes": "\U0001f440",
    "brain": "\U0001f9e0",
    "skull": "\U0001f480",
    "dead": "\U0001f480",
    "ghost": "\U0001f47b",
    "alien": "\U0001f47d",
    "robot": "\U0001f916",
    "poop": "\U0001f4a9",
    "clown": "\U0001f921",
    # things
    "rocket": "\U0001f680",
    "coffee": "\U00002615",
    "tea": "\U0001f375",
    "beer": "\U0001f37a",
    "cheers": "\U0001f37b",
    "wine": "\U0001f377",
    "pizza": "\U0001f355",
    "cake": "\U0001f370",
    "birthday cake": "\U0001f382",
    "money": "\U0001f4b0",
    "gift": "\U0001f381",
    "present": "\U0001f381",
    "bell": "\U0001f514",
    "lock": "\U0001f512",
    "key": "\U0001f511",
    "hourglass": "\U000023f3",
    "clock": "\U0001f552",
    "calendar": "\U0001f4c5",
    "chart": "\U0001f4c8",
    "graph": "\U0001f4c8",
    "laptop": "\U0001f4bb",
    "computer": "\U0001f4bb",
    "phone": "\U0001f4f1",
    "email": "\U0001f4e7",
    "mail": "\U0001f4e7",
    "link": "\U0001f517",
    "search": "\U0001f50d",
    "magnifying glass": "\U0001f50d",
    "wrench": "\U0001f527",
    "hammer": "\U0001f528",
    "gear": "\U00002699",
    "bin": "\U0001f5d1",
    "trash": "\U0001f5d1",
    "book": "\U0001f4d5",
    "pencil": "\U0000270f",
    "clipboard": "\U0001f4cb",
    "pin": "\U0001f4cc",
    "flag": "\U0001f3c1",
    "trophy": "\U0001f3c6",
    "medal": "\U0001f3c5",
    "target": "\U0001f3af",
    "bullseye": "\U0001f3af",
    "bug": "\U0001f41b",
    "snail": "\U0001f40c",
    "dog": "\U0001f436",
    "cat": "\U0001f431",
    "unicorn": "\U0001f984",
    "snake": "\U0001f40d",
    "sun": "\U00002600",
    "moon": "\U0001f319",
    "cloud": "\U00002601",
    "rain": "\U0001f327",
    "snowflake": "\U00002744",
    "rainbow": "\U0001f308",
    "tree": "\U0001f333",
    "flower": "\U0001f338",
    "car": "\U0001f697",
    "plane": "\U00002708",
    "house": "\U0001f3e0",
    "up arrow": "\U00002b06",
    "down arrow": "\U00002b07",
    "left arrow": "\U00002b05",
    "right arrow": "\U000027a1",
    # Words whose emoji the legacy Unicode names call something else entirely, so
    # the index tiers cannot reach them: nobody says "flag in hole" or "hocho".
    "gun": "\U0001f52b",  # PISTOL
    "sword": "\U0001f5e1",  # DAGGER KNIFE
    "golf": "\U000026f3",  # FLAG IN HOLE
    "swimming": "\U0001f3ca",  # SWIMMER
    "cycling": "\U0001f6b4",  # BICYCLIST
    "bike": "\U0001f6b2",  # BICYCLE
    "piano": "\U0001f3b9",  # MUSICAL KEYBOARD
    "headphones": "\U0001f3a7",  # HEADPHONE
    "flashlight": "\U0001f526",  # ELECTRIC TORCH
    "torch": "\U0001f526",
    "sofa": "\U0001f6cb",  # COUCH AND LAMP
    "couch": "\U0001f6cb",
    "backpack": "\U0001f392",  # SCHOOL SATCHEL
    "city": "\U0001f3d9",  # CITYSCAPE
    "cigarette": "\U0001f6ac",  # SMOKING SYMBOL
    "chain": "\U000026d3",  # CHAINS
    "rope": "\U0001faa2",  # KNOT
    "pinch": "\U0001f90f",  # PINCHING HAND
    "burger": "\U0001f354",  # HAMBURGER
    "noodles": "\U0001f35c",  # STEAMING BOWL
    "soup": "\U0001f372",  # POT OF FOOD
    "donut": "\U0001f369",  # DOUGHNUT
    "doughnut": "\U0001f369",
    "corn": "\U0001f33d",  # EAR OF MAIZE
}


def _normalise(phrase: str) -> str:
    """Lower case, punctuation gone, spacing collapsed — "Thumbs-up!" → "thumbs up"."""
    return _SPACING.sub(" ", _PUNCTUATION.sub(" ", phrase.casefold())).strip()


@lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    """Unicode name (lower case) to character, for the ranges emoji live in.

    Built once on the first lookup rather than at import: a user who never says
    "emoji" should not pay for it, even if that price is only milliseconds.
    """
    index: dict[str, str] = {}
    for low, high in RANGES:
        for codepoint in range(low, high + 1):
            char = chr(codepoint)
            name = unicodedata.name(char, None)
            if name is not None:
                index[name.casefold()] = char
    return index


def _shortest(names: list[str], index: dict[str, str]) -> str:
    """The least qualified of several matching names, deterministically.

    Fewest characters means fewest extra words — "waving hand sign" over "waving
    hand with medium skin tone" — and the codepoint breaks a tie so the same
    query cannot answer differently between runs.
    """
    return min(names, key=lambda name: (len(name), ord(index[name])))


@lru_cache(maxsize=512)
def _resolve(phrase: str) -> str | None:
    """The character `phrase` names, or None if it does not name one.

    None is the common answer and not a failure: most words in front of the word
    "emoji" are just words.
    """
    key = _normalise(phrase)
    if len(key) < MIN_QUERY_CHARS:
        return None

    # The alias table first, before any guard: an entry in it is a decision
    # already made, the way an explicit `heard => wanted` line in the vocabulary
    # skips the fuzzy tier's guards. "plus one" is two function words and would
    # not survive the check below, which is exactly why it has to come first.
    alias = ALIASES.get(key)
    if alias is not None:
        return alias

    words = key.split()
    if TRIGGER_WORD in words:
        # The trigger word is not the name of anything. Worth stating outright,
        # because the Unicode data has "EMOJI COMPONENT BALD" and three siblings,
        # so the prefix tier answers "emoji" with a hairstyle.
        return None
    if not words or all(word in CONTINUATION_WORDS for word in words):
        # "an emoji", "of the emoji": function words are never an emoji name, so
        # the index tiers never get to guess at them. The same closed list
        # vocab.py reuses as its never-rewrite guard.
        return None

    index = _index()
    exact = index.get(key)
    if exact is not None:
        return exact

    # A prefix, but only at a word boundary: without it "fire" would be free to
    # match "fireworks".
    prefixed = [name for name in index if name.startswith(f"{key} ")]
    if prefixed:
        return index[_shortest(prefixed, index)]

    wanted = set(words)
    containing = [name for name in index if wanted.issubset(name.split())]
    if containing:
        return index[_shortest(containing, index)]
    return None


def _leads_a_chain(match: Match) -> bool:
    """Is another request right behind this one, as in "man emoji gun emoji"?

    Only when the words between the two triggers name an emoji *in their entirety*.
    That is what tells a chain from a sentence discussing emoji: "gun" resolves, so
    "man emoji gun emoji" is two requests, while the "and the water" of "I use the
    fire emoji and the water emoji" resolves to nothing and stays prose.
    """
    for index, word in enumerate(match.words_after):
        if word.text.casefold() != TRIGGER_WORD:
            continue
        if index == 0 or index > MAX_PHRASE_WORDS:
            return False  # "emoji emoji", or too far off to be one phrase
        run = match.words_after[:index]
        return _resolve(match.transcript[run[0].start : run[-1].end]) is not None
    return False


def _is_a_request(match: Match) -> bool:
    """Is this trigger asking for an emoji, or talking about one?

    Two accepted positions, both structural rather than guesswork:

    - **It ends a clause.** "I like this fire emoji." converts; "the fire emoji is
      best" does not, because there the words are about the emoji. The same guard
      scratch.py puts on the spoken reset phrase, and the same lopsided trade-off:
      it costs the odd mid-sentence conversion and buys immunity from every
      sentence that merely mentions the word.
    - **It leads a chain.** Which is the other way people dictate these — see
      :func:`_leads_a_chain`.
    """
    if _CLAUSE_END.match(match.transcript[match.end :]):
        return True
    return _leads_a_chain(match)


def _claim_span(match: Match, first: Word) -> tuple[int, int, str]:
    """Where the claim runs from and to, plus the separator it has to put back.

    Wider than the name and the trigger: it takes the model's own punctuation with
    it — see :data:`_ABSORB_BEFORE` and :data:`_ABSORB_AFTER`. The trailing mark
    goes only when the emoji ends the transcript, so a comma that separates two
    clauses ("hello, <emoji>, then I left") keeps doing its job.

    Both ends stop at the window the engine gave us, so this can only ever reach
    punctuation adjacent to words this plugin was shown.
    """
    text = match.transcript
    start = first.start
    cursor = start
    while cursor > match.window_start and text[cursor - 1] in _SPACES:
        cursor -= 1
    if cursor > match.window_start and text[cursor - 1] in _ABSORB_BEFORE:
        start = cursor - 1
        while start > match.window_start and text[start - 1] in _SPACES:
            start -= 1

    end = match.end
    if not text[end:].strip(f"{_ABSORB_AFTER}{_SPACES}\r\n"):
        end = match.window_end

    # Absorbing ", " leaves the emoji hard against the previous word, so the
    # separator it removed has to come back as a plain space.
    separator = " " if start > 0 and not text[start - 1].isspace() else ""
    return start, end, separator


def rewrite(match: Match) -> Rewrite | None:
    """Replace "<name> emoji" with the character, or leave the transcript alone.

    Longest phrase first, so "thumbs up" is tried before "up" and wins.
    """
    if not _is_a_request(match):
        return None
    available = min(len(match.words_before), MAX_PHRASE_WORDS)
    for count in range(available, 0, -1):
        words = match.words_before[-count:]
        phrase = match.transcript[words[0].start : words[-1].end]
        character = _resolve(phrase)
        if character is not None:
            start, end, separator = _claim_span(match, words[0])
            return match.claim(start, end, f"{separator}{character}")
    return None
