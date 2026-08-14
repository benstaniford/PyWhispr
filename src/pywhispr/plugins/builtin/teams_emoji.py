"""Teams custom emoji: your organisation's own, which have no Unicode codepoint.

A plugin in its own right rather than a helper :mod:`emoji` imports, and the two
never refer to each other. They cooperate through the framework instead, which
already has the mechanism for it: **a plugin that returns ``None`` declines, and the
words are offered to the next plugin that matched them.** So this one is asked first,
answers for the names you have captured, and says nothing about the rest — and
:mod:`emoji` picks those up with its Unicode tiers.

Being asked *first* is the part that matters, and it is not a detail. ``emoji``'s
last tier is edit-distance matching, which by design answers almost anything: "frown"
lands on "crown" and "shipit" on "ship". Anything running after it would therefore
almost never get a turn. :data:`ALTITUDE` is what makes the order explicit rather
than a consequence of ``BUILTINS`` happening to list this module first.

The store is read but never written. :func:`extract` is the working half — hand it
the clipboard's HTML and it finds the fragment — and it deliberately has no caller,
because every capture route was worse than the gap: ``act`` runs after the injector
has already replaced the clipboard, ``rewrite`` must be reentrant and I/O-free and
also runs on API request threads, a ``pywhispr`` subcommand would put Teams into the
main program's command surface, and a tray entry needs plugins to declare menu
actions generically. Hand-editing ``custom_emoji.json`` works meanwhile.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from platformdirs import user_config_dir

from pywhispr.config import APP_NAME
from pywhispr.plugins.api import Match, Rewrite, Trigger

NAME = "teams_emoji"

# The same words emoji answers to, because a request looks identical either way —
# the difference is only whether Unicode happens to have a character for it.
TRIGGER_WORDS = ("emoji", "emote")
TRIGGERS = tuple(Trigger(phrase=word) for word in TRIGGER_WORDS)

# Below emoji's default of 0, so this is asked first. That is the only reason this
# module declares an altitude at all: emoji is content with the default and says
# nothing, while this one is making a claim about priority and has to.
#
# It has to be first because emoji's fuzzy tier answers almost anything — "frown"
# lands on "crown", "shipit" on "ship" — so a plugin asked afterwards would
# essentially never get a turn. Declared here rather than left to BUILTINS order,
# which would make the behaviour depend on a tuple in another module.
ALTITUDE = -10

# Most words a stored name may span, as in emoji.
MAX_PHRASE_WORDS = 4

# Teams' own id for each standard emoji, which is the one thing that has to be
# verbatim. Everything else about the element is templated, because the asset src
# turned out to be decoration: a deliberately wrong URL on a right id still
# rendered, so Teams re-derives from the id alone.
#
# Hardcoded rather than captured per user, because these are org-independent — the
# CDN path carries no tenant, unlike a custom emoji's.
#
# **Nothing here is derivable, and that is not for want of trying.** Ids come in two
# shapes, and Teams' own isOldEmoticonName export exists to tell them apart:
#
#     "yes"                          👍 — Teams calls the thumbs up "yes"
#     "devil"                        an older, bare short name — the animated art
#     "1f47f_angryfacewithhorns"     newer: codepoint plus modern CLDR name
#
# Guessing from unicodedata reproduced 59 of 83 known ids and missed every common one,
# and "1f44d_thumbsup" 404s at every rendition — the thumbs up is "yes" and the thumbs
# down is "no", which no naming rule was ever going to reach. These were read out of Teams' own map in
# its cached web bundle, then every one confirmed to resolve on the CDN — which caught
# two harvest artefacts, face_enrollment and feed_loaded, whose "codepoints" were just
# the valid hex in "face" and "feed".
#
# **A resolving id is not necessarily the right picture.** A bare id is a *reaction*,
# keyed by what it means rather than what it draws, and for some codepoints those
# differ. Verified by fetching the asset and looking at it:
#
#     heart, cwl, devil, cool, hearteyes   the emoji, animated — which is the point
#     like                                 a face *holding* a thumb, not a thumbs up
#     hi, highfive                         a face performing the gesture
#     coolkoala, cooldog, laughdog         Teams' own animals, one for a *face* codepoint
#
# No rule separates them: "keep faces" admits laughdog, "keep where the name agrees"
# admits like. So the wrong ones were excluded by hand, and roughly 90 reaction ids here
# have *not* been inspected individually. A mismatch is cosmetic and obvious, and the fix
# is to delete a line — but it is a known limit rather than an oversight.
#
# That verification matters more than tidiness for a different reason too: **an id Teams
# does not recognise makes it refuse the entire paste, silently** — not a degraded
# paste, nothing at all.
#
# **This table is maintained by hand.** There is deliberately no regenerate script: it
# would have to scrape Teams' private WebView2 cache, whose layout Microsoft is free to
# change, and its output would still need a human to look at each new picture. To add an
# entry: copy the emoji out of Teams, read the itemid and itemscope from the clipboard's
# CF_HTML (richclip.get_html shows it), check the asset at
# statics.teams.cdn.office.net/evergreen-assets/personal-expressions/v2/assets/emoticons
# /<itemid>/default/50_f.png really draws that emoji, and add the line.
NATIVE_IDS: dict[str, str] = {
    "\U00002600": "2600_sunwithrays",  # Black Sun With Rays
    "\U00002601": "2601_cloud",  # Cloud
    "\U00002620": "2620_skullandcrossbones",  # Skull And Crossbones
    "\U00002639": "2639_frowningface",  # White Frowning Face
    "\U00002699": "2699_gear",  # Gear
    "\U000026a0": "26a0_warningsign",  # Warning Sign
    "\U000026a1": "26a1_highvoltagesign",  # High Voltage Sign
    "\U000026d3": "26d3_chains",  # Chains
    "\U00002705": "2705_whiteheavycheckmark",  # White Heavy Check Mark
    "\U0000270f": "270f_pencil",  # Pencil
    "\U00002714": "2714_heavycheckmark",  # Heavy Check Mark
    "\U00002716": "2716_heavymultiplicationx",  # Heavy Multiplication X
    "\U00002728": "2728_sparkles",  # Sparkles
    "\U0000274c": "274c_crossmark",  # Cross Mark
    "\U00002753": "2753_blackquestionmarkornament",  # Black Question Mark Ornament
    "\U00002757": "2757_heavyexclamationmarksymbol",  # Heavy Exclamation Mark Symbol
    "\U00002763": "2763_heartexclamation",  # Heavy Heart Exclamation Mark O
    "\U00002764": "heart",  # Heavy Black Heart  [reaction art]
    "\U00002795": "2795_heavyplussign",  # Heavy Plus Sign
    "\U000027a1": "27a1_blackrightwardsarrow",  # Black Rightwards Arrow
    "\U00002b05": "2b05_leftwardsblackarrow",  # Leftwards Black Arrow
    "\U00002b06": "2b06_upwardsblackarrow",  # Upwards Black Arrow
    "\U00002b07": "2b07_downwardsblackarrow",  # Downwards Black Arrow
    "\U0001f319": "1f319_crescentmoon",  # Crescent Moon
    "\U0001f33d": "1f33d_earofmaize",  # Ear Of Maize
    "\U0001f369": "1f369_doughnut",  # Doughnut
    "\U0001f372": "1f372_potoffood",  # Pot Of Food
    "\U0001f37b": "1f37b_clinkingbeermugs",  # Clinking Beer Mugs
    "\U0001f389": "1f389_partypopper",  # Party Popper
    "\U0001f38a": "1f38a_confettiball",  # Confetti Ball
    "\U0001f392": "1f392_schoolsatchel",  # School Satchel
    "\U0001f393": "1f393_graduationcap",  # Graduation Cap
    "\U0001f3a7": "headphones",  # Headphone  [reaction art]
    "\U0001f3b9": "1f3b9_musicalkeyboard",  # Musical Keyboard
    "\U0001f3c5": "1f3c5_sportsmedal",  # Sports Medal
    "\U0001f3d9": "1f3d9_cityscape",  # Cityscape
    "\U0001f3ea": "1f3ea_conveniencestore",  # Convenience Store
    "\U0001f440": "1f440_eyes",  # Eyes
    "\U0001f44d": "yes",  # Thumbs Up Sign  [reaction art]
    "\U0001f44e": "no",  # Thumbs Down Sign  [reaction art]
    "\U0001f479": "1f479_japaneseogre",  # Japanese Ogre
    "\U0001f47a": "1f47a_japanesegoblin",  # Japanese Goblin
    "\U0001f47b": "ghost",  # Ghost  [reaction art]
    "\U0001f47d": "1f47d_extraterrestrialalien",  # Extraterrestrial Alien
    "\U0001f47e": "1f47e_alienmonster",  # Alien Monster
    "\U0001f47f": "1f47f_angryfacewithhorns",  # Imp
    "\U0001f480": "skull",  # Skull  [reaction art]
    "\U0001f48b": "lips",  # Kiss Mark  [reaction art]
    "\U0001f48c": "loveletter",  # Love Letter  [reaction art]
    "\U0001f493": "1f493_beatingheart",  # Beating Heart
    "\U0001f494": "brokenheart",  # Broken Heart  [reaction art]
    "\U0001f495": "twohearts",  # Two Hearts  [reaction art]
    "\U0001f496": "sparklingheart",  # Sparkling Heart  [reaction art]
    "\U0001f497": "growingheart",  # Growing Heart  [reaction art]
    "\U0001f498": "1f498_heartwitharrow",  # Heart With Arrow
    "\U0001f499": "heartblue",  # Blue Heart  [reaction art]
    "\U0001f49a": "heartgreen",  # Green Heart  [reaction art]
    "\U0001f49b": "heartyellow",  # Yellow Heart  [reaction art]
    "\U0001f49c": "heartpurple",  # Purple Heart  [reaction art]
    "\U0001f49d": "1f49d_heartwithribbon",  # Heart With Ribbon
    "\U0001f49e": "1f49e_revolvinghearts",  # Revolving Hearts
    "\U0001f49f": "1f49f_heartdecoration",  # Heart Decoration
    "\U0001f4a1": "idea",  # Electric Light Bulb  [reaction art]
    "\U0001f4a2": "1f4a2_angersymbol",  # Anger Symbol
    "\U0001f4a3": "bomb",  # Bomb  [reaction art]
    "\U0001f4a4": "1f4a4_zzz",  # Sleeping Symbol
    "\U0001f4a5": "1f4a5_collisionsymbol",  # Collision Symbol
    "\U0001f4a6": "1f4a6_splashingsweatsymbol",  # Splashing Sweat Symbol
    "\U0001f4a8": "1f4a8_dashsymbol",  # Dash Symbol
    "\U0001f4a9": "poop",  # Pile Of Poo  [reaction art]
    "\U0001f4ab": "1f4ab_dizzysymbol",  # Dizzy Symbol
    "\U0001f4ac": "speechbubble",  # Speech Balloon  [reaction art]
    "\U0001f4ad": "1f4ad_thoughtballoon",  # Thought Balloon
    "\U0001f4af": "1f4af_hundredpointssymbol",  # Hundred Points Symbol
    "\U0001f4b0": "1f4b0_moneybag",  # Money Bag
    "\U0001f4bb": "1f4bb_personalcomputer",  # Personal Computer
    "\U0001f4c5": "1f4c5_calendar",  # Calendar
    "\U0001f4c8": "1f4c8_chartwithupwardstrend",  # Chart With Upwards Trend
    "\U0001f4cb": "1f4cb_clipboard",  # Clipboard
    "\U0001f4cc": "1f4cc_pushpin",  # Pushpin
    "\U0001f4ce": "1f4ce_paperclip",  # Paperclip
    "\U0001f4d5": "1f4d5_closedbook",  # Closed Book
    "\U0001f4e7": "1f4e7_email",  # E-Mail Symbol
    "\U0001f4f7": "1f4f7_camera",  # Camera
    "\U0001f511": "1f511_key",  # Key
    "\U0001f517": "1f517_linksymbol",  # Link Symbol
    "\U0001f51c": "1f51c_soon",  # Soon With Rightwards Arrow Abo
    "\U0001f525": "fire",  # Fire  [reaction art]
    "\U0001f526": "1f526_electrictorch",  # Electric Torch
    "\U0001f527": "1f527_wrench",  # Wrench
    "\U0001f528": "1f528_hammer",  # Hammer
    "\U0001f52b": "1f52b_pistol",  # Pistol
    "\U0001f573": "1f573_hole",  # Hole
    "\U0001f5a4": "heartblack",  # Black Heart  [reaction art]
    "\U0001f5d1": "1f5d1_wastebasket",  # Wastebasket
    "\U0001f5e1": "1f5e1_daggerknife",  # Dagger Knife
    "\U0001f5e8": "1f5e8_leftspeechbubble",  # Left Speech Bubble
    "\U0001f5ef": "1f5ef_rightangerbubble",  # Right Anger Bubble
    "\U0001f601": "1f601_beamingfacewithsmilingeyes",  # Grinning Face With Smiling Eye
    "\U0001f602": "cwl",  # Face With Tears Of Joy  [reaction art]
    "\U0001f603": "1f603_grinningfacewithbigeyes",  # Smiling Face With Open Mouth
    "\U0001f605": "sweatgrinning",  # Smiling Face With Open Mouth A  [reaction art]
    "\U0001f606": "laugh",  # Smiling Face With Open Mouth A  [reaction art]
    "\U0001f607": "angel",  # Smiling Face With Halo  [reaction art]
    "\U0001f608": "devil",  # Smiling Face With Horns  [reaction art]
    "\U0001f609": "wink",  # Winking Face  [reaction art]
    "\U0001f60a": "smileeyes",  # Smiling Face With Smiling Eyes  [reaction art]
    "\U0001f60b": "tongueout",  # Face Savouring Delicious Food  [reaction art]
    "\U0001f60c": "relieved",  # Relieved Face  [reaction art]
    "\U0001f60d": "hearteyes",  # Smiling Face With Heart-Shaped  [reaction art]
    "\U0001f60e": "cool",  # Smiling Face With Sunglasses  [reaction art]
    "\U0001f60f": "smirk",  # Smirking Face  [reaction art]
    "\U0001f610": "speechless",  # Neutral Face  [reaction art]
    "\U0001f611": "expressionless",  # Expressionless Face  [reaction art]
    "\U0001f612": "unamused",  # Unamused Face  [reaction art]
    "\U0001f613": "sweat",  # Face With Cold Sweat  [reaction art]
    "\U0001f614": "pensive",  # Pensive Face  [reaction art]
    "\U0001f615": "confused",  # Confused Face  [reaction art]
    "\U0001f616": "veryconfused",  # Confounded Face  [reaction art]
    "\U0001f617": "kiss",  # Kissing Face  [reaction art]
    "\U0001f618": "1f618_facethrowingakiss",  # Face Throwing A Kiss
    "\U0001f619": "1f619_kissingfacewithsmilingeyes",  # Kissing Face With Smiling Eyes
    "\U0001f61a": "1f61a_kissingfacewithclosedeyes",  # Kissing Face With Closed Eyes
    "\U0001f61b": "1f61b_facewithtongue",  # Face With Stuck-Out Tongue
    "\U0001f61c": "winktongueout",  # Face With Stuck-Out Tongue And  [reaction art]
    "\U0001f61d": "squintingfacewithtongue",  # Face With Stuck-Out Tongue And  [reaction art]
    "\U0001f61e": "disappointed",  # Disappointed Face  [reaction art]
    "\U0001f620": "angry",  # Angry Face  [reaction art]
    "\U0001f621": "angryface",  # Pouting Face  [reaction art]
    "\U0001f622": "cry",  # Crying Face  [reaction art]
    "\U0001f624": "1f624_facewithlookoftriumph",  # Face With Look Of Triumph
    "\U0001f625": "1f625_sadbutrelievedface",  # Disappointed But Relieved Face
    "\U0001f626": "1f626_frowningfacewithopenmouth",  # Frowning Face With Open Mouth
    "\U0001f627": "worry",  # Anguished Face  [reaction art]
    "\U0001f628": "fearful",  # Fearful Face  [reaction art]
    "\U0001f629": "weary",  # Weary Face  [reaction art]
    "\U0001f62a": "sleepy",  # Sleepy Face  [reaction art]
    "\U0001f62b": "doh",  # Tired Face  [reaction art]
    "\U0001f62c": "1f62c_grimacingface",  # Grimacing Face
    "\U0001f62d": "loudlycrying",  # Loudly Crying Face  [reaction art]
    "\U0001f62e": "surprised",  # Face With Open Mouth  [reaction art]
    "\U0001f62f": "1f62f_hushedface",  # Hushed Face
    "\U0001f630": "1f630_anxiousfacewithsweat",  # Face With Open Mouth And Cold 
    "\U0001f631": "screamingfear",  # Face Screaming In Fear  [reaction art]
    "\U0001f632": "1f632_astonishedface",  # Astonished Face
    "\U0001f634": "sleepingface",  # Sleeping Face  [reaction art]
    "\U0001f635": "1f635_dizzyface",  # Dizzy Face
    "\U0001f636": "blankface",  # Face Without Mouth  [reaction art]
    "\U0001f637": "1f637_facewithmedicalmask",  # Face With Medical Mask
    "\U0001f639": "1f639_catwithtearsofjoy",  # Cat Face With Tears Of Joy
    "\U0001f63c": "1f63c_catwithwrysmile",  # Cat Face With Wry Smile
    "\U0001f63d": "1f63d_kissingcat",  # Kissing Cat Face With Closed E
    "\U0001f63e": "1f63e_poutingcat",  # Pouting Cat Face
    "\U0001f640": "1f640_wearycat",  # Weary Cat Face
    "\U0001f641": "sad",  # Slightly Frowning Face  [reaction art]
    "\U0001f642": "smile",  # Slightly Smiling Face  [reaction art]
    "\U0001f643": "upsidedownface",  # Upside-Down Face  [reaction art]
    "\U0001f644": "dull",  # Face With Rolling Eyes  [reaction art]
    "\U0001f648": "seenoevil",  # See-No-Evil Monkey  [reaction art]
    "\U0001f649": "hearnoevil",  # Hear-No-Evil Monkey  [reaction art]
    "\U0001f64a": "speaknoevil",  # Speak-No-Evil Monkey  [reaction art]
    "\U0001f6a2": "1f6a2_ship",  # Ship
    "\U0001f6a7": "1f6a7_constructionsign",  # Construction Sign
    "\U0001f6a8": "1f6a8_policecarsrevolvinglight",  # Police Cars Revolving Light
    "\U0001f6cb": "1f6cb_couchandlamp",  # Couch And Lamp
    "\U0001f90d": "heartwhite",  # White Heart  [reaction art]
    "\U0001f90e": "heartbrown",  # Brown Heart  [reaction art]
    "\U0001f90f": "1f90f_pinchinghand",  # Pinching Hand
    "\U0001f910": "1f910_zippermouthface",  # Zipper-Mouth Face
    "\U0001f911": "1f911_moneymouthface",  # Money-Mouth Face
    "\U0001f912": "ill",  # Face With Thermometer  [reaction art]
    "\U0001f913": "nerdy",  # Nerd Face  [reaction art]
    "\U0001f914": "think",  # Thinking Face  [reaction art]
    "\U0001f915": "1f915_facewithheadbandage",  # Face With Head-Bandage
    "\U0001f916": "smilerobot",  # Robot Face  [reaction art]
    "\U0001f917": "1f917_huggingface",  # Hugging Face
    "\U0001f918": "rock",  # Sign Of The Horns  [reaction art]
    "\U0001f919": "call",  # Call Me Hand  [reaction art]
    "\U0001f91e": "fingerscrossed",  # Hand With Index And Middle Fin  [reaction art]
    "\U0001f920": "1f920_facewithcowboyhat",  # Face With Cowboy Hat
    "\U0001f921": "1f921_clownface",  # Clown Face
    "\U0001f922": "1f922_nauseatedface",  # Nauseated Face
    "\U0001f923": "rofl",  # Rolling On The Floor Laughing  [reaction art]
    "\U0001f924": "1f924_droolingface",  # Drooling Face
    "\U0001f925": "1f925_lyingface",  # Lying Face
    "\U0001f926": "facepalm",  # Face Palm  [reaction art]
    "\U0001f927": "1f927_sneezingface",  # Sneezing Face
    "\U0001f928": "wonder",  # Face With One Eyebrow Raised  [reaction art]
    "\U0001f929": "stareyes",  # Grinning Face With Star Eyes  [reaction art]
    "\U0001f92a": "1f92a_zanyface",  # Grinning Face With One Large A
    "\U0001f92b": "lipssealed",  # Face With Finger Covering Clos  [reaction art]
    "\U0001f92c": "swear",  # Serious Face With Symbols Cove  [reaction art]
    "\U0001f92d": "giggle",  # Smiling Face With Smiling Eyes  [reaction art]
    "\U0001f92e": "puke",  # Face With Open Mouth Vomiting  [reaction art]
    "\U0001f92f": "1f92f_explodinghead",  # Shocked Face With Exploding He
    "\U0001f933": "selfie",  # Selfie  [reaction art]
    "\U0001f94a": "punch",  # Boxing Glove  [reaction art]
    "\U0001f970": "inlove",  # Smiling Face With Smiling Eyes  [reaction art]
    "\U0001f971": "hungover",  # Yawning Face  [reaction art]
    "\U0001f972": "smilingfacewithtear",  # Smiling Face With Tear  [reaction art]
    "\U0001f973": "party",  # Face With Party Horn And Party  [reaction art]
    "\U0001f975": "1f975_hotface",  # Overheated Face
    "\U0001f976": "shivering",  # Freezing Face  [reaction art]
    "\U0001f978": "disguisedface",  # Disguised Face  [reaction art]
    "\U0001f979": "faceholdingbacktears",  # Face Holding Back Tears  [reaction art]
    "\U0001f97a": "1f97a_pleadingface",  # Face With Pleading Eyes
    "\U0001f9d0": "1f9d0_facewithmonocle",  # Face With Monocle
    "\U0001f9e0": "1f9e0_brain",  # Brain
    "\U0001f9e1": "heartorange",  # Orange Heart  [reaction art]
    "\U0001f9e9": "1f9e9_jigsaw",  # Jigsaw Puzzle Piece
    "\U0001fa75": "heartlightblue",  # Light Blue Heart  [reaction art]
    "\U0001fa76": "heartgrey",  # Grey Heart  [reaction art]
    "\U0001fa77": "heartpink",  # Pink Heart  [reaction art]
    "\U0001fae0": "meltingface",  # Melting Face  [reaction art]
    "\U0001fae1": "salute",  # Saluting Face  [reaction art]
    "\U0001fae2": "handovermouth",  # Face With Open Eyes And Hand O  [reaction art]
    "\U0001fae3": "peekingeye",  # Face With Peeking Eye  [reaction art]
    "\U0001fae4": "diagonalmouth",  # Face With Diagonal Mouth  [reaction art]
    "\U0001fae8": "shaking",  # Shaking Face  [reaction art]
}

log = logging.getLogger(__name__)

# Teams wraps *both* kinds of emoji in this element, distinguished only by itemtype:
# ".../CustomEmoji" for your organisation's uploads, ".../Emoji" for the standard set.
# Both are worth having, for different reasons.
#
# A custom one has no codepoint at all, so it can only be asked for by name. A
# standard one has a codepoint — which is what should still arrive in Slack or an
# email — but Teams renders its own asset better than the bare character, so it is
# worth *decorating* rather than replacing. Hence the two stores below, keyed
# differently: names for the first, the codepoint itself for the second.
#
# Neither can be derived. The asset slug in the URL is "1f440_eyes" — Teams' own
# shortname, not Unicode's — and every attempt to guess one 404s off the CDN
# ("1f44d_thumbsup", "1f44d_thumbsupsign" and friends were all tried). So the markup
# has to come from a real copy, and both stores stay empty until something captures.
#
# The element Teams wraps either kind of emoji in. Matched rather than parsed: this
# is a fragment of a foreign application's clipboard format, and a regex that either
# finds the marker or does not is a smaller thing to get wrong than an HTML parser
# whose failure mode is a plausible-looking wrong answer.
_FRAGMENT = re.compile(
    r"<readonly\b[^>]*itemtype\s*=\s*[\"']http://schema\.skype\.com/"
    r"(?P<custom>Custom)?Emoji[\"'][^>]*>.*?</readonly>",
    re.IGNORECASE | re.DOTALL,
)

# The alt/aria label Teams puts on the image, used to suggest a name.
_LABEL = re.compile(r"alt\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

# The codepoint Teams keeps on a standard emoji's element. The whole reason standard
# emoji are capturable at all: it hands us the character to key the markup against.
_ITEMSCOPE = re.compile(r"itemscope\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

# A spoken name has to survive being said and matched. Same normalising idea as the
# emoji plugin's: lower case, no punctuation, single spaces.
_TIDY = re.compile(r"[^\w\s]|_")
_SPACING = re.compile(r"\s+")


def store_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / "custom_emoji.json"


def normalise(name: str) -> str:
    return _SPACING.sub(" ", _TIDY.sub(" ", name.casefold())).strip()


def extract(clipboard_html: str) -> tuple[str, str, str] | None:
    """Pull ``(kind, key, fragment)`` out of clipboard HTML, or None.

    ``kind`` is ``"name"`` for a custom emoji, which has no codepoint and can only be
    asked for by name — the key is then the image's label, a suggestion the caller may
    override. ``kind`` is ``"character"`` for a standard one, and the key is the
    codepoint from its ``itemscope``, which is not a suggestion at all: it is the
    character this markup is a better rendering *of*.
    """
    if not clipboard_html:
        return None
    found = _FRAGMENT.search(clipboard_html)
    if found is None:
        return None
    fragment = found.group(0)
    if found.group("custom") is None:  # a standard emoji: key on its codepoint
        scope = _ITEMSCOPE.search(fragment)
        if scope is None:
            return None
        return "character", scope.group(1), fragment
    label = _LABEL.search(fragment)
    return "name", (label.group(1) if label else ""), fragment


@dataclass(frozen=True)
class Stored:
    """What has been captured, keyed the two different ways it has to be.

    ``names`` is for custom emoji, which have no codepoint and can only be asked for
    by name. ``characters`` is for standard ones, keyed by the codepoint their markup
    is a better rendering of.
    """

    names: dict[str, str] = field(default_factory=dict)
    characters: dict[str, str] = field(default_factory=dict)


def load(path: Path | None = None) -> Stored:
    """Everything captured, or empty. Never raises.

    A corrupt store costs the user their emoji, not their dictation — the same bargain
    :func:`pywhispr.vocab.load_vocabulary` strikes.

    A flat ``{name: markup}`` file is read as names, because that is the shape this
    store had before standard emoji were worth keeping and there is no reason to make
    anyone's file invalid over it.
    """
    path = path or store_path()
    try:
        if not path.exists():
            return Stored()
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read the emoji store at %s", path)
        return Stored()
    if not isinstance(data, dict):
        log.error("Emoji store is not an object; ignoring it")
        return Stored()

    sections = data if "names" in data or "characters" in data else {"names": data}
    names = _strings(sections.get("names"), normalise)
    characters = _strings(sections.get("characters"), lambda key: key)
    # Counts only. The names are the user's own and the fragments carry asset URLs, so
    # neither belongs in a log they might send us.
    if names or characters:
        log.info("Loaded %d named and %d standard emoji", len(names), len(characters))
    return Stored(names=names, characters=characters)


def _strings(section: object, key_of) -> dict[str, str]:
    """The string-to-string entries of `section`, skipping anything unusable."""
    if not isinstance(section, dict):
        return {}
    return {
        key_of(key): markup
        for key, markup in section.items()
        if isinstance(key, str) and isinstance(markup, str) and markup and key_of(key)
    }


def save(stored: Stored, path: Path | None = None) -> None:
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"names": stored.names, "characters": stored.characters}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def remember(kind: str, key: str, fragment: str, path: Path | None = None) -> str:
    """Store `fragment`, returning the key it went under.

    ``kind`` is what :func:`extract` reported: a name is normalised so it can be
    spoken and matched, a character is kept exactly as it is.
    """
    stored = load(path)
    names = dict(stored.names)
    characters = dict(stored.characters)
    if kind == "character":
        characters[key] = fragment
    else:
        key = normalise(key)
        names[key] = fragment
    save(Stored(names=names, characters=characters), path)
    return key


# -- the plugin itself -------------------------------------------------------


@lru_cache(maxsize=1)
def _store() -> Stored:
    """Read once. Nobody who has captured nothing should pay for the file."""
    return load()


def rewrite(match: Match) -> Rewrite | None:
    """Replace "<stored name> emoji" with the tenant's markup.

    Longest phrase first, exactly as :mod:`emoji` does, so a two-word name beats
    its last word. Only the store is consulted — there is no guessing tier here,
    because there is nothing to guess: a name is either one you captured or it is
    not.

    The plain text is the name itself, which is what arrives anywhere HTML does not
    reach, and what Teams' own copy degrades to.
    """
    available = min(len(match.words_before), MAX_PHRASE_WORDS)
    for count in range(available, 0, -1):
        words = match.words_before[-count:]
        phrase = normalise(match.transcript[words[0].start : words[-1].end])
        markup = _store().names.get(phrase)
        if markup is None:
            continue
        # The framework absorbs the punctuation the model invented and puts the
        # separator into both renderings, so this does not have to know about either.
        return match.claim_absorbing(words[0], phrase, html=markup)
    return None


# The element Teams itself produces, with only the id and label varying. Templated
# rather than stored per emoji because the src is decoration and the rest is fixed —
# and because two well-formed elements differing only in src both rendered, which is
# what licenses templating at all. Anything malformed here costs the whole paste, so
# this shape is exactly the one observed on the clipboard, attribute for attribute.
_TEMPLATE = (
    '<readonly contenteditable="false" title="{label}" itemid="{itemid}"'
    ' itemtype="http://schema.skype.com/Emoji" itemscope="{character}"'
    ' aria-label="{label}" data-announcement="{label} emoji">'
    '<img style="margin:0px 1px;vertical-align:bottom;"'
    ' src="https://statics.teams.cdn.office.net/evergreen-assets/personal-expressions'
    '/v2/assets/emoticons/{itemid}/default/50_f.png?v=v10"'
    ' alt="{label}" draggable="false" width="20px" height="20px"></readonly>'
)


def native_markup(character: str) -> str | None:
    """Teams' element for `character`, or None if we have no verified id for it.

    None is the safe answer and the common one. There is deliberately no fallback to
    a guessed id: Teams refuses the entire paste when it does not recognise one, so a
    missing entry must mean "leave the codepoint alone", never "try something".
    """
    itemid = NATIVE_IDS.get(character)
    if itemid is None:
        return None
    label = (unicodedata.name(character, "emoji") or "emoji").title()
    return _TEMPLATE.format(itemid=itemid, character=character, label=_escaped(label))


def _escaped(value: str) -> str:
    """Keep a label from breaking out of the attribute it sits in."""
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def decorate(text: str) -> list[tuple[int, int, str]]:
    """Point Teams at its own asset for the emoji in the finished text.

    Not a rewrite, deliberately: the codepoint stays in the text, so Slack, an email
    or a plain-text field still get a perfectly good emoji. This only says "where HTML
    is accepted, here is a better rendering of that same character" — Teams draws its
    own asset differently from the bare codepoint, which is the whole reason to bother.

    A captured entry wins over the shipped table, so an organisation that has replaced
    a standard emoji gets its own.
    """
    captured = _store().characters
    found: list[tuple[int, int, str]] = []
    for index, character in enumerate(text):
        markup = captured.get(character) or native_markup(character)
        if markup is not None:
            found.append((index, index + 1, markup))
    return found
