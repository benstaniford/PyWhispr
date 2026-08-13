"""Teams custom emoji: markup captured from the clipboard, keyed by what you say.

Belongs to the emoji plugin, not to PyWhispr. Nothing outside
:mod:`pywhispr.plugins.builtin.emoji` imports this, and nothing in the main
program knows that Teams or emoji exist — the framework's whole claim is that it
knows nothing about either. The leading underscore says the same thing to the
registry, whose folder scan skips such files, and to anyone reading the directory:
this is a helper, not a plugin.

A custom emoji is not a character. It has no Unicode codepoint — it is an image
hosted by your tenant, referenced by an ``<img itemid=...>`` inside a marker
element. So unlike everything in :mod:`pywhispr.plugins.builtin.emoji`, it cannot
be produced from a name; the exact markup has to come from somewhere.

It can only come from the clipboard: there is no documented API for listing a
tenant's custom emoji, so the alternative is guessing at asset IDs.

**There is currently no supported way to capture one.** :func:`extract` is the
working half — hand it the clipboard's HTML and it finds the fragment — but nothing
calls it, because every candidate route was worse than leaving the gap:

* A spoken trigger cannot work from ``act``: the injector has already replaced the
  clipboard with the transcript by then, so the markup is gone.
* It cannot work from ``rewrite`` either, which must be reentrant and do no I/O,
  and which runs on API request threads where reading *this* machine's clipboard
  would be nonsense.
* A ``pywhispr`` subcommand would put Teams and emoji into the main program's
  command surface, which is exactly what the plugin framework exists to avoid.
* A tray or settings entry has the same problem unless plugins can declare menu
  actions generically — a framework feature nobody has asked for yet.

So the store is read but not written. Populating ``custom_emoji.json`` by hand
works, and :func:`extract` plus :func:`remember` are ready for whichever route wins,
but in practice this is a developer-only path today.

Two consequences worth stating plainly:

* **The fragments are tenant-scoped.** Their image URLs are fetched with the
  viewer's credentials, so a stored fragment works for colleagues in the same
  tenant and for nobody else. This file is not shareable configuration.
* **JSON, not the line-based format vocabulary.txt uses.** These fragments are
  machine-captured, several hundred characters long and full of quotes; a
  hand-editable one-per-line file would be a fiction. Deleting an entry by hand is
  still easy, which is the part that matters.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from platformdirs import user_config_dir

from pywhispr.config import APP_NAME

log = logging.getLogger(__name__)

# The element Teams wraps a custom emoji in. Matched rather than parsed: this is a
# fragment of a foreign application's clipboard format, and a regex that either
# finds the marker or does not is a smaller thing to get wrong than an HTML parser
# whose failure mode is a plausible-looking wrong answer.
_FRAGMENT = re.compile(
    r"<readonly\b[^>]*itemtype\s*=\s*[\"']http://schema\.skype\.com/CustomEmoji[\"'][^>]*>"
    r".*?</readonly>",
    re.IGNORECASE | re.DOTALL,
)

# The alt/aria label Teams puts on the image, used to suggest a name.
_LABEL = re.compile(r"alt\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

# A spoken name has to survive being said and matched. Same normalising idea as the
# emoji plugin's: lower case, no punctuation, single spaces.
_TIDY = re.compile(r"[^\w\s]|_")
_SPACING = re.compile(r"\s+")


def store_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / "custom_emoji.json"


def normalise(name: str) -> str:
    return _SPACING.sub(" ", _TIDY.sub(" ", name.casefold())).strip()


def extract(clipboard_html: str) -> tuple[str, str] | None:
    """Pull ``(suggested name, fragment)`` out of clipboard HTML, or None.

    The suggested name is the image's alt text, which is what Teams shows and
    therefore what a user is likely to call it — but it is only a suggestion; the
    caller decides what to key it under.
    """
    if not clipboard_html:
        return None
    found = _FRAGMENT.search(clipboard_html)
    if found is None:
        return None
    fragment = found.group(0)
    label = _LABEL.search(fragment)
    return (label.group(1) if label else ""), fragment


def load(path: Path | None = None) -> dict[str, str]:
    """Every stored emoji, keyed by normalised name. Never raises.

    A corrupt store costs the user their custom emoji, not their dictation — the
    same bargain :func:`pywhispr.vocab.load_vocabulary` strikes.
    """
    path = path or store_path()
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read the custom emoji store at %s", path)
        return {}
    if not isinstance(data, dict):
        log.error("Custom emoji store is not an object; ignoring it")
        return {}
    # Counts only. The names are the user's own, and the fragments carry tenant
    # asset URLs, so neither belongs in a log they might send us.
    entries = {
        normalise(name): markup
        for name, markup in data.items()
        if isinstance(name, str) and isinstance(markup, str) and markup
    }
    if entries:
        log.info("Loaded %d custom emoji", len(entries))
    return entries


def save(entries: dict[str, str], path: Path | None = None) -> None:
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def remember(name: str, fragment: str, path: Path | None = None) -> str:
    """Store `fragment` under `name`, returning the key it was stored as."""
    entries = load(path)
    key = normalise(name)
    entries[key] = fragment
    save(entries, path)
    return key
