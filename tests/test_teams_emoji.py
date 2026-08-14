"""Capturing Teams custom emoji off the clipboard, and storing them.

Never touches the developer's real store: every test passes an explicit tmp_path.
"""

from __future__ import annotations

from pywhispr.plugins.builtin import teams_emoji as custom_emoji

FRAGMENT = (
    '<readonly aria-label="jackl_frown" contenteditable="false"'
    ' itemtype="http://schema.skype.com/CustomEmoji">'
    '<img alt="jackl_frown" itemid="jackl_frown;0-eus-d20-abc" src="https://x/y">'
    "</readonly>"
)
CLIPBOARD = f"<html>\r\n<body>\r\n<!--StartFragment-->{FRAGMENT}<!--EndFragment-->\r\n</body>\r\n</html>"


EYES = "👀"
STANDARD = (
    '<readonly contenteditable="false" title="Eyes" itemid="1f440_eyes"'
    ' itemtype="http://schema.skype.com/Emoji" itemscope="' + EYES + '" aria-label="Eyes">'
    '<img alt="Eyes" src="https://statics.example.invalid/1f440_eyes.png"></readonly>'
)


class TestExtract:
    def test_finds_a_custom_emoji_and_its_label(self):
        kind, key, fragment = custom_emoji.extract(CLIPBOARD)
        assert (kind, key) == ("name", "jackl_frown")
        assert fragment == FRAGMENT

    def test_finds_a_standard_emoji_and_keys_it_on_the_codepoint(self):
        """itemscope carries the character, which is why standard ones are capturable."""
        kind, key, fragment = custom_emoji.extract(STANDARD)
        assert (kind, key) == ("character", EYES)
        assert fragment == STANDARD

    def test_a_standard_emoji_without_itemscope_is_refused(self):
        """No codepoint means nothing to key the markup against."""
        assert custom_emoji.extract(STANDARD.replace(' itemscope="' + EYES + '"', "")) is None

    def test_finds_it_among_other_content(self):
        wrapped = f"<html><body>hello <b>there</b> {FRAGMENT} goodbye</body></html>"
        assert custom_emoji.extract(wrapped)[2] == FRAGMENT

    def test_html_without_a_custom_emoji_is_none(self):
        assert custom_emoji.extract("<html><b>just bold</b></html>") is None

    def test_a_plain_image_is_not_a_custom_emoji(self):
        """The itemtype marker is the whole signal; any old <img> is not one."""
        assert custom_emoji.extract('<html><img src="cat.png" alt="cat"></html>') is None

    def test_empty_input_is_none(self):
        assert custom_emoji.extract("") is None

    def test_a_fragment_with_no_label_still_extracts(self):
        bare = (
            '<readonly itemtype="http://schema.skype.com/CustomEmoji">'
            '<img itemid="x;y" src="https://x/y"></readonly>'
        )
        kind, key, fragment = custom_emoji.extract(bare)
        assert (kind, key) == ("name", "")
        assert fragment == bare


class TestStore:
    def test_a_name_round_trips(self, tmp_path):
        path = tmp_path / "emoji.json"
        custom_emoji.remember("name", "jackl frown", FRAGMENT, path)
        assert custom_emoji.load(path).names == {"jackl frown": FRAGMENT}

    def test_a_character_round_trips_unnormalised(self, tmp_path):
        """A codepoint must be kept exactly: normalising would strip it to nothing."""
        path = tmp_path / "emoji.json"
        custom_emoji.remember("character", EYES, STANDARD, path)
        assert custom_emoji.load(path).characters == {EYES: STANDARD}

    def test_the_two_sections_do_not_collide(self, tmp_path):
        path = tmp_path / "emoji.json"
        custom_emoji.remember("name", "frown", FRAGMENT, path)
        custom_emoji.remember("character", EYES, STANDARD, path)
        stored = custom_emoji.load(path)
        assert stored.names == {"frown": FRAGMENT}
        assert stored.characters == {EYES: STANDARD}

    def test_names_are_normalised_so_they_can_be_spoken(self, tmp_path):
        path = tmp_path / "emoji.json"
        key = custom_emoji.remember("name", "Jackl_Frown!", FRAGMENT, path)
        assert key == "jackl frown"
        assert custom_emoji.load(path).names[key] == FRAGMENT

    def test_a_second_capture_replaces_the_first(self, tmp_path):
        path = tmp_path / "emoji.json"
        custom_emoji.remember("name", "frown", "<old>", path)
        custom_emoji.remember("name", "frown", FRAGMENT, path)
        assert custom_emoji.load(path).names == {"frown": FRAGMENT}

    def test_a_flat_file_is_read_as_names(self, tmp_path):
        """The shape this store had before standard emoji were worth keeping."""
        path = tmp_path / "emoji.json"
        path.write_text('{"frown": "<x>"}', encoding="utf-8")
        stored = custom_emoji.load(path)
        assert stored.names == {"frown": "<x>"}
        assert stored.characters == {}

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert custom_emoji.load(tmp_path / "nope.json").names == {}

    def test_corrupt_json_costs_the_emoji_not_the_dictation(self, tmp_path):
        path = tmp_path / "emoji.json"
        path.write_text("{not json at all", encoding="utf-8")
        assert custom_emoji.load(path).names == {}

    def test_a_json_array_is_rejected(self, tmp_path):
        path = tmp_path / "emoji.json"
        path.write_text('["not", "an object"]', encoding="utf-8")
        assert custom_emoji.load(path).names == {}

    def test_unusable_entries_are_skipped(self, tmp_path):
        path = tmp_path / "emoji.json"
        path.write_text('{"good": "<x>", "empty": "", "wrong": 42}', encoding="utf-8")
        assert custom_emoji.load(path).names == {"good": "<x>"}


class TestNativeIds:
    """The shipped table. A bad id here silently costs a whole paste, so it is worth
    more scrutiny than a normal literal."""

    def test_every_id_has_one_of_the_two_shapes(self):
        """A bare short name is a reaction id; hex_name is an emoticon id. Both ship.

        Reaction assets are the animated emoji, which is the point of using them — but
        a reaction is keyed by meaning, not by picture, so the wrong ones are excluded
        by name rather than by shape. See test_known_mismatches_are_excluded.
        """
        import re

        bad = [
            (character, itemid)
            for character, itemid in custom_emoji.NATIVE_IDS.items()
            if not re.fullmatch(r"[a-z0-9]+|[0-9a-f]{4,6}_[a-z0-9]+", itemid)
        ]
        assert bad == []

    def test_known_mismatches_are_excluded(self):
        """Each of these was fetched and looked at, and draws something else.

        "like" is a face holding a thumb rather than a thumbs up; "hi" and "highfive"
        are a face performing the gesture; "laughdog" is a dog for a *face* codepoint,
        which is why no automatic rule can do this job.
        """
        listed = set(custom_emoji.NATIVE_IDS.values())
        for wrong in ("like", "hi", "highfive", "coolkoala", "cooldog", "laughdog"):
            assert wrong not in listed

    def test_verified_reaction_ids_are_kept(self):
        """The animated art that *is* the emoji, which is the reason for shipping these."""
        assert custom_emoji.NATIVE_IDS["❤"] == "heart"
        assert custom_emoji.NATIVE_IDS["😂"] == "cwl"
        assert custom_emoji.NATIVE_IDS["😈"] == "devil"

    def test_a_prefixed_id_still_matches_its_key(self):
        """Only checkable for hex_name ids; a bare name says nothing about its codepoint."""
        wrong = [
            (character, itemid)
            for character, itemid in custom_emoji.NATIVE_IDS.items()
            if "_" in itemid and int(itemid.split("_")[0], 16) != ord(character)
        ]
        assert wrong == []

    def test_keys_are_single_characters(self):
        assert all(len(character) == 1 for character in custom_emoji.NATIVE_IDS)

    def test_markup_is_none_for_anything_unlisted(self):
        """Never a guess: an unrecognised id makes Teams refuse the whole paste."""
        assert custom_emoji.native_markup("q") is None
        assert custom_emoji.native_markup("🧿") is None

    def test_markup_carries_the_id_and_the_codepoint(self):
        markup = custom_emoji.native_markup(EYES)
        assert 'itemid="1f440_eyes"' in markup
        assert 'itemscope="' + EYES + '"' in markup

    def test_a_label_cannot_break_out_of_its_attribute(self, monkeypatch):
        monkeypatch.setitem(custom_emoji.NATIVE_IDS, "😀", "1f600_x")
        monkeypatch.setattr(custom_emoji.unicodedata, "name", lambda c, d="": 'Bad" <x> &')
        markup = custom_emoji.native_markup("😀")
        assert '"' not in markup.split('title="')[1].split('"')[0] or "&quot;" in markup


class TestDecorate:
    """Standard emoji get Teams' asset as *markup*, never as a text change.

    The codepoint stays in the text, which is what should arrive in Slack or an
    email; only the rendering differs where HTML is accepted.
    """

    @staticmethod
    def _with(characters, monkeypatch):
        monkeypatch.setattr(
            custom_emoji, "_store", lambda: custom_emoji.Stored(characters=characters)
        )

    def test_decorates_a_stored_character(self, monkeypatch):
        self._with({EYES: STANDARD}, monkeypatch)
        assert custom_emoji.decorate("look " + EYES + " here") == [(5, 6, STANDARD)]

    def test_decorates_every_occurrence(self, monkeypatch):
        self._with({EYES: STANDARD}, monkeypatch)
        spans = custom_emoji.decorate(EYES + " and " + EYES)
        assert [(s, e) for s, e, _ in spans] == [(0, 1), (6, 7)]

    def test_a_shipped_native_is_decorated_without_capture(self, monkeypatch):
        """The point of hardcoding: no capture needed for the standard set."""
        self._with({}, monkeypatch)
        spans = custom_emoji.decorate("look " + EYES + " here")
        assert len(spans) == 1 and 'itemid="1f440_eyes"' in spans[0][2]

    def test_an_unlisted_character_is_left_alone(self, monkeypatch):
        self._with({}, monkeypatch)
        assert custom_emoji.decorate("just words") == []

    def test_a_captured_character_beats_the_shipped_table(self, monkeypatch):
        """An org that replaced a standard emoji gets its own."""
        self._with({EYES: "<mine/>"}, monkeypatch)
        assert custom_emoji.decorate(EYES) == [(0, 1, "<mine/>")]

    def test_the_text_is_never_consulted_for_a_change(self, monkeypatch):
        """decorate returns spans only; there is no path by which it edits text."""
        self._with({EYES: STANDARD}, monkeypatch)
        spans = custom_emoji.decorate(EYES)
        assert all(isinstance(span, tuple) and len(span) == 3 for span in spans)
