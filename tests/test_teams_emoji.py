"""Capturing Teams custom emoji off the clipboard, and storing them.

Never touches the developer's real store: every test passes an explicit tmp_path.
"""

from __future__ import annotations

from pywhispr.plugins.builtin import _teams_emoji as custom_emoji

FRAGMENT = (
    '<readonly aria-label="jackl_frown" contenteditable="false"'
    ' itemtype="http://schema.skype.com/CustomEmoji">'
    '<img alt="jackl_frown" itemid="jackl_frown;0-eus-d20-abc" src="https://x/y">'
    "</readonly>"
)
CLIPBOARD = f"<html>\r\n<body>\r\n<!--StartFragment-->{FRAGMENT}<!--EndFragment-->\r\n</body>\r\n</html>"


class TestExtract:
    def test_finds_the_fragment_and_its_label(self):
        label, fragment = custom_emoji.extract(CLIPBOARD)
        assert label == "jackl_frown"
        assert fragment == FRAGMENT

    def test_finds_it_among_other_content(self):
        wrapped = f"<html><body>hello <b>there</b> {FRAGMENT} goodbye</body></html>"
        assert custom_emoji.extract(wrapped)[1] == FRAGMENT

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
        label, fragment = custom_emoji.extract(bare)
        assert label == ""
        assert fragment == bare


class TestStore:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "custom_emoji.json"
        custom_emoji.remember("jackl frown", FRAGMENT, path)
        assert custom_emoji.load(path) == {"jackl frown": FRAGMENT}

    def test_keys_are_normalised_so_they_can_be_spoken(self, tmp_path):
        path = tmp_path / "custom_emoji.json"
        key = custom_emoji.remember("Jackl_Frown!", FRAGMENT, path)
        assert key == "jackl frown"
        assert custom_emoji.load(path)[key] == FRAGMENT

    def test_a_second_capture_replaces_the_first(self, tmp_path):
        path = tmp_path / "custom_emoji.json"
        custom_emoji.remember("frown", "<old>", path)
        custom_emoji.remember("frown", FRAGMENT, path)
        assert custom_emoji.load(path) == {"frown": FRAGMENT}

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert custom_emoji.load(tmp_path / "nope.json") == {}

    def test_corrupt_json_costs_the_emoji_not_the_dictation(self, tmp_path):
        path = tmp_path / "custom_emoji.json"
        path.write_text("{not json at all", encoding="utf-8")
        assert custom_emoji.load(path) == {}

    def test_a_json_array_is_rejected(self, tmp_path):
        path = tmp_path / "custom_emoji.json"
        path.write_text('["not", "an object"]', encoding="utf-8")
        assert custom_emoji.load(path) == {}

    def test_unusable_entries_are_skipped(self, tmp_path):
        path = tmp_path / "custom_emoji.json"
        path.write_text('{"good": "<x>", "empty": "", "wrong": 42}', encoding="utf-8")
        assert custom_emoji.load(path) == {"good": "<x>"}
