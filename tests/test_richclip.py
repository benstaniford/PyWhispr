"""CF_HTML construction and the HTML rendering of a plain transcript.

Both are pure functions, so the arithmetic that WebView2 is fussy about can be
checked without a clipboard, a display or a platform.
"""

from __future__ import annotations

import sys

import pytest

from pywhispr import richclip


def offsets(payload: bytes) -> dict[str, int]:
    header = payload.split(b"\r\n<html>", 1)[0]
    return {
        key.decode(): int(value)
        for key, _, value in (line.partition(b":") for line in header.split(b"\r\n"))
        if value.isdigit()
    }


class TestBuildCfHtml:
    """The header Teams needs. Every one of these was a real failure first.

    Qt's setHtml() writes a header-valid but document-invalid payload — StartHTML
    pointing straight at <!--StartFragment--> with no root element — and WebView2
    silently ignores the whole thing and takes the plain text instead.
    """

    def test_has_a_document_root(self):
        payload = richclip.build_cf_html("<b>hi</b>")
        marks = offsets(payload)
        assert payload[marks["StartHTML"] :].startswith(b"<html>")

    def test_start_fragment_points_at_the_fragment(self):
        payload = richclip.build_cf_html("<b>hi</b>")
        marks = offsets(payload)
        assert payload[marks["StartFragment"] : marks["EndFragment"]] == b"<b>hi</b>"

    def test_end_html_is_the_buffer_length(self):
        payload = richclip.build_cf_html("<b>hi</b>")
        assert offsets(payload)["EndHTML"] == len(payload)

    def test_offsets_are_byte_counts_not_character_counts(self):
        """A multi-byte fragment is where a character-based offset would drift."""
        payload = richclip.build_cf_html("<b>\U0001f600 café</b>")
        marks = offsets(payload)
        fragment = payload[marks["StartFragment"] : marks["EndFragment"]]
        assert fragment.decode("utf-8") == "<b>\U0001f600 café</b>"
        assert marks["EndHTML"] == len(payload)

    def test_the_fragment_markers_are_present(self):
        payload = richclip.build_cf_html("x")
        assert b"<!--StartFragment-->" in payload
        assert b"<!--EndFragment-->" in payload


class TestRender:
    def test_escapes_the_transcript_but_not_the_markup(self):
        """The transcript is whatever the user said and may contain < or &."""
        got = richclip.render("a < b & c NAME end", [(10, 14, '<img src="x">')])
        assert got == 'a &lt; b &amp; c <img src="x"> end'

    def test_no_spans_is_just_escaped_text(self):
        assert richclip.render("1 < 2", []) == "1 &lt; 2"

    def test_newlines_become_breaks(self):
        assert richclip.render("one\ntwo", []) == "one<br>two"

    def test_several_spans_all_apply(self):
        got = richclip.render("A B", [(0, 1, "<i>A</i>"), (2, 3, "<i>B</i>")])
        assert got == "<i>A</i> <i>B</i>"

    @pytest.mark.parametrize(
        "spans",
        [
            [(0, 5, "<b>x</b>"), (2, 4, "<i>y</i>")],  # overlapping
            [(99, 100, "<b>x</b>")],  # past the end
            [(-1, 2, "<b>x</b>")],  # before the start
            [(3, 1, "<b>x</b>")],  # backwards
        ],
    )
    def test_unusable_spans_are_dropped_and_the_text_survives(self, spans):
        """Markup is a luxury; the words are not. A bad span must not eat text."""
        got = richclip.render("hello world", spans)
        assert "hello" in got or "world" in got
        assert "&lt;b&gt;" not in got  # markup was never escaped into visible text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows clipboard")
class TestTheClipboardRoundTrip:
    """Reads the raw bytes back, because nothing else catches what went wrong here.

    The first version allocated the clipboard block with ``GMEM_MOVEABLE`` alone,
    which does not zero it — so the two bytes reserved past the payload for a
    terminator held leftover heap, and Teams appended them as text. It pasted the
    emoji followed by two random letters. Every higher-level view of the payload
    looked perfect, including ``build_cf_html`` itself, which is why this test works
    at the level of the actual clipboard.
    """

    @staticmethod
    def _set_or_skip(fragment: str, text: str) -> None:
        """Skip rather than fail when the clipboard is unavailable.

        ``OpenClipboard`` is a single-holder lock, and on a managed machine it can be
        denied outright by a clipboard hook — observed for a whole session here while
        Qt's OLE path kept working. A test that fails then is reporting the
        environment, not the code.
        """
        if not richclip.set_rich(fragment, text):
            pytest.skip("the clipboard is not available in this environment")

    def test_nothing_follows_the_document(self):
        self._set_or_skip("<b>x</b>", "x")
        raw = richclip.get_html()
        assert raw is not None
        # Whatever trails the document must be terminator, never stray characters.
        assert raw.rstrip("\x00").endswith("</html>")

    def test_the_fragment_survives_intact(self):
        fragment = '<img alt="thing" itemid="a;b" src="https://example.invalid/x">'
        self._set_or_skip(fragment, "thing")
        assert fragment in (richclip.get_html() or "")

    def test_multibyte_content_is_not_truncated(self):
        """Byte offsets and character counts diverge exactly here."""
        fragment = "<b>\U0001f600 café</b>"
        self._set_or_skip(fragment, "emoji cafe")
        assert fragment in (richclip.get_html() or "")


class TestSetRich:
    def test_an_empty_fragment_is_refused(self):
        assert richclip.set_rich("", "text") is False

    def test_supported_matches_the_platforms_with_an_implementation(self):
        assert richclip.supported() == (sys.platform in ("win32", "darwin"))
