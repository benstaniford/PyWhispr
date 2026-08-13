"""Putting formatted content on the clipboard, and reading it back.

Only exists because some things cannot be said in plain text. A Teams custom
emoji is not a character — it has no Unicode codepoint — it is a tenant-hosted
image referenced by an ``<img itemid=...>`` inside a marker element. The only way
to hand one to another application is to offer HTML on the clipboard and let that
application render it.

**Qt cannot do this on Windows.** ``QMimeData.setHtml()`` produces a ``CF_HTML``
whose header is valid but whose *document* is not: ``StartHTML`` points straight
at ``<!--StartFragment-->`` with no ``<html>``/``<body>`` root. Teams (WebView2)
rejects that outright and silently takes the plain-text alternative instead, which
looks exactly like "Teams refuses pasted HTML" and is not. Give it the same
fragment inside a real document and it renders — custom emoji, bold, links, all of
it. Hence the hand-built header here.

Everything is best-effort and returns a bool rather than raising: the caller
always has plain text to fall back on, and a transcript that pastes as its emoji's
*name* is a much better outcome than one that does not paste at all.
"""

from __future__ import annotations

import html as html_module
import logging
import sys

log = logging.getLogger(__name__)

# Windows clipboard plumbing. The registered format name is stable across
# versions and is what browsers, Word and Teams all speak.
_CF_HTML_NAME = "HTML Format"
_CF_UNICODETEXT = 13
# ZEROINIT is not optional. GlobalAlloc with MOVEABLE alone leaves the block
# uninitialised, and the two bytes reserved past the payload for a terminator then
# hold whatever was in the heap — which the receiving application reads and appends
# as text. It showed up as an emoji followed by two random letters ("ht", "nl"):
# real, reproducible, and invisible to any test that does not read the raw bytes
# back off the clipboard.
_GMEM_MOVEABLE_ZEROED = 0x0002 | 0x0040

# Wrapping the fragment in a document is the whole point of this module — see the
# note above about what WebView2 does with a rootless one.
_PREFIX = "<html>\r\n<body>\r\n<!--StartFragment-->"
_SUFFIX = "<!--EndFragment-->\r\n</body>\r\n</html>"


def supported() -> bool:
    """Can this platform offer HTML at all? Callers degrade to text when not."""
    return sys.platform in ("win32", "darwin")


def build_cf_html(fragment: str) -> bytes:
    """A complete ``CF_HTML`` payload for `fragment`, offsets and all.

    The offsets are byte counts over the finished buffer, and the header's own
    length is part of every one of them — so the header is laid out once with
    ten-digit zeros to measure it, then again with the real numbers. Pure and
    exported so the arithmetic can be tested without touching a clipboard.
    """
    template = (
        "Version:0.9\r\n"
        "StartHTML:{start_html:010d}\r\n"
        "EndHTML:{end_html:010d}\r\n"
        "StartFragment:{start_fragment:010d}\r\n"
        "EndFragment:{end_fragment:010d}\r\n"
    )
    header_length = len(
        template.format(start_html=0, end_html=0, start_fragment=0, end_fragment=0)
    )
    start_html = header_length
    start_fragment = start_html + len(_PREFIX.encode("utf-8"))
    end_fragment = start_fragment + len(fragment.encode("utf-8"))
    end_html = end_fragment + len(_SUFFIX.encode("utf-8"))
    header = template.format(
        start_html=start_html,
        end_html=end_html,
        start_fragment=start_fragment,
        end_fragment=end_fragment,
    )
    return (header + _PREFIX + fragment + _SUFFIX).encode("utf-8")


def render(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Build an HTML fragment from plain `text` with `spans` replaced by markup.

    ``spans`` are ``(start, end, html)`` in `text`'s own coordinates. Everything
    outside them is **escaped**, because a transcript is whatever the user said and
    may contain ``<`` or ``&``; everything inside is inserted verbatim, because it
    is markup the plugin supplied. Line breaks become ``<br>`` so a dictated
    newline survives the round trip.

    Overlapping or out-of-order spans are dropped rather than trusted — the plain
    text stays correct either way, which is the property worth protecting.
    """

    def escape(chunk: str) -> str:
        return html_module.escape(chunk, quote=False).replace("\r\n", "<br>").replace("\n", "<br>")

    out: list[str] = []
    cursor = 0
    for start, end, markup in sorted(spans, key=lambda span: span[0]):
        if not 0 <= start <= end <= len(text) or start < cursor:
            log.debug("Ignoring an unusable rich span while rendering")
            continue
        out.append(escape(text[cursor:start]))
        out.append(markup)
        cursor = end
    out.append(escape(text[cursor:]))
    return "".join(out)


# -- Windows -----------------------------------------------------------------


def _set_windows(fragment: str, text: str) -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE

    def handle_for(data: bytes):
        # +2 for a terminator wide enough for either format, and zeroed so those
        # two bytes really are a terminator rather than leftover heap.
        handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE_ZEROED, len(data) + 2)
        if not handle:
            raise OSError("GlobalAlloc failed")
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise OSError("GlobalLock failed")
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        return handle

    html_format = user32.RegisterClipboardFormatW(_CF_HTML_NAME)
    if not html_format:
        return False
    if not user32.OpenClipboard(None):
        log.debug("Could not open the clipboard for a rich paste")
        return False
    try:
        user32.EmptyClipboard()
        # Both formats, deliberately: the target chooses, and anything that does
        # not read HTML must still get sensible words.
        # Checked, unlike the first version of this: SetClipboardData returning
        # NULL is how a rich paste fails silently and leaves the caller believing
        # it worked.
        wrote_html = user32.SetClipboardData(html_format, handle_for(build_cf_html(fragment)))
        user32.SetClipboardData(_CF_UNICODETEXT, handle_for(text.encode("utf-16-le")))
        if not wrote_html:
            log.debug("SetClipboardData refused the HTML format")
            return False
    finally:
        user32.CloseClipboard()
    return True


def _get_windows_html() -> str | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE

    html_format = user32.RegisterClipboardFormatW(_CF_HTML_NAME)
    if not html_format or not user32.OpenClipboard(None):
        return None
    try:
        handle = user32.GetClipboardData(html_format)
        if not handle:
            return None
        size = kernel32.GlobalSize(handle)
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None
        raw = ctypes.string_at(pointer, size)
        kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
    return raw.decode("utf-8", "replace")


# -- macOS -------------------------------------------------------------------


def _set_macos(fragment: str, text: str) -> bool:
    """NSPasteboard with both HTML and string types.

    pyobjc is already a transitive dependency of pynput, so this costs no new
    package. **Unverified on a real machine** — written from the documented API —
    so every failure path returns False and the caller pastes plain text.
    """
    from AppKit import NSPasteboard, NSPasteboardTypeHTML, NSPasteboardTypeString

    pasteboard = NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    # The full document, not the bare fragment: WebKit is at least as fussy as
    # WebView2 about a rootless one, and there is no reason to find out the hard
    # way on a platform this cannot be tested on.
    document = _PREFIX + fragment + _SUFFIX
    wrote_html = pasteboard.setString_forType_(document, NSPasteboardTypeHTML)
    pasteboard.setString_forType_(text, NSPasteboardTypeString)
    return bool(wrote_html)


def _get_macos_html() -> str | None:
    from AppKit import NSPasteboard, NSPasteboardTypeHTML

    value = NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeHTML)
    return str(value) if value else None


# -- the public pair ---------------------------------------------------------


def set_rich(fragment: str, text: str) -> bool:
    """Offer `fragment` as HTML and `text` as plain text. True if HTML got there.

    False is a normal answer, not an error: an unsupported platform, a clipboard
    another process is holding, a pyobjc that will not import. The caller pastes
    plain text and the user sees the words rather than the formatting.
    """
    if not fragment:
        return False
    try:
        if sys.platform == "win32":
            return _set_windows(fragment, text)
        if sys.platform == "darwin":
            return _set_macos(fragment, text)
    except Exception:
        log.exception("Could not put formatted content on the clipboard")
        return False
    log.debug("No rich clipboard support on %s", sys.platform)
    return False


def get_html() -> str | None:
    """Whatever HTML is on the clipboard, or None.

    Here because it is the read half of the same platform mess, and because a
    plugin that wants to capture formatted content has no other way to get at it.
    """
    try:
        if sys.platform == "win32":
            return _get_windows_html()
        if sys.platform == "darwin":
            return _get_macos_html()
    except Exception:
        log.exception("Could not read formatted content from the clipboard")
    return None
