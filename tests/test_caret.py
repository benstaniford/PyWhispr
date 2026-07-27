import sys
from unittest.mock import MagicMock, patch

import pytest

from pywhispr.caret import AX_TIMEOUT_SECONDS, ContextTracker, frontmost_app_id, read_preceding_text


class TestContextTracker:
    """The tracker is pure bookkeeping around two best-effort lookups, so both
    are patched out here."""

    def _tracker(self, **kwargs):
        return ContextTracker(max_chars=8, memory_seconds=60, **kwargs)

    def test_no_context_at_all(self):
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=None),
            patch("pywhispr.caret.frontmost_app_id", return_value="com.apple.TextEdit"),
        ):
            assert self._tracker().preceding_text() is None

    def test_live_read_wins_over_memory(self):
        with (
            patch("pywhispr.caret.read_preceding_text", return_value="live text"),
            patch("pywhispr.caret.frontmost_app_id", return_value="com.apple.TextEdit"),
        ):
            tracker = self._tracker()
            tracker.remember("remembered")
            assert tracker.preceding_text() == "live text"

    def test_empty_live_read_is_not_treated_as_missing(self):
        """"" means the caret is at the start of the field — real information,
        and it must not fall through to the memory."""
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=""),
            patch("pywhispr.caret.frontmost_app_id", return_value="com.apple.TextEdit"),
        ):
            tracker = self._tracker()
            tracker.remember("remembered")
            assert tracker.preceding_text() == ""

    def test_memory_used_when_the_app_will_not_say(self):
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=None),
            patch("pywhispr.caret.frontmost_app_id", return_value="com.apple.Terminal"),
        ):
            tracker = self._tracker()
            tracker.remember("the shop")
            assert tracker.preceding_text() == "the shop"

    def test_memory_truncated_to_max_chars(self):
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=None),
            patch("pywhispr.caret.frontmost_app_id", return_value="app"),
        ):
            tracker = self._tracker()
            tracker.remember("far more than eight characters")
            assert tracker.preceding_text() == "aracters"

    def test_focus_change_drops_the_memory(self):
        with patch("pywhispr.caret.read_preceding_text", return_value=None):
            with patch("pywhispr.caret.frontmost_app_id", return_value="com.apple.Terminal"):
                tracker = self._tracker()
                tracker.remember("the shop")
            with patch("pywhispr.caret.frontmost_app_id", return_value="com.apple.Mail"):
                assert tracker.preceding_text() is None
                # Cleared, not merely rejected for this one call.
                assert tracker._text is None

    def test_unknown_app_id_does_not_disable_the_fallback(self):
        """On Linux there is no app id at all; comparing against None would kill
        the fallback everywhere."""
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=None),
            patch("pywhispr.caret.frontmost_app_id", return_value=None),
        ):
            tracker = self._tracker()
            tracker.remember("the shop")
            assert tracker.preceding_text() == "the shop"

    def test_expired_memory_is_dropped(self):
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=None),
            patch("pywhispr.caret.frontmost_app_id", return_value="app"),
        ):
            tracker = ContextTracker(max_chars=8, memory_seconds=0)
            tracker.remember("the shop")
            with patch("pywhispr.caret.time.monotonic", return_value=1e9):
                assert tracker.preceding_text() is None
            assert tracker._text is None

    def test_invalidate(self):
        with (
            patch("pywhispr.caret.read_preceding_text", return_value=None),
            patch("pywhispr.caret.frontmost_app_id", return_value="app"),
        ):
            tracker = self._tracker()
            tracker.remember("the shop")
            tracker.invalidate()
            assert tracker.preceding_text() is None


def test_returns_none_off_darwin():
    with patch("pywhispr.caret.sys.platform", "win32"):
        assert read_preceding_text() is None


def test_returns_none_without_accessibility():
    with (
        patch("pywhispr.caret.sys.platform", "darwin"),
        patch("pywhispr.caret.check_macos_accessibility", return_value=False),
    ):
        assert read_preceding_text() is None


class FakeCFRange:
    def __init__(self, location, length):
        self.location = location
        self.length = length


class FakeCoreFoundation:
    """CoreFoundation is faked alongside HIServices deliberately.

    patch.dict(sys.modules) restores by wiping the dict and putting the original
    contents back, so anything imported for real inside the patch is deleted on
    exit — importing the real CoreFoundation here would tear pyobjc out of
    sys.modules and break every later test. Faking both means no real import
    happens, which also lets these tests run off macOS.
    """

    CFRange = FakeCFRange


class FakeAX:
    """Just enough of HIServices to drive _read_preceding_macos."""

    kAXErrorSuccess = 0
    kAXFocusedUIElementAttribute = "AXFocusedUIElement"
    kAXSelectedTextRangeAttribute = "AXSelectedTextRange"
    kAXStringForRangeParameterizedAttribute = "AXStringForRange"
    kAXValueCFRangeType = 4

    def __init__(self, *, caret=(20, 0), text="I went to the shop", errors=()):
        self.caret = caret
        self.text = text
        self.errors = set(errors)
        self.timeout = None
        self.requested_range = None

    def _err(self, name):
        return -25205 if name in self.errors else self.kAXErrorSuccess

    def AXUIElementCreateSystemWide(self):
        return "system"

    def AXUIElementSetMessagingTimeout(self, element, seconds):
        self.timeout = seconds

    def AXUIElementCopyAttributeValue(self, element, attribute, _none):
        err = self._err(attribute)
        return (err, None) if err else (err, f"value:{attribute}")

    def AXValueGetValue(self, value, kind, _none):
        if "range" in self.errors:
            return (False, None)
        return (True, self.caret)

    def AXValueCreate(self, kind, cfrange):
        self.requested_range = (cfrange.location, cfrange.length)
        return "range-value"

    def AXUIElementCopyParameterizedAttributeValue(self, element, attribute, value, _none):
        err = self._err(attribute)
        return (err, None) if err else (err, self.text)


@pytest.fixture
def ax():
    fake = FakeAX()
    with (
        patch("pywhispr.caret.sys.platform", "darwin"),
        patch("pywhispr.caret.check_macos_accessibility", return_value=True),
        patch.dict(sys.modules, {"HIServices": fake, "CoreFoundation": FakeCoreFoundation()}),
    ):
        yield fake


def test_reads_the_characters_before_the_caret(ax):
    assert read_preceding_text(max_chars=8) == "I went to the shop"
    # Caret at 20, so the eight characters ending there.
    assert ax.requested_range == (12, 8)
    assert ax.timeout == AX_TIMEOUT_SECONDS


def test_clamps_to_the_start_of_the_field(ax):
    ax.caret = (5, 0)
    read_preceding_text(max_chars=64)
    assert ax.requested_range == (0, 5)


def test_selection_length_is_ignored(ax):
    """The paste replaces the selection, so the join happens at its start."""
    ax.caret = (20, 12)
    read_preceding_text(max_chars=8)
    assert ax.requested_range == (12, 8)


def test_caret_at_zero_is_an_empty_context(ax):
    ax.caret = (0, 0)
    assert read_preceding_text() == ""
    assert ax.requested_range is None  # no point asking for nothing


@pytest.mark.parametrize(
    "failing",
    ["AXFocusedUIElement", "AXSelectedTextRange", "AXStringForRange", "range"],
)
def test_every_failure_yields_none(ax, failing):
    ax.errors = {failing}
    assert read_preceding_text() is None


def test_exceptions_are_swallowed(ax):
    ax.AXUIElementCreateSystemWide = MagicMock(side_effect=RuntimeError("objc blew up"))
    assert read_preceding_text() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS accessibility path")
def test_real_read_never_raises():
    """Untrusted from a test runner, so this returns None — the point is that it
    does so quietly rather than throwing."""
    result = read_preceding_text()
    assert result is None or isinstance(result, str)


def test_frontmost_app_id_never_raises():
    result = frontmost_app_id()
    assert result is None or isinstance(result, str)
