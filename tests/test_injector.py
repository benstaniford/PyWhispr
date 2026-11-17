from unittest.mock import MagicMock, patch

from pywhispr.injector import TextInjector


def _mock_clipboard(has_text: bool, text: str = ""):
    clipboard = MagicMock()
    clipboard.mimeData.return_value.hasText.return_value = has_text
    clipboard.text.return_value = text
    return clipboard


def _injector(clipboard, can_paste=True):
    injector = TextInjector(paste_delay_ms=1, restore_delay_ms=1)
    patches = [
        patch.object(injector, "_clipboard", return_value=clipboard),
        patch.object(injector, "can_auto_paste", return_value=can_paste),
    ]
    return injector, patches


def test_sequence_sets_pastes_and_restores(qtbot):
    clipboard = _mock_clipboard(has_text=True, text="previous contents")
    injector, patches = _injector(clipboard)
    calls = []
    clipboard.setText.side_effect = lambda t: calls.append(("set", t))
    patches.append(
        patch.object(
            injector, "_send_paste_keystroke", side_effect=lambda: calls.append(("paste",))
        )
    )

    with patches[0], patches[1], patches[2]:
        with qtbot.waitSignal(injector.finished, timeout=2000) as blocker:
            injector.insert("hello from pywhispr")

    assert blocker.args == [True]
    assert calls == [
        ("set", "hello from pywhispr"),
        ("paste",),
        ("set", "previous contents"),
    ]


def test_non_text_clipboard_is_not_restored(qtbot):
    clipboard = _mock_clipboard(has_text=False)
    injector, patches = _injector(clipboard)

    with patches[0], patches[1], patch.object(injector, "_send_paste_keystroke"):
        with qtbot.waitSignal(injector.finished, timeout=2000):
            injector.insert("hello")

    clipboard.setText.assert_called_once_with("hello")  # no restore call


def test_empty_mimedata_is_handled(qtbot):
    clipboard = _mock_clipboard(has_text=True, text="keep me")
    clipboard.mimeData.return_value = None
    injector, patches = _injector(clipboard)

    with patches[0], patches[1], patch.object(injector, "_send_paste_keystroke"):
        with qtbot.waitSignal(injector.finished, timeout=2000):
            injector.insert("hello")

    clipboard.setText.assert_called_once_with("hello")


def test_clipboard_only_mode_when_accessibility_missing(qtbot):
    clipboard = _mock_clipboard(has_text=True, text="previous contents")
    injector, patches = _injector(clipboard, can_paste=False)

    with (
        patches[0],
        patches[1],
        patch.object(injector, "_send_paste_keystroke") as keystroke,
    ):
        with qtbot.waitSignal(injector.finished, timeout=2000) as blocker:
            injector.insert("hello")

    assert blocker.args == [False]
    keystroke.assert_not_called()
    # Transcript stays on the clipboard: exactly one setText, no restore.
    clipboard.setText.assert_called_once_with("hello")


class _FakeClipboard:
    """Holds a real QMimeData, like the system clipboard, without touching it."""

    def __init__(self, text: str | None = None):
        from PySide6.QtCore import QMimeData

        self._mime = QMimeData()
        if text is not None:
            self._mime.setText(text)

    def mimeData(self):
        return self._mime

    def setMimeData(self, mime):
        self._mime = mime

    def setText(self, text):
        from PySide6.QtCore import QMimeData

        self._mime = QMimeData()
        self._mime.setText(text)

    def text(self):
        return self._mime.text()

    def clear(self):
        from PySide6.QtCore import QMimeData

        self._mime = QMimeData()


def _holding(clipboard, can_paste=False):
    injector = TextInjector(paste_delay_ms=1, restore_delay_ms=1)
    injector._clipboard = lambda: clipboard
    injector.can_auto_paste = lambda: can_paste
    return injector


class TestHoldClipboard:
    """Clipboard-only mode: an early Cmd+V must paste nothing, not the old clipboard."""

    def test_recording_start_empties_the_clipboard(self, qapp):
        clipboard = _FakeClipboard("old secret")
        _holding(clipboard).hold_clipboard()
        assert clipboard.mimeData().formats() == []

    def test_no_transcript_gives_the_old_clipboard_back(self, qapp):
        clipboard = _FakeClipboard("old")
        injector = _holding(clipboard)
        injector.hold_clipboard()
        injector.release_clipboard()
        assert clipboard.text() == "old"

    def test_every_format_comes_back_not_just_text(self, qapp):
        clipboard = _FakeClipboard("old")
        clipboard.mimeData().setData("image/png", b"\x89PNG")
        injector = _holding(clipboard)
        injector.hold_clipboard()
        injector.release_clipboard()
        assert bytes(clipboard.mimeData().data("image/png")) == b"\x89PNG"
        assert clipboard.text() == "old"

    def test_a_transcript_replaces_the_held_clipboard(self, qtbot):
        clipboard = _FakeClipboard("old")
        injector = _holding(clipboard)
        injector.hold_clipboard()
        with qtbot.waitSignal(injector.finished):
            injector.insert("hello")
        injector.release_clipboard()  # what the end of the cycle does
        assert clipboard.text() == "hello"

    def test_something_copied_meanwhile_is_not_overwritten(self, qapp):
        clipboard = _FakeClipboard("old")
        injector = _holding(clipboard)
        injector.hold_clipboard()
        clipboard.setText("copied while dictating")
        injector.release_clipboard()
        assert clipboard.text() == "copied while dictating"

    def test_auto_paste_never_touches_the_clipboard_at_recording_start(self, qapp):
        clipboard = _FakeClipboard("old")
        _holding(clipboard, can_paste=True).hold_clipboard()
        assert clipboard.text() == "old"

    def test_accessibility_granted_mid_dictation_still_restores(self, qtbot):
        clipboard = _FakeClipboard("old")
        injector = _holding(clipboard)
        injector.hold_clipboard()
        injector.can_auto_paste = lambda: True
        injector._send_paste_keystroke = lambda: None
        with qtbot.waitSignal(injector.finished):
            injector.insert("hello")
        assert clipboard.text() == "old"

    def test_an_empty_clipboard_stays_empty(self, qapp):
        clipboard = _FakeClipboard()
        injector = _holding(clipboard)
        injector.hold_clipboard()
        injector.release_clipboard()
        assert clipboard.mimeData().formats() == []
