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
