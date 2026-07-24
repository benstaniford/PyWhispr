from unittest.mock import MagicMock, patch

from pywhispr.injector import TextInjector


def _mock_clipboard(has_text: bool, text: str = ""):
    clipboard = MagicMock()
    clipboard.mimeData.return_value.hasText.return_value = has_text
    clipboard.text.return_value = text
    return clipboard


def test_sequence_sets_pastes_and_restores(qtbot):
    injector = TextInjector(paste_delay_ms=1, restore_delay_ms=1)
    clipboard = _mock_clipboard(has_text=True, text="previous contents")
    calls = []
    clipboard.setText.side_effect = lambda t: calls.append(("set", t))

    with (
        patch.object(injector, "_clipboard", return_value=clipboard),
        patch.object(
            injector, "_send_paste_keystroke", side_effect=lambda: calls.append(("paste",))
        ),
    ):
        with qtbot.waitSignal(injector.finished, timeout=2000):
            injector.insert("hello from pywhispr")

    assert calls == [
        ("set", "hello from pywhispr"),
        ("paste",),
        ("set", "previous contents"),
    ]


def test_non_text_clipboard_is_not_restored(qtbot):
    injector = TextInjector(paste_delay_ms=1, restore_delay_ms=1)
    clipboard = _mock_clipboard(has_text=False)

    with (
        patch.object(injector, "_clipboard", return_value=clipboard),
        patch.object(injector, "_send_paste_keystroke"),
    ):
        with qtbot.waitSignal(injector.finished, timeout=2000):
            injector.insert("hello")

    clipboard.setText.assert_called_once_with("hello")  # no restore call


def test_empty_mimedata_is_handled(qtbot):
    injector = TextInjector(paste_delay_ms=1, restore_delay_ms=1)
    clipboard = _mock_clipboard(has_text=True, text="keep me")
    clipboard.mimeData.return_value = None

    with (
        patch.object(injector, "_clipboard", return_value=clipboard),
        patch.object(injector, "_send_paste_keystroke"),
    ):
        with qtbot.waitSignal(injector.finished, timeout=2000):
            injector.insert("hello")

    clipboard.setText.assert_called_once_with("hello")
