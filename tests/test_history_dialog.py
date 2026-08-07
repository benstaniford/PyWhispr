from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLabel, QToolButton

from pywhispr.ui.history_dialog import HistoryDialog


def row_label(dialog: HistoryDialog, index: int) -> str:
    widget = dialog._list.itemWidget(dialog._list.item(index))
    return widget.findChild(QLabel).text()


def copy_button(dialog: HistoryDialog, index: int) -> QToolButton:
    widget = dialog._list.itemWidget(dialog._list.item(index))
    return widget.findChild(QToolButton)


def test_lists_previews_newest_first(qtbot):
    dialog = HistoryDialog(["newest one", "older   one\nwith a break"])
    qtbot.addWidget(dialog)
    assert dialog._list.count() == 2
    assert row_label(dialog, 0) == "newest one"
    assert row_label(dialog, 1) == "older one with a break"


def test_chosen_is_the_whole_transcript_not_the_preview(qtbot):
    long_text = "a sentence that is very much longer than one list row can show " * 3
    dialog = HistoryDialog([long_text])
    qtbot.addWidget(dialog)
    assert row_label(dialog, 0) != long_text  # clipped for display
    assert dialog.chosen() == long_text  # but the full text is what gets pasted


def test_the_newest_is_preselected(qtbot):
    dialog = HistoryDialog(["newest", "older"])
    qtbot.addWidget(dialog)
    assert dialog.chosen() == "newest"


def test_each_row_has_its_own_copy_button(qtbot):
    dialog = HistoryDialog(["newest", "older"])
    qtbot.addWidget(dialog)
    assert copy_button(dialog, 0) is not None
    assert copy_button(dialog, 1) is not None


def test_row_copy_button_copies_that_whole_transcript(qtbot):
    long_text = "a sentence that is very much longer than one list row can show " * 3
    dialog = HistoryDialog(["newest", long_text])
    qtbot.addWidget(dialog)

    copy_button(dialog, 1).click()

    # The full text, not the clipped preview, and not the selected row.
    assert QGuiApplication.clipboard().text() == long_text
    assert dialog.chosen() == "newest"


def test_copying_does_not_close_the_dialog(qtbot):
    dialog = HistoryDialog(["newest"])
    qtbot.addWidget(dialog)
    dialog.show()

    copy_button(dialog, 0).click()

    assert dialog.isVisible()


def test_close_button_rejects(qtbot):
    dialog = HistoryDialog(["newest"])
    qtbot.addWidget(dialog)
    close = next(b for b in dialog.findChildren(QToolButton) if b.toolTip() == "Close")

    close.click()

    assert dialog.result() == QDialog.DialogCode.Rejected
