from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLabel, QToolButton

from pywhispr.ui.history_dialog import COPIED_FEEDBACK_MS, MAX_VISIBLE_ROWS, HistoryDialog


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


def test_copy_button_stays_visible_at_minimum_width(qtbot):
    """A long preview must give way to the button, not push it off the row."""
    dialog = HistoryDialog(["a transcript far too long to fit on one row " * 3])
    qtbot.addWidget(dialog)
    dialog.resize(dialog.minimumWidth(), dialog.height())
    dialog.show()
    qtbot.waitExposed(dialog)

    button = copy_button(dialog, 0)
    assert button.width() > 0
    # Against the viewport, not the row: the row itself used to be laid out at
    # the width of its whole preview, which put the button off the dialog.
    assert button.geometry().right() <= dialog._list.viewport().width()


def test_short_history_makes_a_short_dialog(qtbot):
    one = HistoryDialog(["only one"])
    qtbot.addWidget(one)
    several = HistoryDialog([f"line {i}" for i in range(MAX_VISIBLE_ROWS)])
    qtbot.addWidget(several)

    # Sized to the content, so one entry does not sit above a column of nothing.
    assert one._list.height() < several._list.height()


def test_more_than_the_visible_rows_scrolls_instead_of_growing(qtbot):
    full = HistoryDialog([f"line {i}" for i in range(MAX_VISIBLE_ROWS)])
    qtbot.addWidget(full)
    overflowing = HistoryDialog([f"line {i}" for i in range(MAX_VISIBLE_ROWS + 4)])
    qtbot.addWidget(overflowing)

    assert overflowing._list.count() == MAX_VISIBLE_ROWS + 4
    assert overflowing._list.height() == full._list.height()

    overflowing.show()  # the scroll range is only real once it is laid out
    qtbot.waitExposed(overflowing)
    assert overflowing._list.verticalScrollBar().maximum() > 0


def test_closing_before_the_copy_feedback_expires_is_survivable(qtbot):
    """The feedback timer must not outlive the button it resets."""
    dialog = HistoryDialog(["newest"])
    copy_button(dialog, 0).click()

    dialog.deleteLater()
    qtbot.wait(COPIED_FEEDBACK_MS + 200)  # would segfault on a deleted widget


def test_the_only_close_control_is_the_window_frame(qtbot):
    """A ✕ of our own sat directly under the window's own one — two close buttons."""
    dialog = HistoryDialog(["newest", "older"])
    qtbot.addWidget(dialog)

    assert not [b for b in dialog.findChildren(QToolButton) if b.toolTip() == "Close"]


def test_escape_rejects(qtbot):
    dialog = HistoryDialog(["newest"])
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.keyClick(dialog, Qt.Key.Key_Escape)

    assert dialog.result() == QDialog.DialogCode.Rejected
