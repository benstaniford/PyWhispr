from pywhispr.ui.history_dialog import HistoryDialog


def test_lists_previews_newest_first(qtbot):
    dialog = HistoryDialog(["newest one", "older   one\nwith a break"])
    qtbot.addWidget(dialog)
    assert dialog._list.count() == 2
    assert dialog._list.item(0).text() == "newest one"
    assert dialog._list.item(1).text() == "older one with a break"


def test_chosen_is_the_whole_transcript_not_the_preview(qtbot):
    long_text = "a sentence that is very much longer than one list row can show " * 3
    dialog = HistoryDialog([long_text])
    qtbot.addWidget(dialog)
    assert dialog._list.item(0).text() != long_text  # clipped for display
    assert dialog.chosen() == long_text  # but the full text is what gets pasted


def test_the_newest_is_preselected(qtbot):
    dialog = HistoryDialog(["newest", "older"])
    qtbot.addWidget(dialog)
    assert dialog.chosen() == "newest"
