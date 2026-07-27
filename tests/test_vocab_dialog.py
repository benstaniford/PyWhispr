from pywhispr.ui.vocab_dialog import VocabularyDialog


def test_round_trips_the_text_it_was_given(qtbot):
    dialog = VocabularyDialog("BeyondTrust\n# a note\n")
    qtbot.addWidget(dialog)
    assert dialog.text() == "BeyondTrust\n# a note\n"


def test_status_counts_terms_live(qtbot):
    dialog = VocabularyDialog("")
    qtbot.addWidget(dialog)
    assert dialog._status.text() == "0 terms"

    dialog._editor.setPlainText("BeyondTrust")
    assert dialog._status.text() == "1 term"

    dialog._editor.setPlainText("BeyondTrust\nKubernetes\n# comment\n")
    assert dialog._status.text() == "2 terms"


def test_status_flags_lines_that_cannot_match(qtbot):
    dialog = VocabularyDialog("BeyondTrust\n...\n")
    qtbot.addWidget(dialog)
    assert dialog._status.text().startswith("1 term · 1 line ignored")
