"""Dialog for editing the custom-vocabulary list.

A plain text box rather than a table of rows: the list is a list, people paste
into it from elsewhere, and the file's own comments explain the format better
than any label could.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from pywhispr.vocab import count_entries


class VocabularyDialog(QDialog):
    """Modal editor over the raw vocabulary text."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vocabulary")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumSize(460, 380)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "One word or phrase per line, spelled the way you want it written.\n"
                "Use  heard => wanted  to fix a mishearing the model makes every time."
            )
        )

        self._editor = QPlainTextEdit()
        self._editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._editor.setPlainText(text)
        self._editor.setTabChangesFocus(True)
        self._editor.textChanged.connect(self._update_status)
        layout.addWidget(self._editor)

        self._status = QLabel()
        self._status.setStyleSheet("color: gray;")
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_status()

    def _update_status(self) -> None:
        terms, ignored = count_entries(self._editor.toPlainText())
        status = f"{terms} term{'' if terms == 1 else 's'}"
        if ignored:
            status += f" · {ignored} line{'' if ignored == 1 else 's'} ignored (nothing to match on)"
        self._status.setText(status)

    def text(self) -> str:
        return self._editor.toPlainText()

    @staticmethod
    def edit(text: str, parent=None) -> str | None:
        """Show the dialog; return the edited text, or None if cancelled."""
        dialog = VocabularyDialog(text, parent)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.text()
        return None
