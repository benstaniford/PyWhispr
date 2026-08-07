"""Picker for the last few transcripts, to paste one where it should have gone.

A list of one-line previews rather than the full text: this is chosen by
someone who dictated the words seconds ago and only needs to tell them apart.
The whole transcript is on the row's tooltip for the case where they can't.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from pywhispr.history import preview
from pywhispr.ui.foreground import centre_on_active_screen


class HistoryDialog(QDialog):
    """Modal list of recent transcripts, newest first."""

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recent dictations")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Pick one to paste where the caret is now:"))

        self._list = QListWidget()
        for text in items:
            item = QListWidgetItem(preview(text))
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setToolTip(text)
            self._list.addItem(item)
        self._list.setCurrentRow(0)
        # Double-click, or Return with the list focused, is the whole interaction.
        self._list.itemActivated.connect(lambda _item: self.accept())
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Paste")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._list.setFocus()

    def chosen(self) -> str | None:
        item = self._list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    @staticmethod
    def choose(items: list[str], parent=None) -> str | None:
        """Show the picker; return the chosen transcript, or None if cancelled."""
        dialog = HistoryDialog(items, parent)
        centre_on_active_screen(dialog)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.chosen()
        return None
