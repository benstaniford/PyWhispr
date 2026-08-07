"""Picker for the last few transcripts, to paste one where it should have gone.

A list of one-line previews rather than the full text: this is chosen by
someone who dictated the words seconds ago and only needs to tell them apart.
The whole transcript is on the row's tooltip for the case where they can't.

Each row carries its own copy button, so a transcript can be taken to the
clipboard without pasting it anywhere — the dialog closes either way, and the
paste needs the caret that the picker itself stole the focus from.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pywhispr.history import preview
from pywhispr.ui.foreground import centre_on_active_screen

# Shown on the copy button after a copy, so the click has an answer. Long enough
# to read, short enough that the button is itself again before the next one.
COPIED_FEEDBACK_MS = 1200


def _icon(theme_name: str, fallback: str) -> tuple[QIcon | None, str]:
    """A themed icon where the platform has one, else a text glyph."""
    icon = QIcon.fromTheme(theme_name)
    return (None, fallback) if icon.isNull() else (icon, "")


class HistoryDialog(QDialog):
    """Modal list of recent transcripts, newest first."""

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recent dictations")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Pick one to paste where the caret is now:"))
        header.addStretch(1)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Close")
        close.setAutoRaise(True)
        close.clicked.connect(self.reject)
        header.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        self._list = QListWidget()
        for text in items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setToolTip(text)
            self._list.addItem(item)
            row = self._make_row(text)
            item.setSizeHint(row.sizeHint())
            self._list.setItemWidget(item, row)
        self._list.setCurrentRow(0)
        # Double-click, or Return with the list focused, is the whole interaction.
        self._list.itemActivated.connect(lambda _item: self.accept())
        layout.addWidget(self._list)

        paste = QPushButton("Paste")
        paste.setDefault(True)
        paste.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(paste)
        layout.addLayout(footer)

        self._list.setFocus()

    def _make_row(self, text: str) -> QWidget:
        """One list row: the preview, and a copy button for that transcript."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(preview(text))
        label.setToolTip(text)
        row_layout.addWidget(label)
        row_layout.addStretch(1)

        icon, fallback = _icon("edit-copy", "⧉")
        button = QToolButton()
        if icon is not None:
            button.setIcon(icon)
        else:
            button.setText(fallback)
        button.setToolTip("Copy to clipboard")
        button.setAutoRaise(True)
        button.clicked.connect(lambda: self._copy(text, button))
        row_layout.addWidget(button)
        return row

    @staticmethod
    def _copy(text: str, button: QToolButton) -> None:
        QGuiApplication.clipboard().setText(text)
        icon, fallback = button.icon(), button.text()
        button.setIcon(QIcon())
        button.setText("✓")
        QTimer.singleShot(
            COPIED_FEEDBACK_MS,
            lambda: (button.setIcon(icon), button.setText(fallback)),
        )

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
