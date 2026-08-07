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
from PySide6.QtGui import QFontMetrics, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pywhispr.history import preview
from pywhispr.ui.foreground import centre_on_active_screen

# Shown on the copy button after a copy, so the click has an answer. Long enough
# to read, short enough that the button is itself again before the next one.
COPIED_FEEDBACK_MS = 1200

# How many rows are shown before the list scrolls. The dialog is sized to its
# contents, so fewer transcripts than this means a shorter dialog, not empty
# space, and more means a scrollbar rather than a window down to the taskbar.
MAX_VISIBLE_ROWS = 6


def _icon(theme_name: str, fallback: str) -> tuple[QIcon | None, str]:
    """A themed icon where the platform has one, else a text glyph."""
    icon = QIcon.fromTheme(theme_name)
    return (None, fallback) if icon.isNull() else (icon, "")


class _ElidingLabel(QLabel):
    """A preview that gives way to the copy button instead of pushing it off.

    A QLabel's minimum width is its full text, so in a row layout the button
    next to it is the thing that gets clipped when the dialog is at its minimum
    width. This one is free to shrink and elides what no longer fits — the whole
    transcript is on the tooltip anyway.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setText(text)

    def text(self) -> str:
        # What the row *is*, not what currently fits: callers (and tests) mean
        # the preview, and the elision is a painting detail of the width.
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, self.width()))


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
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Double-click, or Return with the list focused, is the whole interaction.
        self._list.itemActivated.connect(lambda _item: self.accept())
        layout.addWidget(self._list)
        self._fit_list_to_contents()

        paste = QPushButton("Paste")
        paste.setDefault(True)
        paste.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(paste)
        layout.addLayout(footer)

        self._list.setFocus()
        # The list is now exactly as tall as it needs to be; take the dialog
        # down with it rather than leaving the default window height.
        self.adjustSize()

    def _make_row(self, text: str) -> QWidget:
        """One list row: the preview, and a copy button for that transcript."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        label = _ElidingLabel(preview(text))
        label.setToolTip(text)
        row_layout.addWidget(label, 1)

        icon, fallback = _icon("edit-copy", "⧉")
        button = QToolButton()
        if icon is not None:
            button.setIcon(icon)
        else:
            button.setText(fallback)
        button.setToolTip("Copy to clipboard")
        button.setAutoRaise(True)
        # Never squeezed out by a long preview, at any dialog width.
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.clicked.connect(lambda: self._copy(text, button))
        row_layout.addWidget(button, 0)
        return row

    def _fit_list_to_contents(self) -> None:
        """Height of the rows there are, capped at MAX_VISIBLE_ROWS.

        Without this the list keeps its default height whatever it holds: a
        column of empty space under two transcripts, and a window taller than
        the screen once the history is full.
        """
        rows = self._list.count()
        if not rows:
            return
        visible = min(rows, MAX_VISIBLE_ROWS)
        row_height = max(self._list.sizeHintForRow(i) for i in range(rows))
        frame = 2 * self._list.frameWidth()
        self._list.setFixedHeight(visible * row_height + frame)

    @staticmethod
    def _copy(text: str, button: QToolButton) -> None:
        QGuiApplication.clipboard().setText(text)
        icon, fallback = button.icon(), button.text()
        button.setIcon(QIcon())
        button.setText("✓")
        # Bound to the button: copy, then close the picker before the feedback
        # expires, and a timer that outlived its widget would reach into a
        # deleted C++ object and take the app down.
        QTimer.singleShot(
            COPIED_FEEDBACK_MS,
            button,
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
