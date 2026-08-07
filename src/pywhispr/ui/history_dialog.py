"""Picker for the last few transcripts, to paste one where it should have gone.

A list of one-line previews rather than the full text: this is chosen by
someone who dictated the words seconds ago and only needs to tell them apart.
The whole transcript is on the row's tooltip for the case where they can't.

Each row carries its own copy button, so a transcript can be taken to the
clipboard without pasting it anywhere; copying leaves the picker open, pasting
closes it. Double-click and Return are what paste.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QFontMetrics, QGuiApplication, QIcon, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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


def _stylesheet(palette: QPalette) -> str:
    """Row hover, flat copy button, thin scrollbar. The rest is the platform's."""

    text = palette.color(QPalette.ColorRole.Text)

    def tint(alpha: int) -> str:
        return f"rgba({text.red()},{text.green()},{text.blue()},{alpha})"

    return f"""
    QWidget#historyRow {{ background: transparent; border-radius: 6px; }}
    QWidget#historyRow:hover {{ background: {tint(18)}; }}

    QToolButton {{
        border: none;
        background: transparent;
        border-radius: 6px;
        color: {tint(160)};
    }}
    QToolButton:hover {{ background: {tint(18)}; color: {text.name()}; }}
    QToolButton:pressed {{ background: {tint(34)}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0px; }}
    QScrollBar::handle:vertical {{ background: {tint(55)}; border-radius: 5px; min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: {tint(95)}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    """


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

    def sizeHint(self) -> QSize:
        # Zero width: a QLabel asks for its whole text even when its size policy
        # is Ignored, which laid the rows out wider than the viewport and put the
        # copy button off the right-hand edge of the dialog.
        return QSize(0, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

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
        self.setMinimumWidth(480)
        self.setStyleSheet(_stylesheet(self.palette()))

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        hint = QLabel("Double-click one to paste it where the caret is now, or copy it.")
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        self._list = QListWidget()
        for text in items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setToolTip(text)
            self._list.addItem(item)
            row = self._make_row(text)
            item.setSizeHint(QSize(0, row.sizeHint().height()))  # width is the view's
            self._list.setItemWidget(item, row)
        # Before the first selection, so row 0 is recoloured too.
        self._list.currentItemChanged.connect(self._recolour_for_selection)
        self._list.setCurrentRow(0)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Double-click, or Return with the list focused, is the whole interaction.
        self._list.itemActivated.connect(lambda _item: self.accept())
        layout.addWidget(self._list)
        self._fit_list_to_contents()

        self._list.setFocus()
        # The list is now exactly as tall as it needs to be; take the dialog
        # down with it rather than leaving the default window height.
        self.adjustSize()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # A row keeps the width it was given before the dialog reached its final
        # one; without this the copy buttons start off the right-hand edge.
        self._list.doItemsLayout()

    def _make_row(self, text: str) -> QWidget:
        """One list row: the preview, and a copy button for that transcript."""
        row = QWidget()
        # The stylesheet hovers the row, not the item: the mouse is over this.
        row.setObjectName("historyRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(10, 8, 6, 8)
        row_layout.setSpacing(8)
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
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedSize(28, 28)
        # Never squeezed out by a long preview, at any dialog width.
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.clicked.connect(lambda: self._copy(text, button))
        row_layout.addWidget(button, 0)
        return row

    def _recolour_for_selection(self, current, previous) -> None:
        """Selection is painted behind the row widgets, so their text has to follow
        it by hand — otherwise the selected row is black on solid blue."""
        highlighted = self.palette().color(QPalette.ColorRole.HighlightedText).name()
        for item, colour in ((previous, ""), (current, f"color: {highlighted};")):
            row = None if item is None else self._list.itemWidget(item)
            if row is None:
                continue
            for child in row.findChildren(QWidget):
                child.setStyleSheet(colour)

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

    def _copy(self, text: str, button: QToolButton) -> None:
        QGuiApplication.clipboard().setText(text)
        # The sheet too: on the selected row it carries _recolour_for_selection's.
        icon, fallback, tooltip = button.icon(), button.text(), button.toolTip()
        sheet = button.styleSheet()
        button.setIcon(QIcon())
        button.setText("✓")
        button.setToolTip("Copied")
        accent = self.palette().color(QPalette.ColorRole.Highlight)
        button.setStyleSheet(f"color: {accent.name()}; font-weight: 700; font-size: 15px;")

        def restore() -> None:
            button.setStyleSheet(sheet)
            button.setIcon(icon)
            button.setText(fallback)
            button.setToolTip(tooltip)

        # Bound to the button: copy, then close the picker before the feedback
        # expires, and a timer that outlived its widget would reach into a
        # deleted C++ object and take the app down.
        QTimer.singleShot(COPIED_FEEDBACK_MS, button, restore)

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
