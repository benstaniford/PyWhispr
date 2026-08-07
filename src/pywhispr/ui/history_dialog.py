"""Picker for the last few transcripts, to paste one where it should have gone.

A list of one-line previews rather than the full text: this is chosen by
someone who dictated the words seconds ago and only needs to tell them apart.
The whole transcript is on the row's tooltip for the case where they can't.

Each row carries its own copy button, so a transcript can be taken to the
clipboard without pasting it anywhere; copying leaves the picker open, pasting
closes it. Double-click and Return are what paste — there is no Paste button,
because the row is what you are aiming at either way.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QGuiApplication, QIcon, QPalette
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

# Rounding for the row hover and the copy button — enough to look deliberate,
# not so much that it stops looking like the rest of the app's dialogs.
RADIUS_PX = 6


def _icon(theme_name: str, fallback: str) -> tuple[QIcon | None, str]:
    """A themed icon where the platform has one, else a text glyph."""
    icon = QIcon.fromTheme(theme_name)
    return (None, fallback) if icon.isNull() else (icon, "")


def _rgba(colour: QColor, alpha: int) -> str:
    """``colour`` at ``alpha`` (0-255) as a Qt stylesheet colour."""
    return f"rgba({colour.red()},{colour.green()},{colour.blue()},{alpha})"


def _stylesheet(palette: QPalette) -> str:
    """The little that is styled here: row hover, the copy button, the scrollbar.

    Everything else — the window background, the list frame, the selection — is
    left to the platform style, so this dialog looks like the app's others. The
    few colours that are set come out of the active palette rather than being
    written down: a hard-coded tint is the one thing guaranteed to be wrong in
    whichever of dark and light it was not picked in.
    """
    text = palette.color(QPalette.ColorRole.Text)
    dark = palette.color(QPalette.ColorRole.Base).lightness() < 128

    # A tint of the text colour reads as "slightly raised off the surface" in
    # either mode, where a fixed grey only does in one.
    hover = _rgba(text, 20 if dark else 14)
    pressed = _rgba(text, 38 if dark else 30)
    muted = _rgba(text, 160)
    handle = _rgba(text, 55)
    handle_hover = _rgba(text, 95)
    return f"""
    QWidget#historyRow {{ background: transparent; border-radius: {RADIUS_PX}px; }}
    QWidget#historyRow:hover {{ background: {hover}; }}

    QToolButton {{
        border: none;
        background: transparent;
        border-radius: {RADIUS_PX}px;
        color: {muted};
    }}
    QToolButton:hover {{ background: {hover}; color: {text.name()}; }}
    QToolButton:pressed {{ background: {pressed}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0px; }}
    QScrollBar::handle:vertical {{
        background: {handle};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {handle_hover}; }}
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
        # Zero width, real height. A QLabel asks for its whole text even under
        # an Ignored size policy, and that hint propagates up through the row
        # to the list item — which laid rows out wider than the viewport and
        # left the copy button off the right-hand edge of the dialog.
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

        # No close button of our own: the window's own ✕ is right above it, and
        # two of them one under the other is what this dialog used to show.
        hint = QLabel("Double-click one to paste it where the caret is now, or copy it.")
        hint.setStyleSheet("color: gray;")  # as the app's other dialogs do it
        layout.addWidget(hint)

        self._list = QListWidget()
        for text in items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setToolTip(text)
            self._list.addItem(item)
            row = self._make_row(text)
            # Height from the row, width from the view: the row's own width hint
            # is its whole preview, which laid the row out wider than the
            # viewport and carried the copy button off the right-hand edge.
            item.setSizeHint(QSize(0, row.sizeHint().height()))
            self._list.setItemWidget(item, row)
        # Connected before the first selection, so row 0 is recoloured too.
        self._list.currentItemChanged.connect(self._recolour_for_selection)
        self._list.setCurrentRow(0)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Double-click, or Return with the list focused, is the whole interaction.
        self._list.itemActivated.connect(lambda _item: self.accept())
        layout.addWidget(self._list)
        self._fit_list_to_contents()

        # No Paste button: double-click and Return already paste, and a button
        # for it only adds a third thing to aim at in a dialog this small.
        self._list.setFocus()
        # The list is now exactly as tall as it needs to be; take the dialog
        # down with it rather than leaving the default window height.
        self.adjustSize()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # A row widget keeps the geometry the view gave it when it was added,
        # which predates the dialog's final width; only a later resize corrects
        # it. Without this the copy buttons start off the right-hand edge.
        self._list.doItemsLayout()

    def _make_row(self, text: str) -> QWidget:
        """One list row: the preview, and a copy button for that transcript."""
        row = QWidget()
        # Named so the stylesheet can give the row itself a hover state: the
        # list's own ::item:hover never fires, because the row widget is what
        # the mouse is actually over.
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
        """Put the selected row's text on the platform's highlighted-text colour.

        The rows are widgets, so the view's selection paints *behind* them and
        their labels keep the ordinary text colour — black on a solid blue
        selection, which is the one row you cannot read.
        """
        highlighted = self.palette().color(QPalette.ColorRole.HighlightedText).name()
        for item, colour in ((previous, ""), (current, f"color: {highlighted};")):
            row = None if item is None else self._list.itemWidget(item)
            if row is None:
                continue
            for child in row.findChildren(QWidget):  # the preview and its copy button
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
        icon, fallback, tooltip = button.icon(), button.text(), button.toolTip()
        # Kept, not cleared, on the way back: the selected row's button carries a
        # colour of its own (see _recolour_for_selection).
        sheet = button.styleSheet()
        button.setIcon(QIcon())
        button.setText("✓")
        button.setToolTip("Copied")
        # The tick has to be spotted in passing, so it gets the accent colour and
        # some weight rather than the muted grey the copy glyph sits in.
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
