"""Dialog for setting the transcription server (PyWhisprLite only).

Lite runs no model of its own; it sends audio to a PyWhispr server's
``/v1/transcribe``. This asks for that server's address — on first run when none
is set. The settings page has the same field. A plain text field: the
value is one URL, and the placeholder shows the shape it should take.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from pywhispr import flavor
from pywhispr.ui.foreground import show_in_front


class ServerDialog(QDialog):
    """Modal prompt for the server URL."""

    def __init__(self, current: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{flavor.PRODUCT_NAME} — set server")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Enter the address of the PyWhispr server to transcribe with.\n"
                "This is the machine running the full PyWhispr with its API on."
            )
        )

        self._field = QLineEdit(current)
        self._field.setPlaceholderText("e.g. 192.168.1.20:9149  or  http://desktop.local:9149")
        self._field.setClearButtonEnabled(True)
        self._field.textChanged.connect(self._update_enabled)
        layout.addWidget(self._field)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._update_enabled()

    def _update_enabled(self) -> None:
        # A blank server is the one thing that cannot work, so Save is gated on it.
        save = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setEnabled(bool(self._field.text().strip()))

    def server_url(self) -> str:
        return self._field.text().strip()

    @staticmethod
    def get_server_url(current: str = "", parent=None) -> str | None:
        """Show the dialog; return the entered URL, or None if cancelled."""
        dialog = ServerDialog(current, parent)
        show_in_front(dialog)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.server_url()
        return None
