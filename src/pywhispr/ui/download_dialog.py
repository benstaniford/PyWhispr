"""First-run progress for the model download.

Dictation cannot start until it finishes, so this is shown rather than hidden in
the tray tooltip. Progress is polled from the cache size (see
:mod:`pywhispr.download`) because the downloader offers no callback.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout

from pywhispr.download import APPROXIMATE_MODEL_MB, cache_bytes

POLL_MS = 500
MEGABYTE = 1024 * 1024


class ModelDownloadDialog(QDialog):
    """A bar that follows the Hugging Face cache growing."""

    def __init__(self, expected_mb: int = APPROXIMATE_MODEL_MB):
        super().__init__()
        self.setWindowTitle("PyWhispr — first run")
        self.setModal(False)
        self.setMinimumWidth(440)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self._expected = expected_mb
        self._start_bytes = cache_bytes()

        self._status = QLabel(
            f"Downloading the speech model (about {expected_mb} MB, once).\n"
            "Dictation starts as soon as it is ready."
        )
        self._status.setWordWrap(True)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._buttons = QDialogButtonBox()
        hide = self._buttons.addButton("Hide", QDialogButtonBox.ButtonRole.RejectRole)
        hide.clicked.connect(self.hide)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(self._bar)
        layout.addWidget(self._buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _poll(self) -> None:
        downloaded = max(0, cache_bytes() - self._start_bytes) // MEGABYTE
        if downloaded >= self._expected:
            self._bar.setRange(0, 0)  # overshot the estimate: keep it honest
            self._status.setText(f"Downloading the speech model ({downloaded} MB so far)…")
            return
        self._bar.setValue(int(downloaded / self._expected * 1000))
        self._status.setText(
            f"Downloading the speech model — {downloaded} of about {self._expected} MB.\n"
            "Dictation starts as soon as it is ready."
        )

    def finish(self, message: str | None = None) -> None:
        """Called when the model is ready, or the load failed."""
        self._timer.stop()
        if message is None:
            self.accept()
            return
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._status.setText(message)
        self._buttons.clear()
        close = self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close.clicked.connect(self.accept)
