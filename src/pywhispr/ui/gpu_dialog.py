"""The GPU acceleration offer, its progress, and its verdict.

The user is never told they have GPU acceleration until a real transcription has
run on it — see :func:`pywhispr.cuda.verify`.

The download runs on a worker thread; widgets are touched only from the main one.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
)

from pywhispr import cuda

log = logging.getLogger(__name__)


class _Worker(QObject):
    """Downloads, then verifies, reporting progress as it goes."""

    progress = Signal(float, str)
    finished = Signal(bool, str)  # worked, detail

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if not cuda.is_installed():
                cuda.download(lambda fraction, message: self._report(fraction, message))
            self.progress.emit(0.97, "Checking that the GPU is really being used…")
            worked, detail = cuda.verify()
        except KeyboardInterrupt:
            self.finished.emit(False, "Cancelled.")
        except Exception as exc:
            log.exception("GPU setup failed")
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(worked, detail)

    def _report(self, fraction: float, message: str) -> bool:
        self.progress.emit(fraction * 0.95, message)
        return not self._cancelled


class GpuSetupDialog(QDialog):
    """Progress and outcome for the CUDA download."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyWhispr — GPU acceleration")
        self.setModal(False)
        self.setMinimumWidth(460)
        self.worked = False

        self._status = QLabel("Starting…")
        self._status.setWordWrap(True)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._buttons.rejected.connect(self._cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(self._bar)
        layout.addWidget(self._buttons)

        self._worker = _Worker()
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

    def start(self) -> None:
        self._thread.start()

    def _cancel(self) -> None:
        self._status.setText("Cancelling…")
        self._buttons.setEnabled(False)
        self._worker.cancel()

    def _on_progress(self, fraction: float, message: str) -> None:
        self._bar.setValue(int(fraction * 1000))
        self._status.setText(message)

    def _on_finished(self, worked: bool, detail: str) -> None:
        self.worked = worked
        self._thread.quit()
        self._thread.wait(5000)
        self._bar.setValue(1000 if worked else self._bar.value())
        self._buttons.clear()
        self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._buttons.setEnabled(True)
        self._buttons.rejected.connect(self.accept)
        if worked:
            self._status.setText(
                "GPU acceleration is ready and working — "
                f"{detail}.\n\nRestart PyWhispr to start using it."
            )
        else:
            from pywhispr.logging_setup import log_path

            self._status.setText(
                f"GPU acceleration was not enabled: {detail}\n\n"
                f"Dictation keeps working on the CPU. Details are in {log_path()}."
            )


def ask_to_enable(parent=None) -> bool | None:
    """Offer GPU acceleration. True = yes, False = not now, None = never ask again."""
    box = QMessageBox(parent)
    box.setWindowTitle("PyWhispr — GPU acceleration available")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText("Your NVIDIA GPU can transcribe several times faster than the CPU.")
    box.setInformativeText(
        f"PyWhispr can download the CUDA libraries it needs "
        f"(about {cuda.APPROXIMATE_DOWNLOAD_MB} MB, one time). No admin rights are needed, "
        "and dictation keeps working while it downloads.\n\n"
        "PyWhispr will check that the GPU is genuinely being used before telling you it "
        "worked, and you can undo it any time with “pywhispr disable-gpu”."
    )
    download = box.addButton("Download", QMessageBox.ButtonRole.AcceptRole)
    later = box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
    never = box.addButton("Never", QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(download)
    box.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    box.exec()

    clicked = box.clickedButton()
    if clicked is download:
        return True
    if clicked is never:
        return None
    assert clicked is later or clicked is None
    return False


def run_setup() -> GpuSetupDialog:
    """Show the progress window and start work. The caller keeps the reference.

    Not modal: the download is a gigabyte and dictation keeps working throughout.
    """
    dialog = GpuSetupDialog()
    dialog.show()
    dialog.start()
    return dialog


__all__ = ["GpuSetupDialog", "ask_to_enable", "run_setup"]
