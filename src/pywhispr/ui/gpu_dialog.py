"""The GPU acceleration offer, its progress, and its verdict.

The user is never told they have GPU acceleration until a real transcription has
run on it — see :func:`pywhispr.cuda.start_verification`.

Both downloads — the CUDA libraries and the full-precision weights the GPU wants —
are shown as one bar counting real megabytes, because they are one decision as far
as the user is concerned. The weights are fetched by the verification process
itself, so there is no separate silent wait afterwards.

The work runs on a worker thread; widgets are touched only from the main one.
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
)

from pywhispr import cuda
from pywhispr.download import APPROXIMATE_MODEL_MB, cache_bytes
from pywhispr.ui.foreground import show_in_front

log = logging.getLogger(__name__)

MEGABYTE = 1024 * 1024
POLL_SECONDS = 0.5

# Everything the offer commits the user to, in one number.
TOTAL_DOWNLOAD_MB = cuda.APPROXIMATE_DOWNLOAD_MB + APPROXIMATE_MODEL_MB

# The GPU is only worth having on the full-precision weights: int8 has no CUDA
# kernels and measured *slower* than the CPU (1.60s against 0.43s).
GPU_QUANTIZATION = ""

# A wall-clock timeout cannot tell a 2.4 GB download from a hang, so what is
# bounded is silence: no new bytes and no exit for this long is a hang.
STALL_SECONDS = 240.0


def _directory_bytes(path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


class _Worker(QObject):
    """Downloads the libraries and the weights, then proves the GPU is used."""

    progress = Signal(float, str)
    finished = Signal(bool, str)  # worked, detail

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False
        self._process = None
        self._cache_at_start = 0

    def cancel(self) -> None:
        """Stop as soon as possible, including killing the check if it is running."""
        self._cancelled = True
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()  # otherwise cancelling waits out a multi-gigabyte download

    def run(self) -> None:
        try:
            self._cache_at_start = cache_bytes()
            if not cuda.is_installed():
                cuda.download(lambda fraction, message: self._report_wheel(message))
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return
            worked, detail = self._verify()
        except KeyboardInterrupt:
            self.finished.emit(False, "Cancelled.")
        except Exception as exc:
            log.exception("GPU setup failed")
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(worked, detail)

    def _verify(self) -> tuple[bool, str]:
        """Run the check, reporting the weights it downloads as it goes."""
        self._process = cuda.start_verification(GPU_QUANTIZATION)
        seen = self._downloaded_mb()
        last_change = time.monotonic()
        while self._process.poll() is None:
            time.sleep(POLL_SECONDS)
            if self._cancelled:
                return False, "Cancelled."
            downloaded = self._downloaded_mb()
            if downloaded != seen:
                seen, last_change = downloaded, time.monotonic()
            elif time.monotonic() - last_change > STALL_SECONDS:
                self._process.kill()
                return False, "the download stopped making progress"
            self._emit(downloaded)
        return cuda.finish_verification(self._process)

    def _downloaded_mb(self) -> int:
        libraries = _directory_bytes(cuda.install_dir())
        weights = max(0, cache_bytes() - self._cache_at_start)
        return (libraries + weights) // MEGABYTE

    def _emit(self, downloaded_mb: int) -> None:
        self.progress.emit(
            min(0.99, downloaded_mb / TOTAL_DOWNLOAD_MB),
            f"Downloading — {downloaded_mb} of about {TOTAL_DOWNLOAD_MB} MB.",
        )

    def _report_wheel(self, message: str) -> bool:
        """Called between wheels: the bar follows bytes, the text follows the step."""
        self._emit(self._downloaded_mb())
        log.debug("GPU setup: %s", message)
        return not self._cancelled


class GpuSetupDialog(QDialog):
    """Progress and outcome for the CUDA download."""

    # Emitted when the work is done, not when the window is closed: on a first run
    # the model load is waiting on this.
    setup_finished = Signal(bool)

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
                f"GPU acceleration is ready and working — {detail}.\n\n"
                "Restart PyWhispr to start using it. Nothing else to download."
            )
        else:
            from pywhispr.logging_setup import log_path

            self._status.setText(
                f"GPU acceleration was not enabled: {detail}\n\n"
                f"Dictation keeps working on the CPU. Details are in {log_path()}."
            )
        self.setup_finished.emit(worked)


def ask_to_enable(parent=None) -> bool | None:
    """Offer GPU acceleration. True = yes, False = not now, None = never ask again."""
    box = QMessageBox(parent)
    box.setWindowTitle("PyWhispr — GPU acceleration available")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText("GPU acceleration is available — it makes transcription near instant.")
    box.setInformativeText(
        "Set it up now? There is a one-time download, shown as it goes. Dictation keeps "
        "working throughout, and “pywhispr disable-gpu” undoes it."
    )
    download = box.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
    later = box.addButton("No", QMessageBox.ButtonRole.RejectRole)
    never = box.addButton("Never", QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(download)
    show_in_front(box)
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

    Not modal: the download is gigabytes and dictation keeps working throughout.
    """
    dialog = GpuSetupDialog()
    show_in_front(dialog)
    dialog.start()
    return dialog


__all__ = ["GpuSetupDialog", "ask_to_enable", "run_setup"]
