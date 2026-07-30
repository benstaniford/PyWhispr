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

from PySide6.QtCore import QObject, QThread, QTimer, Signal
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


class _Worker(QObject):
    """Downloads the libraries and the weights, then proves the GPU is used."""

    progress = Signal(float, str)
    finished = Signal(bool, str)  # worked, detail

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False
        self._process = None
        self._cache_at_start = 0
        self._wheel_bytes = 0
        # Libraries already on disk are not part of what is about to be downloaded,
        # so the total is what this run will actually fetch.
        self._total_mb = APPROXIMATE_MODEL_MB
        if not cuda.is_installed():
            self._total_mb += cuda.APPROXIMATE_DOWNLOAD_MB

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
                cuda.download(
                    lambda fraction, message: self._report_wheel(message),
                    on_bytes=self._on_wheel_bytes,
                )
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
        """What this run has fetched: wheels as they stream, weights as they land.

        Wheel bytes come from the downloader rather than from the installed files,
        which are half again bigger and only appear once each wheel is extracted.
        """
        weights = max(0, cache_bytes() - self._cache_at_start)
        return (self._wheel_bytes + weights) // MEGABYTE

    def _on_wheel_bytes(self, total: int) -> None:
        self._wheel_bytes = total
        self._emit(self._downloaded_mb())

    def _emit(self, downloaded_mb: int) -> None:
        self.progress.emit(
            downloaded_mb / self._total_mb,  # may exceed 1: the dialog says so honestly
            f"Downloading — {downloaded_mb} of about {self._total_mb} MB.",
        )

    def _report_wheel(self, message: str) -> bool:
        """Called per wheel: the text follows the step, the bar follows the bytes."""
        log.debug("GPU setup: %s", message)
        return not self._cancelled


class GpuSetupDialog(QDialog):
    """Progress and outcome for the CUDA download."""

    # Emitted when the work is done, not when the window is closed: on a first run
    # the model load is waiting on this.
    setup_finished = Signal(bool)

    def __init__(self, first_run: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("PyWhispr — GPU acceleration")
        self.setModal(False)
        self.setMinimumWidth(460)
        self.worked = False
        # On a first run the model loads straight after this, in this process:
        # nothing has built a session yet, so there is nothing to restart for.
        self._first_run = first_run
        self._model_timer = None
        self._model_expected_mb = APPROXIMATE_MODEL_MB
        self._model_start_bytes = 0

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
        if fraction >= 1.0:
            self._bar.setRange(0, 0)  # past the estimate: busy, rather than a fake 99%
        else:
            self._bar.setRange(0, 1000)
            self._bar.setValue(int(fraction * 1000))
        self._status.setText(message)

    def _on_finished(self, worked: bool, detail: str) -> None:
        self.worked = worked
        # The verdict used to live only in this label, so a failure the user closed
        # left no trace anywhere.
        log.info("GPU setup finished: worked=%s (%s)", worked, detail)
        self._thread.quit()
        self._thread.wait(5000)
        self._bar.setRange(0, 1000)
        self._bar.setValue(1000 if worked else self._bar.value())
        self._buttons.clear()
        self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._buttons.setEnabled(True)
        self._buttons.rejected.connect(self.accept)
        if worked:
            next_step = (
                "Loading the model now — dictation is about to be ready."
                if self._first_run
                else "Restart PyWhispr to start using it."
            )
            self._status.setText(
                f"GPU acceleration is ready and working — {detail}.\n\n"
                f"{next_step} Nothing else to download."
            )
        else:
            from pywhispr.logging_setup import log_path

            on_the_cpu = (
                "Dictation will start on the CPU instead"
                if self._first_run
                else "Dictation keeps working on the CPU"
            )
            self._status.setText(
                f"GPU acceleration was not enabled: {detail}\n\n"
                f"{on_the_cpu}. Details are in {log_path()}."
            )
        self.setup_finished.emit(worked)

    # -- carrying on as the model download ----------------------------------
    #
    # One window from the offer to "ready". Closing this and opening another left
    # two of them on screen counting the same bytes.

    def track_model_download(self, expected_mb: int) -> None:
        """Follow the model download in this window, after the setup has finished."""
        self._model_expected_mb = expected_mb
        self._model_start_bytes = cache_bytes()
        self._buttons.clear()
        hide = self._buttons.addButton("Hide", QDialogButtonBox.ButtonRole.RejectRole)
        hide.clicked.connect(self.hide)
        self._buttons.setEnabled(True)
        self._model_timer = QTimer(self)
        self._model_timer.setInterval(int(POLL_SECONDS * 1000))
        self._model_timer.timeout.connect(self._poll_model)
        self._model_timer.start()
        self._poll_model()

    def _poll_model(self) -> None:
        downloaded = max(0, cache_bytes() - self._model_start_bytes) // MEGABYTE
        if downloaded >= self._model_expected_mb:
            self._bar.setRange(0, 0)
            self._status.setText(f"Loading the speech model ({downloaded} MB so far)…")
            return
        self._bar.setRange(0, 1000)
        self._bar.setValue(int(downloaded / self._model_expected_mb * 1000))
        self._status.setText(
            f"Downloading the speech model — {downloaded} of about "
            f"{self._model_expected_mb} MB.\nDictation starts as soon as it is ready."
        )

    def finish(self, message: str | None = None) -> None:
        """The model is ready, or it failed. Same contract as ModelDownloadDialog."""
        if self._model_timer is not None:
            self._model_timer.stop()
        if message is None:
            self.accept()
            return
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._status.setText(message)
        self._buttons.clear()
        close = self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close.clicked.connect(self.accept)


def ask_to_enable(parent=None, first_run: bool = False) -> bool | None:
    """Offer GPU acceleration. True = yes, False = not now, None = never ask again.

    ``first_run`` because the two cases promise different things: on a first run
    there is no model yet, so dictation starts when the download finishes rather
    than carrying on through it.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("PyWhispr — GPU acceleration available")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText("GPU acceleration is available — it makes transcription near instant.")
    if first_run:
        # Either answer downloads something — the GPU and CPU models are separate
        # files — and neither can dictate until it has finished.
        body = (
            "Set it up now? Either way there is a one-time download first, and dictation "
            "cannot start until it finishes. Saying yes downloads more. "
            "“pywhispr disable-gpu” undoes it later."
        )
    else:
        body = (
            "Set it up now? There is a one-time download first. Dictation keeps working "
            "while it runs, and “pywhispr disable-gpu” undoes it."
        )
    box.setInformativeText(body)
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


def run_setup(first_run: bool = False) -> GpuSetupDialog:
    """Show the progress window and start work. The caller keeps the reference.

    Not modal: the download is gigabytes, and on anything but a first run dictation
    keeps working throughout.
    """
    dialog = GpuSetupDialog(first_run)
    show_in_front(dialog)
    dialog.start()
    return dialog


__all__ = ["GpuSetupDialog", "ask_to_enable", "run_setup"]
