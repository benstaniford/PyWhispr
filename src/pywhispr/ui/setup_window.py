"""The one window that reports everything a first run has to download.

There are two things that can be downloading — the speech model and the CUDA
libraries — and either can start while the other is running: the model begins at
startup, and GPU setup can be triggered from the tray at any moment. Two windows
for that is what the user actually saw, each with its own bar, one of them
counting bytes the other had already counted.

So there is one window with a line per activity and a single bar over the sum.
It is created when the first thing starts, and closes when nothing is left.
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
from pywhispr.download import APPROXIMATE_MODEL_MB, cache_bytes, start_model_download
from pywhispr.ui.foreground import show_in_front

log = logging.getLogger(__name__)

MEGABYTE = 1024 * 1024
POLL_SECONDS = 0.5

# The GPU is only worth having on the full-precision weights: int8 has no CUDA
# kernels and measured *slower* than the CPU (1.60s against 0.43s).
GPU_QUANTIZATION = ""

# A wall-clock timeout cannot tell a 2.4 GB download from a hang, so what is
# bounded is silence: no new bytes and no exit for this long is a hang.
STALL_SECONDS = 240.0

# How long the byte count may stand still before the bar goes busy rather than
# looking frozen. Well under STALL_SECONDS: this is presentation, not a verdict.
QUIET_SECONDS = 4.0


class _Worker(QObject):
    """Three steps, in this order: CUDA libraries, model weights, then the check.

    The download is the long part, so it happens as its own visible step and the
    check that follows it is a session build and one transcription — seconds. Doing
    it the other way round hid a 2.4 GB download behind the word "checking".
    """

    progress = Signal(int, int)  # downloaded MB, total MB — libraries and weights
    step = Signal(str)  # what is happening while no bytes are moving
    checking = Signal()  # downloading is done; the bar has nothing left to show
    finished = Signal(bool, str)  # worked, detail

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False
        self._process = None
        self._wheel_bytes = 0
        self._cache_at_start = 0
        self._total_mb = APPROXIMATE_MODEL_MB
        if not cuda.is_installed():
            # Libraries already on disk are not part of what is about to be fetched.
            self._total_mb += cuda.APPROXIMATE_DOWNLOAD_MB

    def cancel(self) -> None:
        """Stop as soon as possible, including killing whatever step is running."""
        self._cancelled = True
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()  # otherwise cancelling waits out a multi-gigabyte download

    def run(self) -> None:
        try:
            self._cache_at_start = cache_bytes()
            if not cuda.is_installed():
                cuda.download(
                    lambda fraction, message: self._step(message),
                    on_bytes=self._on_wheel_bytes,
                )
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return

            # Full precision: the only variant the GPU will ever run, so it has to
            # come down either way and int8 would prove the wrong thing. Downloaded
            # as its own step so the bar covers it, rather than inside the check.
            self.step.emit("Starting the speech model download…")
            ok, detail = self._await(start_model_download(GPU_QUANTIZATION))
            if not ok:
                self.finished.emit(False, detail)
                return

            self.checking.emit()
            worked, detail = self._await(cuda.start_verification(GPU_QUANTIZATION))
        except KeyboardInterrupt:
            self.finished.emit(False, "Cancelled.")
        except Exception as exc:
            log.exception("GPU setup failed")
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(worked, detail)

    def _await(self, process) -> tuple[bool, str]:
        """Wait for a step, cancellable, bounded by silence rather than by a clock.

        A 2.4 GB download is slow but fine; no new bytes and no exit is a hang.
        """
        self._process = process
        seen = self._downloaded_mb()
        last_change = time.monotonic()
        while process.poll() is None:
            time.sleep(POLL_SECONDS)
            if self._cancelled:
                return False, "Cancelled."
            downloaded = self._downloaded_mb()
            if downloaded != seen:
                seen, last_change = downloaded, time.monotonic()
                self.progress.emit(downloaded, self._total_mb)
            elif time.monotonic() - last_change > STALL_SECONDS:
                process.kill()
                return False, "it stopped making progress"
        return cuda.finish_verification(process)

    def _downloaded_mb(self) -> int:
        """Wheels as they stream, weights as they land — one rising number.

        Wheel bytes come from the downloader rather than the installed files, which
        are half again bigger and only appear once each wheel is extracted.
        """
        weights = max(0, cache_bytes() - self._cache_at_start)
        return (self._wheel_bytes + weights) // MEGABYTE

    def _on_wheel_bytes(self, total: int) -> None:
        self._wheel_bytes = total
        self.progress.emit(self._downloaded_mb(), self._total_mb)

    def _step(self, message: str) -> bool:
        """Wheel-level progress. The byte count stands still while a wheel extracts —
        cuDNN takes over a minute — so the step itself is what there is to report."""
        log.debug("GPU setup: %s", message)
        if message.startswith(("Finding", "Installing")):
            self.step.emit(message)
        return not self._cancelled


class SetupWindow(QDialog):
    """One window, one bar, a line per thing being downloaded."""

    # Emitted when the GPU work is done, not when the window is closed: on a first
    # run the model load is waiting on this.
    setup_finished = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyWhispr — setting up")
        self.setModal(False)
        self.setMinimumWidth(470)
        self.worked = False

        self._model_line = QLabel()
        self._gpu_line = QLabel()
        for label in (self._model_line, self._gpu_line):
            label.setWordWrap(True)
            label.hide()  # a line appears when its download does
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._buttons = QDialogButtonBox()
        self._hide_button = self._buttons.addButton("Hide", QDialogButtonBox.ButtonRole.ResetRole)
        self._hide_button.clicked.connect(self.hide)

        layout = QVBoxLayout(self)
        layout.addWidget(self._model_line)
        layout.addWidget(self._gpu_line)
        layout.addWidget(self._bar)
        layout.addWidget(self._buttons)

        self._model_done = 0
        self._model_total = 0
        self._gpu_done = 0
        self._gpu_total = 0
        self._model_start_bytes = 0
        self._model_timer = None
        self._thread = None
        self._worker = None
        self._cancel_button = None
        self._first_run = False
        self._quiet_timer = None
        self._last_progress_at = 0.0

    # -- the speech model ----------------------------------------------------

    def track_model_download(self, expected_mb: int) -> None:
        self._model_total = expected_mb
        self._model_start_bytes = cache_bytes()
        self._model_line.show()
        self._model_timer = QTimer(self)
        self._model_timer.setInterval(int(POLL_SECONDS * 1000))
        self._model_timer.timeout.connect(self._poll_model)
        self._model_timer.start()
        self._poll_model()

    def _poll_model(self) -> None:
        self._model_done = max(0, cache_bytes() - self._model_start_bytes) // MEGABYTE
        self._model_line.setText(
            f"Speech model — {self._model_done} of about {self._model_total} MB."
        )
        self._refresh_bar()

    def finish_model(self, message: str | None = None) -> None:
        """The model is ready, or its load failed."""
        if self._model_timer is not None:
            self._model_timer.stop()
            self._model_timer = None
        if message is not None:
            self._model_line.setText(message)
            self._model_line.show()
            self._close_when_idle()
            return
        self._model_line.setText("Speech model — ready.")
        self._model_total = 0
        self._close_when_idle()

    # -- GPU acceleration ----------------------------------------------------

    def start_gpu_setup(self, first_run: bool = False) -> None:
        self._first_run = first_run
        self._gpu_line.setText("GPU acceleration — starting…")
        self._gpu_line.show()
        self._cancel_button = self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._cancel_button.clicked.connect(self._cancel_gpu)

        self._last_progress_at = time.monotonic()
        self._quiet_timer = QTimer(self)
        self._quiet_timer.setInterval(1000)
        self._quiet_timer.timeout.connect(self._watch_for_quiet)
        self._quiet_timer.start()

        self._worker = _Worker()
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_gpu_progress)
        self._worker.step.connect(self._on_step)
        self._worker.checking.connect(self._on_checking)
        self._worker.finished.connect(self._on_gpu_finished)
        self._thread.start()

    @property
    def gpu_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _cancel_gpu(self) -> None:
        self._gpu_line.setText("GPU acceleration — cancelling…")
        if self._cancel_button is not None:
            self._cancel_button.setEnabled(False)
        if self._worker is not None:
            self._worker.cancel()

    def _on_gpu_progress(self, downloaded_mb: int, total_mb: int) -> None:
        self._gpu_done, self._gpu_total = downloaded_mb, total_mb
        self._last_progress_at = time.monotonic()
        self._gpu_line.setText(
            f"GPU acceleration — downloading {downloaded_mb} of about {total_mb} MB."
        )
        self._refresh_bar()

    def _watch_for_quiet(self) -> None:
        """Show motion while the byte count stands still.

        Hugging Face commits its blobs in large chunks, so the count jumps rather
        than climbs — measured at 11 updates across 2431 MB, with 70s between two of
        them. A bar frozen that long reads as a hang, so it goes busy instead; the
        number stays as the last thing actually known.
        """
        if not self.gpu_running or self._gpu_total <= 0:
            return
        if time.monotonic() - self._last_progress_at > QUIET_SECONDS:
            self._bar.setRange(0, 0)

    def _on_step(self, message: str) -> None:
        """No bytes are moving, so say what is happening and let the bar run busy.

        A frozen number for 72 seconds — measured, extracting cuDNN — reads as a
        hang; a moving bar with a caption does not.
        """
        self._gpu_line.setText(f"GPU acceleration — {message[0].lower()}{message[1:]}")
        self._bar.show()
        self._bar.setRange(0, 0)

    def _on_checking(self) -> None:
        """Downloading done. Nothing left to measure, so the bar goes away."""
        self._gpu_done = self._gpu_total = 0
        self._gpu_line.setText("Verifying GPU acceleration…")
        self._refresh_bar()

    def _on_gpu_finished(self, worked: bool, detail: str) -> None:
        self.worked = worked
        # The verdict used to live only in a label, so a failure the user closed left
        # no trace anywhere.
        log.info("GPU setup finished: worked=%s (%s)", worked, detail)
        if self._quiet_timer is not None:
            self._quiet_timer.stop()
            self._quiet_timer = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._gpu_total = 0
        if self._cancel_button is not None:
            self._buttons.removeButton(self._cancel_button)
            self._cancel_button = None
        if worked:
            next_step = "" if self._first_run else " Restart PyWhispr to start using it."
            self._gpu_line.setText(f"GPU acceleration — ready and working ({detail}).{next_step}")
        else:
            self._gpu_line.setText(f"GPU acceleration — not enabled: {detail}")
        self._refresh_bar()
        self.setup_finished.emit(worked)
        self._close_when_idle()

    # -- shared --------------------------------------------------------------

    def _refresh_bar(self) -> None:
        total = self._model_total + self._gpu_total
        done = self._model_done + self._gpu_done
        if total <= 0:
            self._bar.hide()  # nothing is downloading: a bar would be theatre
            return
        self._bar.show()
        if done >= total:
            self._bar.setRange(0, 0)  # past the estimate: busy rather than a fake 99%
            return
        self._bar.setRange(0, 1000)
        self._bar.setValue(int(done / total * 1000))

    def _close_when_idle(self) -> None:
        """Nothing left to report: get out of the way, unless something failed."""
        if self._model_total or self.gpu_running:
            return
        failed = "not enabled" in self._gpu_line.text() or "could not" in self._model_line.text()
        if not failed:
            self.accept()
            return
        self._hide_button.hide()
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


__all__ = ["GPU_QUANTIZATION", "SetupWindow", "ask_to_enable"]
