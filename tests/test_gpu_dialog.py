from unittest.mock import MagicMock, patch

from pywhispr.ui import gpu_dialog


def worker(monkeypatch, tmp_path, cache_mb=0):
    """A worker whose byte counts come from a directory we control."""
    monkeypatch.setattr(gpu_dialog.cuda, "install_dir", lambda: tmp_path / "cuda")
    monkeypatch.setattr(gpu_dialog, "cache_bytes", lambda: cache_mb * gpu_dialog.MEGABYTE)
    return gpu_dialog._Worker()


class TestOneBarForEverything:
    def test_the_total_covers_both_downloads(self):
        """Naming only the CUDA libraries would understate it by the larger half."""
        assert gpu_dialog.TOTAL_DOWNLOAD_MB == (
            gpu_dialog.cuda.APPROXIMATE_DOWNLOAD_MB + gpu_dialog.APPROXIMATE_MODEL_MB
        )

    def test_progress_counts_libraries_and_weights_together(self, monkeypatch, tmp_path):
        libraries = tmp_path / "cuda"
        libraries.mkdir()
        (libraries / "cudart64_13.dll").write_bytes(b"\x00" * 3 * gpu_dialog.MEGABYTE)

        w = worker(monkeypatch, tmp_path, cache_mb=7)
        assert w._downloaded_mb() == 10

    def test_weights_already_cached_are_not_counted_as_progress(self, monkeypatch, tmp_path):
        w = worker(monkeypatch, tmp_path, cache_mb=500)
        w._cache_at_start = 500 * gpu_dialog.MEGABYTE
        assert w._downloaded_mb() == 0

    def test_the_bar_never_reaches_the_end_before_the_check_does(self, monkeypatch, tmp_path):
        w = worker(monkeypatch, tmp_path)
        seen = []
        w.progress.connect(lambda fraction, message: seen.append((fraction, message)))
        w._emit(gpu_dialog.TOTAL_DOWNLOAD_MB * 2)
        fraction, message = seen[-1]
        assert fraction <= 0.99
        assert str(gpu_dialog.TOTAL_DOWNLOAD_MB) in message


class TestWhatItPromises:
    """A first run has no model, so it cannot promise dictation carries on."""

    def _offer_text(self, first_run):
        with patch.object(gpu_dialog, "QMessageBox") as message_box:
            box = message_box.return_value
            box.clickedButton.return_value = box.addButton.return_value
            gpu_dialog.ask_to_enable(first_run=first_run)
        return box.setInformativeText.call_args.args[0]

    def test_the_offer_does_not_claim_dictation_works_during_a_first_run(self):
        text = self._offer_text(first_run=True)
        assert "keeps working" not in text
        assert "cannot start until it finishes" in text
        assert "Either way there is a one-time download" in text  # no is a download too

    def test_the_offer_says_dictation_continues_for_an_existing_install(self):
        assert "keeps working" in self._offer_text(first_run=False)

    def test_a_first_run_is_not_told_to_restart(self, qtbot):
        """The model loads in this process straight afterwards, no session built yet."""
        dialog = gpu_dialog.GpuSetupDialog(first_run=True)
        qtbot.addWidget(dialog)
        dialog._on_finished(True, "via CUDAExecutionProvider")
        assert "Restart" not in dialog._status.text()

    def test_an_existing_install_is_told_to_restart(self, qtbot):
        """Its sessions are already built on the CPU and cannot be re-resolved."""
        dialog = gpu_dialog.GpuSetupDialog(first_run=False)
        qtbot.addWidget(dialog)
        dialog._on_finished(True, "via CUDAExecutionProvider")
        assert "Restart" in dialog._status.text()


class TestCancelling:
    def test_kills_the_check_instead_of_waiting_for_it(self, monkeypatch, tmp_path):
        """It used to wait out a multi-gigabyte download before giving up."""
        w = worker(monkeypatch, tmp_path)
        process = MagicMock()
        process.poll.return_value = None  # still running
        w._process = process

        w.cancel()

        process.kill.assert_called_once()

    def test_a_finished_check_is_not_killed(self, monkeypatch, tmp_path):
        w = worker(monkeypatch, tmp_path)
        process = MagicMock()
        process.poll.return_value = 0
        w._process = process

        w.cancel()

        process.kill.assert_not_called()

    def test_cancelling_during_the_wheels_stops_before_the_check(self, monkeypatch, tmp_path):
        w = worker(monkeypatch, tmp_path)
        outcome = []
        w.finished.connect(lambda worked, detail: outcome.append((worked, detail)))
        monkeypatch.setattr(gpu_dialog.cuda, "is_installed", lambda: False)

        def download(progress):
            w.cancel()
            progress(0.5, "Downloading nvidia-cublas…")

        monkeypatch.setattr(gpu_dialog.cuda, "download", download)
        with patch.object(gpu_dialog.cuda, "start_verification") as start:
            w.run()

        start.assert_not_called()
        assert outcome == [(False, "Cancelled.")]


class TestStallDetection:
    def test_silence_rather_than_slowness_is_the_failure(self, monkeypatch, tmp_path):
        """A 2.4 GB download is slow but fine; no bytes at all is a hang."""
        w = worker(monkeypatch, tmp_path)
        process = MagicMock()
        process.poll.return_value = None  # never exits
        monkeypatch.setattr(gpu_dialog.cuda, "start_verification", lambda q: process)
        monkeypatch.setattr(gpu_dialog, "POLL_SECONDS", 0)
        monkeypatch.setattr(gpu_dialog, "STALL_SECONDS", -1)  # already stalled

        worked, detail = w._verify()

        assert worked is False
        assert "progress" in detail
        process.kill.assert_called_once()

    def test_the_check_uses_full_precision(self, monkeypatch, tmp_path):
        """int8 has no CUDA kernels, so the GPU is only proven on the weights it runs."""
        w = worker(monkeypatch, tmp_path)
        process = MagicMock()
        process.poll.return_value = 0
        monkeypatch.setattr(gpu_dialog.cuda, "start_verification", MagicMock(return_value=process))
        monkeypatch.setattr(gpu_dialog.cuda, "finish_verification", lambda p: (True, "via CUDA"))

        assert w._verify() == (True, "via CUDA")
        gpu_dialog.cuda.start_verification.assert_called_once_with(gpu_dialog.GPU_QUANTIZATION)
