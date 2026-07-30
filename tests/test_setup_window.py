from unittest.mock import MagicMock, patch

from pywhispr.ui import setup_window


def worker(monkeypatch, tmp_path, cache_mb=0, cuda_installed=False):
    """A worker whose byte counts come from values we control."""
    monkeypatch.setattr(setup_window.cuda, "install_dir", lambda: tmp_path / "cuda")
    monkeypatch.setattr(setup_window.cuda, "is_installed", lambda: cuda_installed)
    monkeypatch.setattr(setup_window, "cache_bytes", lambda: cache_mb * setup_window.MEGABYTE)
    return setup_window._Worker()


def window(qtbot, monkeypatch, cache_mb=0):
    monkeypatch.setattr(setup_window, "cache_bytes", lambda: cache_mb * setup_window.MEGABYTE)
    w = setup_window.SetupWindow()
    qtbot.addWidget(w)
    return w


class TestCountingBytes:
    def test_progress_counts_wheels_and_weights_together(self, monkeypatch, tmp_path):
        w = worker(monkeypatch, tmp_path, cache_mb=7)
        w._wheel_bytes = 3 * setup_window.MEGABYTE
        assert w._downloaded_mb() == 10

    def test_wheel_bytes_come_from_the_download_not_the_installed_files(
        self, monkeypatch, tmp_path
    ):
        """The extracted DLLs are half again bigger, so counting them overshot the total."""
        libraries = tmp_path / "cuda"
        libraries.mkdir()
        (libraries / "cudart64_13.dll").write_bytes(b"\x00" * 40 * setup_window.MEGABYTE)

        w = worker(monkeypatch, tmp_path)
        w._on_wheel_bytes(10 * setup_window.MEGABYTE)
        assert w._downloaded_mb() == 10

    def test_the_total_covers_both_downloads(self, monkeypatch, tmp_path):
        """Naming only the CUDA libraries would understate it by the larger half."""
        w = worker(monkeypatch, tmp_path)
        assert w._total_mb == (
            setup_window.cuda.APPROXIMATE_DOWNLOAD_MB + setup_window.APPROXIMATE_MODEL_MB
        )

    def test_the_total_drops_the_libraries_when_they_are_already_there(
        self, monkeypatch, tmp_path
    ):
        w = worker(monkeypatch, tmp_path, cuda_installed=True)
        assert w._total_mb == setup_window.APPROXIMATE_MODEL_MB


class TestOneBarOverEverything:
    def test_both_activities_share_one_bar(self, qtbot, monkeypatch):
        """A window each was two bars counting overlapping bytes."""
        w = window(qtbot, monkeypatch)
        w.track_model_download(expected_mb=100)
        w._on_gpu_progress(300, 900, "GPU acceleration")

        monkeypatch.setattr(setup_window, "cache_bytes", lambda: 50 * setup_window.MEGABYTE)
        w._poll_model()

        assert w._bar.value() == int((50 + 300) / (100 + 900) * 1000)
        assert "Speech model" in w._model_line.text()
        assert "GPU acceleration" in w._gpu_line.text()

    def test_a_line_only_appears_once_its_download_does(self, qtbot, monkeypatch):
        w = window(qtbot, monkeypatch)
        assert not w._model_line.isVisibleTo(w)
        assert not w._gpu_line.isVisibleTo(w)
        w.track_model_download(expected_mb=100)
        assert w._model_line.isVisibleTo(w)
        assert not w._gpu_line.isVisibleTo(w)

    def test_overshooting_the_estimate_goes_busy_rather_than_pretending(self, qtbot, monkeypatch):
        """It read "3903 of about 3650 MB" at 99%, which is two lies for one bug."""
        w = window(qtbot, monkeypatch)
        w._on_gpu_progress(4000, 3650, "GPU acceleration")
        assert w._bar.maximum() == 0  # indeterminate


class TestFinishing:
    def test_the_window_closes_when_nothing_is_left(self, qtbot, monkeypatch):
        w = window(qtbot, monkeypatch)
        w.track_model_download(expected_mb=100)
        w.finish_model()
        assert w.result() == w.DialogCode.Accepted

    def test_a_failed_load_stays_up_to_be_read(self, qtbot, monkeypatch):
        w = window(qtbot, monkeypatch)
        w.track_model_download(expected_mb=100)
        w.finish_model("The model could not be loaded.\n\nRuntimeError: offline")
        assert w._model_timer is None
        assert w.result() != w.DialogCode.Accepted

    def test_a_ready_model_does_not_close_a_running_gpu_setup(self, qtbot, monkeypatch):
        w = window(qtbot, monkeypatch)
        w.track_model_download(expected_mb=100)
        with patch.object(type(w), "gpu_running", property(lambda self: True)):
            w.finish_model()
        assert w.result() != w.DialogCode.Accepted


class TestWhatItPromises:
    """A first run has no model, so it cannot promise dictation carries on."""

    def _offer_text(self, first_run):
        with patch.object(setup_window, "QMessageBox") as message_box:
            box = message_box.return_value
            box.clickedButton.return_value = box.addButton.return_value
            setup_window.ask_to_enable(first_run=first_run)
        return box.setInformativeText.call_args.args[0]

    def test_the_offer_does_not_claim_dictation_works_during_a_first_run(self):
        text = self._offer_text(first_run=True)
        assert "keeps working" not in text
        assert "cannot start until it finishes" in text
        assert "Either way there is a one-time download" in text  # no is a download too

    def test_the_offer_says_dictation_continues_for_an_existing_install(self):
        assert "keeps working" in self._offer_text(first_run=False)

    def test_a_first_run_is_not_told_to_restart(self, qtbot, monkeypatch):
        """The model loads in this process straight afterwards, no session built yet."""
        w = window(qtbot, monkeypatch)
        w._first_run = True
        w._on_gpu_finished(True, "via CUDAExecutionProvider")
        assert "Restart" not in w._gpu_line.text()

    def test_an_existing_install_is_told_to_restart(self, qtbot, monkeypatch):
        """Its sessions are already built on the CPU and cannot be re-resolved."""
        w = window(qtbot, monkeypatch)
        w._first_run = False
        w._on_gpu_finished(True, "via CUDAExecutionProvider")
        assert "Restart" in w._gpu_line.text()

    def test_a_failed_setup_says_so_and_stays_up(self, qtbot, monkeypatch):
        w = window(qtbot, monkeypatch)
        w._on_gpu_finished(False, "the driver is too old")
        assert "not enabled" in w._gpu_line.text()
        assert w.result() != w.DialogCode.Accepted


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
        monkeypatch.setattr(setup_window.cuda, "is_installed", lambda: False)

        def download(progress, on_bytes=None):
            w.cancel()
            progress(0.5, "Downloading nvidia-cublas…")

        monkeypatch.setattr(setup_window.cuda, "download", download)
        with patch.object(setup_window.cuda, "start_verification") as start:
            w.run()

        start.assert_not_called()
        assert outcome == [(False, "Cancelled.")]


class TestStallDetection:
    def test_silence_rather_than_slowness_is_the_failure(self, monkeypatch, tmp_path):
        """A 2.4 GB download is slow but fine; no bytes at all is a hang."""
        w = worker(monkeypatch, tmp_path)
        process = MagicMock()
        process.poll.return_value = None  # never exits
        monkeypatch.setattr(setup_window.cuda, "start_verification", lambda q: process)
        monkeypatch.setattr(setup_window, "POLL_SECONDS", 0)
        monkeypatch.setattr(setup_window, "STALL_SECONDS", -1)  # already stalled

        worked, detail = w._verify()

        assert worked is False
        assert "progress" in detail
        process.kill.assert_called_once()

    def test_the_check_uses_full_precision(self, monkeypatch, tmp_path):
        """int8 has no CUDA kernels, so the GPU is only proven on the weights it runs."""
        w = worker(monkeypatch, tmp_path)
        process = MagicMock()
        process.poll.return_value = 0
        monkeypatch.setattr(
            setup_window.cuda, "start_verification", MagicMock(return_value=process)
        )
        monkeypatch.setattr(setup_window.cuda, "finish_verification", lambda p: (True, "via CUDA"))

        assert w._verify() == (True, "via CUDA")
        setup_window.cuda.start_verification.assert_called_once_with(setup_window.GPU_QUANTIZATION)
