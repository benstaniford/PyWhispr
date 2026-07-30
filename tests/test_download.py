from pywhispr import download


class TestCacheProbe:
    def test_counts_only_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
        (tmp_path / "models--x").mkdir()
        (tmp_path / "models--x" / "weights.onnx").write_bytes(b"\x00" * 2048)
        assert download.cache_bytes() == 2048

    def test_a_missing_cache_is_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path / "nope")
        assert download.cache_bytes() == 0

    def test_model_cached_is_a_size_test(self, tmp_path, monkeypatch):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
        (tmp_path / "weights.onnx").write_bytes(b"\x00" * 1024)
        assert download.model_cached(minimum_mb=1) is False
        (tmp_path / "big.onnx").write_bytes(b"\x00" * 2 * 1024 * 1024)
        assert download.model_cached(minimum_mb=1) is True


class TestProgressInTheWindow:
    """Progress comes from the cache growing, since the downloader has no callback."""

    def _window(self, tmp_path, monkeypatch, qtbot, expected_mb=10):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
        from pywhispr.ui.setup_window import SetupWindow

        window = SetupWindow()
        qtbot.addWidget(window)
        window.track_model_download(expected_mb)
        return window

    def test_tracks_growth_from_where_it_started(self, tmp_path, monkeypatch, qtbot):
        (tmp_path / "already-there").write_bytes(b"\x00" * 5 * 1024 * 1024)
        window = self._window(tmp_path, monkeypatch, qtbot)
        assert window._bar.value() == 0  # the 5 MB already cached is not progress

        (tmp_path / "new").write_bytes(b"\x00" * 5 * 1024 * 1024)
        window._poll_model()
        assert window._bar.value() == 500

    def test_overshooting_the_estimate_goes_indeterminate(self, tmp_path, monkeypatch, qtbot):
        window = self._window(tmp_path, monkeypatch, qtbot, expected_mb=1)
        (tmp_path / "new").write_bytes(b"\x00" * 3 * 1024 * 1024)
        window._poll_model()
        assert window._bar.maximum() == 0

    def test_failure_is_reported_rather_than_closing(self, tmp_path, monkeypatch, qtbot):
        window = self._window(tmp_path, monkeypatch, qtbot)
        window.finish_model("The model could not be loaded.\n\nRuntimeError: offline")
        assert window._model_timer is None
        assert "could not be loaded" in window._model_line.text()
        assert window.result() != window.DialogCode.Accepted  # stays up to be read
