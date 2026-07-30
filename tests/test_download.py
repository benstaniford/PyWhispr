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


class TestDialog:
    """Progress comes from the cache growing, since the downloader has no callback."""

    def test_tracks_growth_from_where_it_started(self, tmp_path, monkeypatch, qtbot):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
        (tmp_path / "already-there").write_bytes(b"\x00" * 5 * 1024 * 1024)

        from pywhispr.ui.download_dialog import ModelDownloadDialog

        dialog = ModelDownloadDialog(expected_mb=10)
        qtbot.addWidget(dialog)
        assert dialog._bar.value() == 0  # the 5 MB already cached is not progress

        (tmp_path / "new").write_bytes(b"\x00" * 5 * 1024 * 1024)
        dialog._poll()
        assert dialog._bar.value() == 500

    def test_overshooting_the_estimate_goes_indeterminate(self, tmp_path, monkeypatch, qtbot):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
        from pywhispr.ui.download_dialog import ModelDownloadDialog

        dialog = ModelDownloadDialog(expected_mb=1)
        qtbot.addWidget(dialog)
        (tmp_path / "new").write_bytes(b"\x00" * 3 * 1024 * 1024)
        dialog._poll()
        assert dialog._bar.maximum() == 0

    def test_failure_is_reported_rather_than_closing(self, tmp_path, monkeypatch, qtbot):
        monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
        from pywhispr.ui.download_dialog import ModelDownloadDialog

        dialog = ModelDownloadDialog()
        qtbot.addWidget(dialog)
        dialog.finish("The model could not be loaded.\n\nRuntimeError: offline")
        assert not dialog._timer.isActive()
        assert "could not be loaded" in dialog._status.text()
