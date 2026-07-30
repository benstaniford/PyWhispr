import subprocess
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from pywhispr import cuda


@pytest.fixture(autouse=True)
def install_dir(tmp_path, monkeypatch):
    """Never touch the developer's real CUDA install."""
    target = tmp_path / "cuda"
    monkeypatch.setattr(cuda, "install_dir", lambda: target)
    return target


def make_wheel(path, names=("nvidia/cu13/bin/x86_64/cublas64_13.dll", "nvidia/__init__.py")):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"\x00" * 16)
    return path


class TestOffering:
    def test_not_offered_without_an_nvidia_driver(self, monkeypatch):
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: None)
        with patch("sys.platform", "win32"):
            offer, why_not = cuda.can_offer()
        assert offer is False
        assert "NVIDIA" in why_not

    def test_not_offered_on_an_old_driver(self, monkeypatch):
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 550.1)
        with patch("sys.platform", "win32"):
            offer, why_not = cuda.can_offer()
        assert offer is False
        assert "550" in why_not and str(cuda.MINIMUM_DRIVER) in why_not

    def test_offered_on_a_current_driver(self, monkeypatch):
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 596.08)
        with patch("sys.platform", "win32"):
            assert cuda.can_offer() == (True, "")

    def test_not_offered_on_macos(self, monkeypatch):
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 596.08)
        with patch("sys.platform", "darwin"):
            offer, _ = cuda.can_offer()
        assert offer is False

    def test_not_offered_when_already_installed(self, monkeypatch, install_dir):
        install_dir.mkdir(parents=True)
        for dll in cuda.REQUIRED_DLLS:
            (install_dir / dll).touch()
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 596.08)
        with patch("sys.platform", "win32"):
            offer, why_not = cuda.can_offer()
        assert offer is False
        assert "already installed" in why_not

    def test_driver_version_parsing(self):
        result = MagicMock(returncode=0, stdout="596.08\n")
        with patch("subprocess.run", return_value=result):
            assert cuda.nvidia_driver_version() == 596.08

    def test_no_nvidia_smi_means_no_gpu(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert cuda.nvidia_driver_version() is None


class TestWheelUrls:
    def _client(self, body):
        client = MagicMock()
        client.get.return_value = MagicMock(text=body, raise_for_status=lambda: None)
        return client

    def test_picks_the_newest_matching_wheel(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        body = """
        <a href="nvidia_cublas-13.1.0-py3-none-win_amd64.whl#sha256=a">a</a>
        <a href="nvidia_cublas-13.6.0.2-py3-none-win_amd64.whl#sha256=b">b</a>
        <a href="nvidia_cublas-13.6.0.2-py3-none-manylinux2014_x86_64.whl">c</a>
        """
        url = cuda._wheel_url(self._client(body), cuda.Wheel("nvidia-cublas"))
        assert url.endswith("nvidia_cublas-13.6.0.2-py3-none-win_amd64.whl")
        assert url.startswith(cuda.PYPI)

    def test_absolute_urls_are_kept(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        body = '<a href="https://developer.download.nvidia.com/x/nvidia_cufft-12.3-win_amd64.whl">x</a>'
        url = cuda._wheel_url(self._client(body), cuda.Wheel("nvidia-cufft", cuda.NVIDIA))
        assert url == "https://developer.download.nvidia.com/x/nvidia_cufft-12.3-win_amd64.whl"

    def test_no_wheel_for_this_platform_is_an_error(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        with pytest.raises(RuntimeError, match="win_amd64"):
            cuda._wheel_url(self._client('<a href="thing-1.0-manylinux.whl">x</a>'), cuda.Wheel("x"))


class TestDownload:
    """The wheels are zips, so nothing needs pip — but the DLLs must land flat."""

    @pytest.fixture
    def fake_index(self, tmp_path, monkeypatch):
        wheel = make_wheel(tmp_path / "nvidia_thing-1.0-py3-none-win_amd64.whl")
        payload = wheel.read_bytes()

        class Response:
            headers = {"content-length": str(len(payload))}

            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                yield payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        client = MagicMock()
        client.__enter__ = lambda self: client
        client.__exit__ = lambda *_: False
        client.stream.return_value = Response()
        httpx = MagicMock()
        httpx.Client.return_value = client
        monkeypatch.setitem(__import__("sys").modules, "httpx", httpx)
        monkeypatch.setattr(cuda, "_wheel_url", lambda client, wheel: "https://x/y.whl")
        monkeypatch.setattr(cuda, "WHEELS", (cuda.Wheel("nvidia-thing"),))
        monkeypatch.setattr(cuda, "REQUIRED_DLLS", ("cublas64_13.dll",))
        monkeypatch.setattr("sys.platform", "win32")
        return client

    def test_extracts_the_libraries_flat(self, fake_index, install_dir):
        target = cuda.download()
        assert (target / "cublas64_13.dll").exists()
        assert not (target / "nvidia").exists()  # flattened, not nested
        assert (target / "READY").exists()
        assert cuda.is_installed()

    def test_progress_is_reported_and_cancellable(self, fake_index):
        seen = []
        with pytest.raises(KeyboardInterrupt):
            cuda.download(lambda fraction, message: seen.append((fraction, message)) or False)
        assert seen and all(0.0 <= fraction <= 1.0 for fraction, _ in seen)

    def test_a_missing_library_fails_loudly(self, fake_index, monkeypatch, install_dir):
        monkeypatch.setattr(cuda, "REQUIRED_DLLS", ("cudnn64_9.dll",))
        with pytest.raises(RuntimeError, match="missing"):
            cuda.download()
        assert not (install_dir / "READY").exists()

    def test_remove_deletes_everything(self, fake_index, install_dir):
        cuda.download()
        assert cuda.remove() is True
        assert not install_dir.exists()
        assert cuda.remove() is False


class TestVerify:
    """Installed is not working, and only working may be reported."""

    def test_reports_success_from_the_exit_code(self):
        result = MagicMock(returncode=0, stdout="transcription runs on the GPU via CUDA\n", stderr="")
        with patch("subprocess.run", return_value=result):
            assert cuda.verify() == (True, "transcription runs on the GPU via CUDA")

    def test_reports_failure_with_the_reason(self):
        result = MagicMock(returncode=1, stdout="runs on the CPU: no GPU provider\n", stderr="")
        with patch("subprocess.run", return_value=result):
            worked, detail = cuda.verify()
        assert worked is False
        assert "CPU" in detail

    def test_a_hung_check_is_a_failure(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
            worked, detail = cuda.verify()
        assert worked is False
        assert "in time" in detail

    def test_runs_a_fresh_process_of_this_program(self):
        assert cuda._self_command("verify-gpu")[-1] == "verify-gpu"
