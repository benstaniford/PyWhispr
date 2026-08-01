import sys
import zipfile
from unittest.mock import patch

import pytest

from pywhispr import cuda, directml
from pywhispr.config import Config


@pytest.fixture(autouse=True)
def install_dir(tmp_path, monkeypatch):
    """Never touch a real DirectML install, and never leave one on sys.path."""
    target = tmp_path / "directml"
    monkeypatch.setattr(directml, "install_dir", lambda: target)
    original = list(sys.path)
    yield target
    sys.path[:] = original


def make_installed(target):
    package = target / "onnxruntime"
    (package / "capi").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (target / directml.MARKER).write_text("ok\n", encoding="utf-8")
    return target


class TestInstalled:
    def test_not_installed_when_the_directory_is_empty(self, install_dir):
        assert directml.is_installed() is False

    def test_installed_when_the_package_and_marker_are_there(self, install_dir):
        make_installed(install_dir)
        assert directml.is_installed() is True

    def test_a_missing_marker_means_a_half_extracted_download(self, install_dir):
        make_installed(install_dir)
        (install_dir / directml.MARKER).unlink()
        assert directml.is_installed() is False


class TestOffering:
    @pytest.fixture(autouse=True)
    def windows_with_a_gpu(self, monkeypatch):
        monkeypatch.setattr(directml, "has_direct3d_device", lambda: True)
        monkeypatch.setattr(cuda, "is_installed", lambda: False)

    def test_not_offered_when_cuda_would_work(self, monkeypatch):
        """CUDA is faster where it runs, and two onnxruntimes is a machine confused."""
        monkeypatch.setattr(cuda, "can_offer", lambda: (True, ""))
        with patch("sys.platform", "win32"):
            offer, why_not = directml.can_offer()
        assert offer is False
        assert "CUDA" in why_not

    def test_offered_when_the_gpu_is_too_old_for_cuda(self, monkeypatch):
        monkeypatch.setattr(cuda, "can_offer", lambda: (False, "compute capability 6.1"))
        with patch("sys.platform", "win32"):
            offer, why_not = directml.can_offer()
        assert offer is True, why_not

    def test_offered_when_there_is_no_nvidia_gpu_at_all(self, monkeypatch):
        """An AMD or Intel GPU has no other acceleration available."""
        monkeypatch.setattr(cuda, "can_offer", lambda: (False, "no NVIDIA GPU"))
        with patch("sys.platform", "win32"):
            offer, _why = directml.can_offer()
        assert offer is True

    def test_not_offered_without_a_display_adapter(self, monkeypatch):
        monkeypatch.setattr(cuda, "can_offer", lambda: (False, "no NVIDIA GPU"))
        monkeypatch.setattr(directml, "has_direct3d_device", lambda: False)
        with patch("sys.platform", "win32"):
            offer, why_not = directml.can_offer()
        assert offer is False
        assert "DirectX" in why_not

    def test_not_offered_off_windows(self, monkeypatch):
        monkeypatch.setattr(cuda, "can_offer", lambda: (False, "no NVIDIA GPU"))
        with patch("sys.platform", "darwin"):
            offer, why_not = directml.can_offer()
        assert offer is False
        assert "Windows" in why_not

    def test_not_offered_when_already_installed(self, install_dir, monkeypatch):
        make_installed(install_dir)
        monkeypatch.setattr(cuda, "can_offer", lambda: (False, "no NVIDIA GPU"))
        with patch("sys.platform", "win32"):
            offer, why_not = directml.can_offer()
        assert offer is False
        assert "already" in why_not


class TestActivation:
    def test_does_nothing_when_not_installed(self, install_dir):
        assert directml.activate() is False
        assert str(install_dir) not in sys.path

    def test_puts_the_download_first_on_the_path(self, install_dir, monkeypatch):
        make_installed(install_dir)
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
        assert directml.activate() is True
        assert sys.path[0] == str(install_dir)

    def test_refuses_once_onnxruntime_is_imported(self, install_dir, monkeypatch):
        """A half-switched process is worse than a CPU one."""
        make_installed(install_dir)
        monkeypatch.setitem(sys.modules, "onnxruntime", type(sys)("onnxruntime"))
        assert directml.activate() is False
        assert str(install_dir) not in sys.path

    def test_already_active_is_not_a_failure(self, install_dir, monkeypatch):
        make_installed(install_dir)
        module = type(sys)("onnxruntime")
        module.__file__ = str(install_dir / "onnxruntime" / "__init__.py")
        monkeypatch.setitem(sys.modules, "onnxruntime", module)
        assert directml.is_active() is True
        assert directml.activate() is True


class TestActivateIfEnabled:
    def test_off_when_the_config_says_false(self, install_dir, monkeypatch):
        make_installed(install_dir)
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
        assert directml.activate_if_enabled(Config(use_directml=False)) is False
        assert str(install_dir) not in sys.path

    def test_on_by_default_once_it_is_installed(self, install_dir, monkeypatch):
        make_installed(install_dir)
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
        assert directml.activate_if_enabled(Config()) is True

    def test_nothing_to_do_when_it_was_never_downloaded(self, install_dir):
        assert directml.activate_if_enabled(Config(use_directml=True)) is False


class TestDownload:
    def test_unpacks_the_whole_package_and_writes_the_marker(self, install_dir, tmp_path, monkeypatch):
        wheel = tmp_path / "onnxruntime_directml-1.0-cp312-win_amd64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("onnxruntime/__init__.py", "x = 1\n")
            archive.writestr("onnxruntime/capi/onnxruntime.dll", b"\x00" * 8)

        class FakeResponse:
            headers = {"content-length": str(wheel.stat().st_size)}

            def raise_for_status(self):
                pass

            def iter_bytes(self, _size):
                yield wheel.read_bytes()

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def stream(self, _method, _url, **_kwargs):
                return FakeResponse()

        monkeypatch.setattr("httpx.Client", lambda **_kwargs: FakeClient())
        monkeypatch.setattr(cuda, "_wheel_url", lambda _client, _wheel: "https://example/x.whl")

        directml.download()
        assert (install_dir / "onnxruntime" / "capi" / "onnxruntime.dll").exists()
        assert directml.is_installed() is True

    def test_removing_it_leaves_nothing_behind(self, install_dir):
        make_installed(install_dir)
        directml.remove()
        assert install_dir.exists() is False
        assert directml.is_installed() is False


class TestCudaComputeGate:
    def test_a_pascal_card_is_refused_before_the_download(self, monkeypatch):
        """A GTX 1080 reports 6.1: CUDA 13 has no kernels for it however new the driver."""
        monkeypatch.setattr(cuda, "is_installed", lambda: False)
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 580.0)
        monkeypatch.setattr(cuda, "compute_capability", lambda: 6.1)
        with patch("sys.platform", "win32"):
            offer, why_not = cuda.can_offer()
        assert offer is False
        assert "6.1" in why_not

    def test_a_turing_card_is_still_offered(self, monkeypatch):
        monkeypatch.setattr(cuda, "is_installed", lambda: False)
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 580.0)
        monkeypatch.setattr(cuda, "compute_capability", lambda: 7.5)
        with patch("sys.platform", "win32"):
            offer, why_not = cuda.can_offer()
        assert offer is True, why_not

    def test_an_unknown_capability_is_not_treated_as_too_old(self, monkeypatch):
        """An older nvidia-smi has no compute_cap field; that is not a verdict."""
        monkeypatch.setattr(cuda, "is_installed", lambda: False)
        monkeypatch.setattr(cuda, "nvidia_driver_version", lambda: 580.0)
        monkeypatch.setattr(cuda, "compute_capability", lambda: None)
        with patch("sys.platform", "win32"):
            offer, _why = cuda.can_offer()
        assert offer is True
