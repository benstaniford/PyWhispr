from unittest.mock import patch

import pytest

from pywhispr import cuda, directml, gpu
from pywhispr.config import Config


@pytest.fixture(autouse=True)
def install_dirs(tmp_path, monkeypatch):
    """Never look at — or delete — the developer's real installs."""
    monkeypatch.setattr(cuda, "install_dir", lambda: tmp_path / "cuda")
    monkeypatch.setattr(directml, "install_dir", lambda: tmp_path / "directml")


@pytest.fixture(autouse=True)
def never_write_the_real_config():
    """gpu.turn_off/turn_on save; unpatched that is the developer's config.toml."""
    with patch("pywhispr.gpu.save_config") as save:
        yield save


class TestSupported:
    def test_no_gpu_path_exists_on_macos(self):
        with patch("sys.platform", "darwin"):
            assert gpu.supported() is False

    def test_windows_could_have_one(self):
        with patch("sys.platform", "win32"):
            assert gpu.supported() is True

    def test_linux_could_have_one(self):
        with patch("sys.platform", "linux"):
            assert gpu.supported() is True


class TestInstalled:
    @pytest.mark.parametrize(
        "has_cuda, has_directml, expected",
        [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
    )
    def test_either_path_counts(self, has_cuda, has_directml, expected):
        with (
            patch("pywhispr.cuda.is_installed", return_value=has_cuda),
            patch("pywhispr.directml.is_installed", return_value=has_directml),
        ):
            assert gpu.installed() is expected


class TestActive:
    def test_installed_and_switched_on(self):
        with patch("pywhispr.gpu.installed", return_value=True):
            assert gpu.active(Config(use_gpu=True)) is True

    def test_switched_off_is_not_active_even_though_it_is_installed(self):
        """The whole point of the flag: the libraries stay, the acceleration stops."""
        with patch("pywhispr.gpu.installed", return_value=True):
            assert gpu.active(Config(use_gpu=False)) is False

    def test_nothing_installed_is_not_active(self):
        with patch("pywhispr.gpu.installed", return_value=False):
            assert gpu.active(Config(use_gpu=True)) is False


class TestTurningItOff:
    def test_the_flag_goes_off_and_is_saved(self, never_write_the_real_config):
        cfg = Config()
        gpu.turn_off(cfg)
        assert cfg.use_gpu is False
        never_write_the_real_config.assert_called_once_with(cfg)

    def test_the_offer_does_not_come_back_on_the_next_start(self):
        """Otherwise can_offer() goes True again and asks to install what was just off."""
        cfg = Config(offer_gpu_setup=True)
        gpu.turn_off(cfg)
        assert cfg.offer_gpu_setup is False

    def test_full_precision_weights_are_handed_back_to_the_cpu(self):
        """"" is what the first-run CUDA path wrote; on the CPU it is the slow choice."""
        cfg = Config(model_quantization="")
        gpu.turn_off(cfg)
        assert cfg.model_quantization is None

    def test_a_quantisation_the_user_chose_is_left_alone(self):
        cfg = Config(model_quantization="int8")
        gpu.turn_off(cfg)
        assert cfg.model_quantization == "int8"

    def test_the_directml_preference_is_left_alone(self):
        """It has to survive, or DirectML would not come back when the GPU does."""
        cfg = Config(use_directml=True)
        gpu.turn_off(cfg)
        assert cfg.use_directml is True

    def test_nothing_is_deleted(self, tmp_path):
        """This is the switch, not "pywhispr disable-gpu" — the download stays."""
        library = tmp_path / "cuda" / "cudart64_13.dll"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"\x00")
        gpu.turn_off(Config())
        assert library.exists()


class TestTurningItBackOn:
    def test_the_flag_goes_on_and_is_saved(self, never_write_the_real_config):
        cfg = Config(use_gpu=False)
        gpu.turn_on(cfg)
        assert cfg.use_gpu is True
        never_write_the_real_config.assert_called_once_with(cfg)
