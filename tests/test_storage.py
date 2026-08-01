from pathlib import Path

import pytest

from pywhispr import storage
from pywhispr.config import Config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """The overrides are environment variables, so a stray one would leak between tests."""
    for name in (storage.MODEL_ENV, storage.CUDA_ENV, "HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)


class TestResolution:
    def test_defaults_when_nothing_is_configured(self):
        cfg = Config()
        assert storage.model_dir(cfg) == storage.default_model_dir()
        assert storage.cuda_dir(cfg) == storage.default_cuda_dir()

    def test_config_is_used(self, tmp_path):
        cfg = Config(model_cache_dir=str(tmp_path / "m"), cuda_dir=str(tmp_path / "c"))
        assert storage.model_dir(cfg) == tmp_path / "m"
        assert storage.cuda_dir(cfg) == tmp_path / "c"

    def test_environment_beats_the_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv(storage.MODEL_ENV, str(tmp_path / "from-env"))
        cfg = Config(model_cache_dir=str(tmp_path / "from-config"))
        assert storage.model_dir(cfg) == tmp_path / "from-env"

    def test_a_home_relative_path_is_expanded(self):
        cfg = Config(cuda_dir="~/pywhispr-cuda")
        assert storage.cuda_dir(cfg) == Path.home() / "pywhispr-cuda"


class TestApplyOverrides:
    def test_points_hugging_face_at_the_configured_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "models"
        storage.apply_overrides(Config(model_cache_dir=str(target)))
        import os

        assert os.environ["HF_HUB_CACHE"] == str(target)
        # HF_HOME is the parent, so tokens and metadata follow the weights.
        assert os.environ["HF_HOME"] == str(tmp_path)

    def test_exports_for_the_helper_processes(self, tmp_path):
        import os

        storage.apply_overrides(Config(cuda_dir=str(tmp_path / "cuda")))
        assert os.environ[storage.CUDA_ENV] == str(tmp_path / "cuda")

    def test_nothing_is_set_when_nothing_is_configured(self):
        import os

        storage.apply_overrides(Config())
        assert "HF_HUB_CACHE" not in os.environ
        assert storage.MODEL_ENV not in os.environ

    def test_an_existing_environment_variable_is_left_alone(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv(storage.MODEL_ENV, str(tmp_path / "already"))
        storage.apply_overrides(Config(model_cache_dir=str(tmp_path / "config")))
        assert os.environ[storage.MODEL_ENV] == str(tmp_path / "already")
        assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "already")


class TestFreeSpace:
    def test_reports_free_space_for_an_existing_directory(self, tmp_path):
        assert storage.free_mb(tmp_path) is not None

    def test_walks_up_to_a_parent_that_exists(self, tmp_path):
        """A directory the user has only just named does not exist yet."""
        assert storage.free_mb(tmp_path / "not" / "yet" / "there") is not None

    def test_unknown_for_a_path_with_no_real_parent(self):
        assert storage.free_mb(Path("Q:/nothing/here")) is None


class TestBaseDir:
    def test_splits_one_choice_into_both_keys(self, tmp_path):
        cfg = Config()
        storage.set_base_dir(cfg, tmp_path)
        assert cfg.model_cache_dir == str(tmp_path / "models")
        assert cfg.cuda_dir == str(tmp_path / "cuda")

    def test_the_two_directories_are_separate(self, tmp_path):
        cfg = Config()
        storage.set_base_dir(cfg, tmp_path)
        assert cfg.model_cache_dir != cfg.cuda_dir


class TestConfigRoundTrip:
    def test_the_keys_survive_a_save_and_load(self, tmp_path):
        from pywhispr.config import load_config, save_config

        path = tmp_path / "config.toml"
        cfg = Config()
        storage.set_base_dir(cfg, tmp_path / "elsewhere")
        save_config(cfg, path)
        assert load_config(path).model_cache_dir == cfg.model_cache_dir
        assert load_config(path).cuda_dir == cfg.cuda_dir


class TestCudaUsesIt:
    def test_the_install_directory_follows_the_config(self, tmp_path, monkeypatch):
        from pywhispr import cuda

        monkeypatch.setenv(storage.CUDA_ENV, str(tmp_path / "libs"))
        assert cuda.install_dir() == tmp_path / "libs"

    def test_an_unreadable_config_falls_back_to_the_default(self, monkeypatch):
        from pywhispr import cuda

        def explode(*_args, **_kwargs):
            raise OSError("no config")

        monkeypatch.setattr("pywhispr.config.load_config", explode)
        assert cuda.install_dir() == storage.default_cuda_dir()
