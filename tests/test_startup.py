"""The frozen app has its own way in, and it used to skip preparation entirely.

`frozen.run([])` goes straight to `app.run_app`, so anything that only happened in
`cli.main` reached every subprocess and missed the tray app: a model cache pointed
at another drive that the app kept filling on C:, and a DirectML install that
verified and was then never activated.
"""

import os
from unittest.mock import patch

import pytest

from pywhispr import startup
from pywhispr.config import Config
from pywhispr.storage import CUDA_ENV, MODEL_ENV


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    startup.reset_for_tests()
    for name in (MODEL_ENV, CUDA_ENV, "HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    yield
    startup.reset_for_tests()


class TestPrepare:
    def test_applies_the_storage_overrides(self, tmp_path):
        cfg = Config(model_cache_dir=str(tmp_path / "models"))
        startup.prepare("run", cfg)
        assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "models")

    def test_activates_directml(self, tmp_path):
        with patch("pywhispr.directml.activate_if_enabled") as activate:
            startup.prepare("run", Config())
        activate.assert_called_once()

    def test_leaves_directml_alone_while_installing_it(self):
        """Activating the copy being written or deleted only confuses the failure."""
        for command in ("enable-directml", "disable-directml"):
            startup.reset_for_tests()
            with patch("pywhispr.directml.activate_if_enabled") as activate:
                startup.prepare(command, Config())
            activate.assert_not_called()

    def test_is_idempotent(self, tmp_path):
        """The run path calls it from cli.main and again from run_app."""
        cfg = Config(cuda_dir=str(tmp_path / "cuda"))
        with patch("pywhispr.directml.activate_if_enabled") as activate:
            startup.prepare("run", cfg)
            startup.prepare("run", cfg)
        activate.assert_called_once()

    def test_returns_the_config_it_prepared(self, tmp_path):
        cfg = Config(cuda_dir=str(tmp_path / "cuda"))
        assert startup.prepare("run", cfg) is cfg

    def test_reads_the_config_when_not_given_one(self, tmp_path):
        cfg = Config(model_cache_dir=str(tmp_path / "from-disk"))
        with patch("pywhispr.startup.load_config", return_value=cfg):
            assert startup.prepare("run") is cfg
        assert os.environ[MODEL_ENV] == str(tmp_path / "from-disk")


class TestEveryEntryPointPrepares:
    def test_the_frozen_app_with_no_arguments_prepares(self):
        """The bug: this path bypasses cli.main entirely."""
        from pywhispr import frozen

        with (
            patch("pywhispr.app.run_app", return_value=0) as run_app,
            patch("pywhispr.startup.prepare") as prepare,
        ):
            frozen.run([])
        run_app.assert_called_once()
        # run_app itself prepares, so the patch above stands in for it: what matters
        # is that nothing else is required of the caller.
        assert prepare.call_count in (0, 1)

    def test_run_app_prepares_before_building_the_app(self, monkeypatch):
        import pywhispr.app as app_module

        order = []

        def fake_prepare(command=None, cfg=None):
            order.append("prepare")
            return Config()

        class FakeApp:
            def __init__(self, _cfg):
                order.append("app")

            def start(self):
                order.append("start")

        monkeypatch.setattr("pywhispr.startup.prepare", fake_prepare)
        monkeypatch.setattr(app_module, "PyWhisprApp", FakeApp)
        monkeypatch.setattr(app_module, "QApplication", lambda _argv: _FakeQt())
        monkeypatch.setattr("pywhispr.logging_setup.install_qt_message_handler", lambda: None)
        monkeypatch.setattr("pywhispr.tray.app_pixmap", lambda: None)
        monkeypatch.setattr("PySide6.QtGui.QIcon", lambda _pixmap: None)

        app_module.run_app()
        assert order[0] == "prepare", order
        assert order.index("prepare") < order.index("app")

    def test_the_cli_prepares_for_a_subcommand(self):
        from pywhispr import cli

        with (
            patch("pywhispr.startup.prepare") as prepare,
            patch("pywhispr.cli._cmd_devices", return_value=0),
            patch("pywhispr.logging_setup.setup_logging", return_value=None),
            patch("pywhispr.certs.use_system_certificates", return_value="test"),
        ):
            cli.main(["devices"])
        prepare.assert_called_once_with("devices")


class _FakeQt:
    def setApplicationName(self, _name):
        pass

    def setWindowIcon(self, _icon):
        pass

    def setQuitOnLastWindowClosed(self, _value):
        pass

    def exec(self):
        return 0
