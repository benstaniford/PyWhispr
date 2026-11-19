import pytest


@pytest.fixture(autouse=True)
def _never_the_real_config(tmp_path, monkeypatch):
    """Point the default config path into the test's own tmp_path.

    ``save_config(cfg)`` with no path writes the developer's real
    ``config.toml``, and plenty of app code reaches it that way — a settings
    change, a hotkey change, the GPU switches. One unpatched call in
    ``TestAudioDucking`` was quietly resetting the real settings to defaults on
    every ``pytest`` run, which looked exactly like "upgrading loses my
    settings", because the suite runs before every release. Tests that want a
    specific file still pass an explicit path; this only catches the default.
    """
    monkeypatch.setattr(
        "pywhispr.config.config_path", lambda: tmp_path / "pywhispr-config" / "config.toml"
    )
