from pywhispr.config import Config, load_config, save_config


def test_first_run_creates_default_file(tmp_path):
    path = tmp_path / "config.toml"
    cfg = load_config(path)
    assert path.exists()
    assert cfg == Config()


def test_round_trip(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config(
        hotkey="<ctrl>+<shift>+d",
        input_device=3,
        model_override="someone/some-model",
        max_recording_seconds=60,
        play_sounds=False,
        paste_delay_ms=200,
    )
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_none_values_survive_round_trip(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config(input_device=None, model_override=None)
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.input_device is None
    assert loaded.model_override is None


def test_unknown_keys_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('hotkey = "<f9>"\nfuture_option = true\n')
    cfg = load_config(path)
    assert cfg.hotkey == "<f9>"
    assert cfg.max_recording_seconds == Config().max_recording_seconds
