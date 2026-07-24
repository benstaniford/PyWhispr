import io
import logging
import sys

import pytest

from pywhispr import logging_setup


@pytest.fixture
def unconfigured(monkeypatch):
    """setup_logging() is idempotent by design; reset it so tests can call it."""
    monkeypatch.setattr(logging_setup, "_configured", False)
    root = logging.getLogger()
    original = list(root.handlers)
    original_level = root.level
    original_hook = sys.excepthook
    yield
    root.handlers[:] = original
    root.setLevel(original_level)
    sys.excepthook = original_hook


def test_log_path_is_under_log_dir():
    assert logging_setup.log_path().parent == logging_setup.log_dir()
    assert logging_setup.stderr_path().parent == logging_setup.log_dir()


def test_setup_logging_writes_to_the_file(unconfigured, monkeypatch, tmp_path):
    monkeypatch.setattr(logging_setup, "log_dir", lambda: tmp_path)
    path = logging_setup.setup_logging(verbose=True)

    assert path == tmp_path / logging_setup.LOG_FILENAME
    logging.getLogger("pywhispr.test").debug("hello from the log")
    logging.shutdown()
    assert "hello from the log" in path.read_text()


def test_setup_logging_is_idempotent(unconfigured, monkeypatch, tmp_path):
    monkeypatch.setattr(logging_setup, "log_dir", lambda: tmp_path)
    logging_setup.setup_logging()
    before = len(logging.getLogger().handlers)
    logging_setup.setup_logging()
    assert len(logging.getLogger().handlers) == before


def test_setup_logging_survives_an_unwritable_log_dir(unconfigured, monkeypatch, tmp_path):
    blocked = tmp_path / "nope"
    blocked.write_text("this is a file, not a directory")
    monkeypatch.setattr(logging_setup, "log_dir", lambda: blocked)
    assert logging_setup.setup_logging() is None  # console-only, no exception


def test_thread_exceptions_are_logged(unconfigured, monkeypatch, tmp_path):
    import threading

    monkeypatch.setattr(logging_setup, "log_dir", lambda: tmp_path)
    path = logging_setup.setup_logging()

    def boom():
        raise ValueError("worker exploded")

    thread = threading.Thread(target=boom, name="test-worker")
    thread.start()
    thread.join()
    logging.shutdown()

    contents = path.read_text()
    assert "worker exploded" in contents
    assert "test-worker" in contents


class TestRedirectStdio:
    def test_noop_when_streams_exist(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        assert logging_setup.redirect_stdio_if_headless() is None

    def test_replaces_none_streams(self, monkeypatch, tmp_path):
        monkeypatch.setattr(logging_setup, "log_dir", lambda: tmp_path)
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)

        path = logging_setup.redirect_stdio_if_headless()

        assert path == tmp_path / logging_setup.STDERR_FILENAME
        # The point of the exercise: writing must not raise.
        print("tqdm would write here")
        sys.stderr.write("and here\n")
        sys.stdout.flush()
        assert "tqdm would write here" in path.read_text()


class TestDebugEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_truthy(self, monkeypatch, value):
        monkeypatch.setenv(logging_setup.DEBUG_ENV_VAR, value)
        assert logging_setup.debug_enabled()

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_falsy(self, monkeypatch, value):
        monkeypatch.setenv(logging_setup.DEBUG_ENV_VAR, value)
        assert not logging_setup.debug_enabled()

    def test_unset(self, monkeypatch):
        monkeypatch.delenv(logging_setup.DEBUG_ENV_VAR, raising=False)
        assert not logging_setup.debug_enabled()


def test_environment_report_covers_the_basics():
    report = "\n".join(logging_setup.environment_report())
    for expected in ("PyWhispr", "python:", "platform:", "config:", "log file:", "onnxruntime"):
        assert expected in report
