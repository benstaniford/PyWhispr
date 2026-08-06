"""The stage timer must be free when off and report per-stage deltas when on."""

import logging

from pywhispr import perf


class TestEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(perf.ENV_VAR, raising=False)
        assert not perf.enabled()

    def test_falsey_values_are_off(self, monkeypatch):
        for value in ("", "0", "false", "no", " NO "):
            monkeypatch.setenv(perf.ENV_VAR, value)
            assert not perf.enabled(), value

    def test_anything_else_is_on(self, monkeypatch):
        monkeypatch.setenv(perf.ENV_VAR, "1")
        assert perf.enabled()


class TestCycle:
    def test_marks_are_dropped_when_off(self, monkeypatch, caplog):
        monkeypatch.delenv(perf.ENV_VAR, raising=False)
        perf.begin("cycle")
        perf.mark("something")
        with caplog.at_level(logging.INFO, logger="pywhispr.perf"):
            perf.end()
        assert caplog.records == []

    def test_logs_one_line_of_deltas(self, monkeypatch, caplog):
        monkeypatch.setenv(perf.ENV_VAR, "1")
        perf.begin("post-transcribe")
        perf.mark("clipboard-set")
        perf.mark("keystroke-sent")
        with caplog.at_level(logging.INFO, logger="pywhispr.perf"):
            perf.end()
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert message.startswith("post-transcribe: ")
        assert "clipboard-set=+" in message
        assert "keystroke-sent=+" in message
        assert "total" in message

    def test_end_is_idempotent(self, monkeypatch, caplog):
        monkeypatch.setenv(perf.ENV_VAR, "1")
        perf.begin("cycle")
        perf.mark("one")
        with caplog.at_level(logging.INFO, logger="pywhispr.perf"):
            perf.end()
            perf.end()  # _finish_cycle can be reached twice on the error paths
        assert len(caplog.records) == 1

    def test_mark_without_begin_is_harmless(self, monkeypatch):
        monkeypatch.setenv(perf.ENV_VAR, "1")
        perf.end()  # clear anything a previous test left behind
        perf.mark("stray")  # must not raise
