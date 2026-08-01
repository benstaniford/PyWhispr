from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from pywhispr.ui import storage_dialog


@pytest.fixture
def default_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_dialog, "default_model_dir", lambda: tmp_path)
    return tmp_path


class TestShouldAsk:
    def test_quiet_when_the_default_has_room(self, default_dir, monkeypatch):
        monkeypatch.setattr(storage_dialog, "free_mb", lambda _p: 500_000)
        assert storage_dialog.should_ask() is False

    def test_asks_when_the_default_is_short(self, default_dir, monkeypatch):
        monkeypatch.setattr(storage_dialog, "free_mb", lambda _p: 1_000)
        assert storage_dialog.should_ask() is True

    def test_quiet_when_free_space_is_unknown(self, default_dir, monkeypatch):
        """A question nobody needed is worse than a check that stayed quiet."""
        monkeypatch.setattr(storage_dialog, "free_mb", lambda _p: None)
        assert storage_dialog.should_ask() is False


class TestAsking:
    """The two dialogs are answered by replacing our own functions — patching
    QMessageBox.exec on the Qt class killed the interpreter (exit 127)."""

    @pytest.fixture(autouse=True)
    def seams(self, monkeypatch, default_dir):
        monkeypatch.setattr(storage_dialog, "free_mb", lambda _p: 1_000)

    def test_keeping_the_default_returns_none(self, monkeypatch):
        monkeypatch.setattr(storage_dialog, "wants_to_change", lambda *a, **k: False)
        monkeypatch.setattr(
            storage_dialog, "browse_for_directory", lambda *a: pytest.fail("must not browse")
        )
        assert storage_dialog.ask_where_to_store() is None

    def test_a_chosen_directory_gets_its_own_subfolder(self, monkeypatch, tmp_path):
        monkeypatch.setattr(storage_dialog, "wants_to_change", lambda *a, **k: True)
        monkeypatch.setattr(storage_dialog, "browse_for_directory", lambda *a: str(tmp_path))
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        assert storage_dialog.ask_where_to_store() == tmp_path / "PyWhispr"

    def test_cancelling_the_browser_keeps_the_default(self, monkeypatch):
        monkeypatch.setattr(storage_dialog, "wants_to_change", lambda *a, **k: True)
        monkeypatch.setattr(storage_dialog, "browse_for_directory", lambda *a: "")
        assert storage_dialog.ask_where_to_store() is None

    def test_a_tight_choice_is_warned_about_but_accepted(self, monkeypatch, tmp_path):
        """An external drive may be the only option they have."""
        monkeypatch.setattr(storage_dialog, "wants_to_change", lambda *a, **k: True)
        monkeypatch.setattr(storage_dialog, "browse_for_directory", lambda *a: str(tmp_path))
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        assert storage_dialog.ask_where_to_store() == tmp_path / "PyWhispr"
        assert warned, "a choice that is still too small has to say so"

    def test_no_warning_when_there_is_room(self, monkeypatch, tmp_path):
        monkeypatch.setattr(storage_dialog, "wants_to_change", lambda *a, **k: True)
        monkeypatch.setattr(storage_dialog, "browse_for_directory", lambda *a: str(tmp_path))
        monkeypatch.setattr(storage_dialog, "free_mb", lambda _p: 500_000)
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        assert storage_dialog.ask_where_to_store() == tmp_path / "PyWhispr"
        assert not warned

    def test_the_returned_path_is_absolute(self, monkeypatch, tmp_path):
        monkeypatch.setattr(storage_dialog, "wants_to_change", lambda *a, **k: True)
        monkeypatch.setattr(storage_dialog, "browse_for_directory", lambda *a: str(tmp_path))
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        assert Path(storage_dialog.ask_where_to_store()).is_absolute()
