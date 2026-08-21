"""The MSI rows that stop the running app before an upgrade replaces its files.

None of the MSI's behaviour can be exercised off Windows — `msilib` is
Windows-only, so cx_Freeze's `bdist_msi.finalize_options` refuses to run at all --
but the table rows are just data, and they encode decisions worth pinning: the
custom action type bits, the sequence numbers, and the condition.

`packaging/` is not a package and `setup_msi.py` calls `setup()` at import time,
so it is loaded by path with a stand-in cx_Freeze that records what it was given.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SETUP_MSI = ROOT / "packaging" / "setup_msi.py"

STOP = "PyWhisprStopRunning"
FORCE_STOP = "PyWhisprForceStopRunning"


@pytest.fixture
def msi(monkeypatch):
    """The kwargs setup_msi.py hands to cx_Freeze, plus its module.

    `monkeypatch.setitem`, not `patch.dict(sys.modules, ...)`: the latter restores
    by wiping the dict, so anything imported for real inside the block is deleted
    on the way out (see CLAUDE.md — it once tore pyobjc out of sys.modules and
    made later tests pass for the wrong reason).
    """
    captured = {}

    fake = SimpleNamespace(
        setup=lambda **kwargs: captured.update(kwargs),
        Executable=lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setitem(sys.modules, "cx_Freeze", fake)

    spec = importlib.util.spec_from_file_location("_pywhispr_setup_msi", SETUP_MSI)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, kwargs=captured)


@pytest.fixture
def data(msi):
    return msi.kwargs["options"]["bdist_msi"]["data"]


def _row(data, table, name):
    return next(row for row in data[table] if row[0] == name)


class TestStopTheRunningApp:
    def test_both_actions_are_authored(self, data):
        names = [row[0] for row in data["CustomAction"]]
        assert STOP in names and FORCE_STOP in names

    def test_the_graceful_action_runs_the_installed_exe(self, data):
        _name, type_bits, source, target = _row(data, "CustomAction", STOP)
        # 34 = exe with a working directory; +64 = ignore the exit code, so a
        # custom action can never fail an install. Deliberately *not* +192, which
        # is what cx_Freeze's own launch-on-finish uses to avoid waiting.
        assert type_bits == 34 + 64
        assert source == "TARGETDIR"
        assert target == '"[TARGETDIR]PyWhispr.exe" quit'

    def test_the_target_is_quoted(self, data):
        """It resolves under %LOCALAPPDATA%, and a username may contain spaces."""
        target = _row(data, "CustomAction", STOP)[3]
        assert target.startswith('"[TARGETDIR]')

    def test_the_force_action_needs_nothing_of_the_old_build(self, data):
        """Every build before this one exits 2 on an unknown subcommand, so for one
        upgrade this is the only thing that works — and afterwards it is the cure
        for a wedged instance, or an old install somewhere other than TARGETDIR."""
        _name, type_bits, _source, target = _row(data, "CustomAction", FORCE_STOP)
        assert type_bits == 34 + 64
        assert "taskkill.exe" in target
        assert "PyWhispr.exe" in target

    def test_every_source_is_a_directory_key_that_exists(self, data):
        """The one that shipped broken: type 34's Source is a *Directory table key*,
        not a property, and cx_Freeze authors that table from the build tree alone.
        `SystemFolder` has no row, so the install died at error 2727, "the directory
        entry does not exist in the Directory table". TARGETDIR is the only key we can
        name; a system path belongs in the command line, where the installer formats
        the property itself."""
        for name, type_bits, source, _target in data["CustomAction"]:
            if type_bits & 32:  # msidbCustomActionTypeDirectory
                assert source == "TARGETDIR", name

    def test_graceful_first(self, data):
        """The graceful path restores the audio mixer levels, which Windows
        remembers per app and a killed process would leave turned down."""
        order = {row[0]: row[2] for row in data["InstallExecuteSequence"]}
        assert order[STOP] < order[FORCE_STOP]

    def test_sequenced_after_costing_and_before_validation(self, data):
        for row in data["InstallExecuteSequence"]:
            # > CostFinalize (1000), or [TARGETDIR] is not resolved yet and would
            # format to nothing; < InstallValidate (1400), where files-in-use is
            # computed, and so also < the forced RemoveExistingProducts at 1450.
            assert 1000 < row[2] < 1400, row

    def test_it_does_not_collide_with_cx_freezes_own_rows(self, data):
        """cx_Freeze puts A_SET_TARGET_DIR at 401 and A_SET_REINSTALL_MODE at 402."""
        assert all(row[2] not in (401, 402) for row in data["InstallExecuteSequence"])

    def test_the_condition_covers_upgrade_and_uninstall_but_not_a_first_install(self, data):
        for name in (STOP, FORCE_STOP):
            condition = _row(data, "InstallExecuteSequence", name)[1]
            # Set by FindRelatedProducts from cx_Freeze's Upgrade rows...
            assert "REMOVEOLDVERSION" in condition
            assert "REMOVENEWVERSION" in condition
            # ...and Installed for repair, reinstall and uninstall, where the app
            # must also let go of the exe. None of them holds on a first install,
            # where [TARGETDIR]PyWhispr.exe does not exist yet.
            assert "Installed" in condition


class TestTheAutostartRowsSurvived:
    """The stop actions went into the same `data` dict, so they are easy to break."""

    def test_the_run_key_is_still_written(self, data):
        registry = _row(data, "Registry", "PyWhisprRunKey")
        assert registry[1] == 1  # HKCU, matching the per-user install
        assert registry[3] == "PyWhispr"

    def test_autostart_is_the_only_property_we_add(self, data):
        """cx_Freeze writes SecureCustomProperties itself, and msilib raises on a
        duplicate primary key at build time."""
        assert [row[0] for row in data["Property"]] == ["AUTOSTART"]

    def test_the_upgrade_code_is_stable(self, msi):
        options = msi.kwargs["options"]["bdist_msi"]
        assert options["upgrade_code"] == "{80F879CB-098B-413D-B82B-EA0CE82A6CB5}"
        assert options["all_users"] is False  # per-user: no admin rights needed
