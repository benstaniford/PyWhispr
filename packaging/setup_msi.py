"""cx_Freeze setup for the Windows MSI.

Build with:  uv run --with cx_Freeze python packaging/setup_msi.py bdist_msi
Run from the repository root; output lands in dist/.
"""

import sys
from pathlib import Path

from cx_Freeze import Executable, setup

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pywhispr import __version__  # noqa: E402

build_exe_options = {
    # pynput must be forced in whole: it loads its _win32 backend via a
    # string-built importlib call that cx_Freeze's scanner can't trace.
    # truststore is imported lazily inside pywhispr.certs, so the scanner never
    # sees it; leaving it out costs the OS trust store and, behind corporate TLS
    # inspection, the first-run model download.
    # email and importlib.metadata in whole because a half-frozen stdlib package is
    # worse than none: the exe records the build Python's stdlib directory, and where
    # that path exists at runtime, `email.__init__` came from library.zip while
    # `email.message` came from disk — "cannot import name 'header' from 'email'",
    # which killed `PyWhispr.exe verify-gpu`. The GUI only escaped it because
    # api.py imports email.parser early enough to win the race.
    "packages": [
        "pywhispr", "onnx_asr", "onnxruntime", "pynput", "_sounddevice_data", "truststore",
        "email", "importlib.metadata",
    ],
    # Keep pywhispr as real files so importlib.resources finds assets/.
    # _sounddevice_data must also stay unzipped: sounddevice dlopens the
    # portaudio DLL from that package's directory.
    "zip_include_packages": "*",
    "zip_exclude_packages": ["pywhispr", "onnx_asr", "onnxruntime", "numpy", "_sounddevice_data"],
    "include_msvcr": True,
}

# Run PyWhispr at logon. The MSI writes its own HKCU Run value rather than
# reusing the executable's component, because that component's id is generated
# by cx_Freeze internals; a component of our own keeps the row stable across
# cx_Freeze versions and lets the installer remove it again on uninstall.
#
# Root 1 is HKCU, matching the per-user install below. Attributes 4 marks the
# registry value as the component's key path (it owns no files), and the
# condition lets `msiexec /i PyWhispr.msi AUTOSTART=0` opt out.
AUTOSTART_COMPONENT = "PyWhisprAutoStart"
AUTOSTART_REGISTRY = "PyWhisprRunKey"
msi_data = {
    "Property": [("AUTOSTART", "1")],
    "Component": [
        (
            AUTOSTART_COMPONENT,
            "{6E36A2C5-4A28-4C1D-8B7A-1F0B1B6D4F21}",
            "TARGETDIR",
            4,
            'AUTOSTART<>"0"',
            AUTOSTART_REGISTRY,
        )
    ],
    "FeatureComponents": [("default", AUTOSTART_COMPONENT)],
    "Registry": [
        (
            AUTOSTART_REGISTRY,
            1,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            "PyWhispr",
            r'"[TARGETDIR]PyWhispr.exe"',  # quoted: a username may contain spaces
            AUTOSTART_COMPONENT,
        )
    ],
}

# Stop the running app before any file is replaced. Windows Installer otherwise
# reaches InstallValidate with PyWhispr.exe holding its own files open and falls
# back to the "files in use" dialog — which asks the user to close an app that
# has no window — or, in a quiet install, schedules a reboot.
#
# Type 34 is msidbCustomActionTypeExe|...Directory: Source is a Directory key and
# Target is the command line, run from that directory. cx_Freeze's own
# launch-on-finish action is 34+192, the 192 being "async, do not wait", so
# leaving it off is what makes these synchronous — and note that a synchronous
# custom action has no timeout of its own, which is why `quit` bounds itself.
# +64 is ...Continue: ignore the exit code, so neither of these can fail an
# install.
#
# **Source must be TARGETDIR for both, because it is the only Directory key that
# exists.** cx_Freeze authors the Directory table from the build tree and nothing
# else, so `SystemFolder` — a perfectly good *property*, and the obvious thing to
# put here — has no row, and Windows Installer fails the whole install with
# **error 2727, "the directory entry does not exist in the Directory table"**. That
# is what v0.2.23 shipped. `[SystemFolder]` is still right in the *command line*:
# Target is a formatted string and the installer sets that property itself during
# initialisation, and if it ever did not, taskkill.exe is on the PATH anyway.
#
# Both run at 1398/1399: after CostFinalize (1000) has resolved [TARGETDIR] —
# before that it would format to nothing — and before InstallValidate (1400)
# computes files-in-use and cx_Freeze's forced RemoveExistingProducts (1450).
# cx_Freeze's own rows are at 401 and 402, so nothing collides.
#
# The condition keeps them off a first install, where the exe does not exist yet.
# REMOVEOLDVERSION/REMOVENEWVERSION come from cx_Freeze's Upgrade rows (set by the
# standard FindRelatedProducts at 25); Installed covers repair, reinstall and
# uninstall, where the app must also let go of the exe.
#
# **Two actions, and the order is the point.** The graceful one runs the *old*
# build's exe, and any build before this one has no `quit` subcommand — argparse
# exits 2 and nothing happens. So for exactly one upgrade the taskkill is the only
# thing that works, and it stays afterwards as the cure for a wedged instance and
# for an old install that lives somewhere other than [TARGETDIR]. It is second so
# that the graceful path gets first refusal: it restores the audio mixer levels,
# which Windows remembers per app and a killed process would leave turned down.
STOP_ACTION = "PyWhisprStopRunning"
FORCE_STOP_ACTION = "PyWhisprForceStopRunning"
STOP_CONDITION = "REMOVEOLDVERSION OR REMOVENEWVERSION OR Installed"
TASKKILL = '"[SystemFolder]taskkill.exe" /F /IM PyWhispr.exe'
msi_data["CustomAction"] = [
    (STOP_ACTION, 34 + 64, "TARGETDIR", '"[TARGETDIR]PyWhispr.exe" quit'),
    (FORCE_STOP_ACTION, 34 + 64, "TARGETDIR", TASKKILL),
]
msi_data["InstallExecuteSequence"] = [
    (STOP_ACTION, STOP_CONDITION, 1398),
    (FORCE_STOP_ACTION, STOP_CONDITION, 1399),
]

bdist_msi_options = {
    # Stable across releases so newer MSIs upgrade older installs in place.
    "upgrade_code": "{80F879CB-098B-413D-B82B-EA0CE82A6CB5}",
    "all_users": False,  # per-user install: no admin rights needed
    "install_icon": str(ROOT / "packaging" / "PyWhispr.ico"),
    "initial_target_dir": r"[LocalAppDataFolder]\Programs\PyWhispr",
    # Offer "launch on finish" on the last page. cx_Freeze hides the checkbox
    # when the MSI is being run to uninstall.
    "launch_on_finish": True,
    "data": msi_data,
}

setup(
    name="PyWhispr",
    version=__version__,
    description="Local voice dictation with open source models",
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=[
        Executable(
            str(ROOT / "packaging" / "launch.py"),
            base="gui",  # no console window
            target_name="PyWhispr.exe",
            icon=str(ROOT / "packaging" / "PyWhispr.ico"),
            shortcut_name="PyWhispr",
            shortcut_dir="ProgramMenuFolder",
        )
    ],
)
