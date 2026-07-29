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
    "packages": [
        "pywhispr", "onnx_asr", "onnxruntime", "pynput", "_sounddevice_data", "truststore",
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
autostart_data = {
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

bdist_msi_options = {
    # Stable across releases so newer MSIs upgrade older installs in place.
    "upgrade_code": "{80F879CB-098B-413D-B82B-EA0CE82A6CB5}",
    "all_users": False,  # per-user install: no admin rights needed
    "install_icon": str(ROOT / "packaging" / "PyWhispr.ico"),
    "initial_target_dir": r"[LocalAppDataFolder]\Programs\PyWhispr",
    # Offer "launch on finish" on the last page. cx_Freeze hides the checkbox
    # when the MSI is being run to uninstall.
    "launch_on_finish": True,
    "data": autostart_data,
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
