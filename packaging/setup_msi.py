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
    # pynput must be forced in whole: it loads its _win32 backend via a
    # string-built importlib call that cx_Freeze's scanner can't trace.
    "packages": ["pywhispr", "onnx_asr", "onnxruntime", "pynput", "_sounddevice_data"],
    # Keep pywhispr as real files so importlib.resources finds assets/.
    # _sounddevice_data must also stay unzipped: sounddevice dlopens the
    # portaudio DLL from that package's directory.
    "zip_include_packages": "*",
    "zip_exclude_packages": ["pywhispr", "onnx_asr", "onnxruntime", "numpy", "_sounddevice_data"],
    "include_msvcr": True,
}

bdist_msi_options = {
    # Stable across releases so newer MSIs upgrade older installs in place.
    "upgrade_code": "{80F879CB-098B-413D-B82B-EA0CE82A6CB5}",
    "all_users": False,  # per-user install: no admin rights needed
    "install_icon": str(ROOT / "packaging" / "PyWhispr.ico"),
    "initial_target_dir": r"[LocalAppDataFolder]\Programs\PyWhispr",
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
