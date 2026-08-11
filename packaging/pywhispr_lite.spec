# PyInstaller spec for the PyWhisprLite macOS app bundle (Intel / x86_64).
#
# Lite talks to a remote transcription server instead of running a model, so it
# bundles none of the STT machinery the full spec collects (mlx/parakeet/librosa,
# the Metal shader library). That is also why it can target x86_64: there is no
# arm64-only inference library in the graph. Built for macOS 13 (Ventura) and up.
#
# Build with:  uv run --with pyinstaller pyinstaller packaging/pywhispr_lite.spec
from PyInstaller.utils.hooks import collect_submodules

datas = [("../src/pywhispr/assets", "pywhispr/assets")]
binaries = []
# truststore is imported lazily inside pywhispr.certs, so nothing in the graph
# points at it; without this the bundle silently loses the OS trust store and
# falls back to certifi (i.e. no https to a TLS-inspected server). Collect the
# submodules too — the backends sit behind sys.platform conditionals.
hiddenimports = collect_submodules("truststore")

# The version string, read from the source rather than hard-coded, so the bundle
# tracks the number make-release bumps.
import os
import re

version = "0.0.0"
with open(os.path.join(SPECPATH, "..", "src", "pywhispr", "__init__.py")) as f:
    match = re.search(r'^__version__ = "(.+)"', f.read(), re.MULTILINE)
    if match:
        version = match.group(1)

a = Analysis(
    ["launch.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Sets PYWHISPR_LITE=1 before any pywhispr import, so this bundle *is* Lite.
    runtime_hooks=["rthook_lite.py"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="PyWhisprLite",
    console=False,
    target_arch="x86_64",
)

coll = COLLECT(exe, a.binaries, a.datas, name="PyWhisprLite")

app = BUNDLE(
    coll,
    name="PyWhisprLite.app",
    icon="PyWhispr.icns",
    bundle_identifier="com.benstaniford.pywhisprlite",
    info_plist={
        "LSUIElement": True,  # menu-bar app: no Dock icon, no app switcher entry
        "NSMicrophoneUsageDescription": "PyWhisprLite records your voice to transcribe it.",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",  # Ventura floor, for Intel Macs
        "CFBundleShortVersionString": version,
    },
)
