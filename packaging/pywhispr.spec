# PyInstaller spec for the macOS app bundle.
# Build with:  uv run --with pyinstaller pyinstaller packaging/pywhispr.spec
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("../src/pywhispr/assets", "pywhispr/assets")]
binaries = []
# truststore is imported lazily inside pywhispr.certs, so nothing in the graph
# points at it; without this the bundle silently loses the OS trust store and
# falls back to certifi (i.e. no model download behind TLS inspection). Collect
# the submodules too — the backends sit behind sys.platform conditionals.
hiddenimports = collect_submodules("truststore")

# mlx ships a compiled core + metal shader library; parakeet_mlx and librosa
# have data files and lazy imports that static analysis misses.
for pkg in ("mlx", "parakeet_mlx", "librosa"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# PyInstaller relocates the loaded libmlx.dylib to the Frameworks root, but
# libmlx locates mlx.metallib via dladdr in its own directory — so the
# metallib needs a copy at the root as well.
import importlib.util
import os

mlx_dir = importlib.util.find_spec("mlx").submodule_search_locations[0]
datas += [(os.path.join(mlx_dir, "lib", "mlx.metallib"), ".")]

a = Analysis(
    ["launch.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="PyWhispr",
    console=False,
    target_arch="arm64",
)

coll = COLLECT(exe, a.binaries, a.datas, name="PyWhispr")

app = BUNDLE(
    coll,
    name="PyWhispr.app",
    icon="PyWhispr.icns",
    bundle_identifier="com.benstaniford.pywhispr",
    info_plist={
        "LSUIElement": True,  # menu-bar app: no Dock icon, no app switcher entry
        "NSMicrophoneUsageDescription": "PyWhispr records your voice to transcribe it locally.",
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "0.1.0",
    },
)
