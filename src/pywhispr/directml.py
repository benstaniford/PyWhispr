"""GPU acceleration for the cards CUDA 13 will not talk to.

CUDA 13 dropped everything older than Turing, and it only ever covered NVIDIA, so
a GTX 1080 and every AMD or Intel GPU are stuck on the CPU. DirectML runs on any
DirectX 12 device, which is all of them on Windows.

It cannot be bolted on the way the CUDA libraries are, though, and that shapes
everything here. The CUDA provider is *already compiled into* ``onnxruntime-gpu``;
:mod:`pywhispr.cuda` only supplies the DLLs it looks for at run time. The DirectML
provider is in a different build of onnxruntime — a package that installs under
the same name, ``onnxruntime``, so the two cannot be installed side by side.

So this fetches that wheel into a directory of its own and puts it *ahead* of the
frozen one on ``sys.path``, before anything imports onnxruntime. That is a real
import-order trick and it is treated as one: :func:`activate` refuses once
onnxruntime is loaded, everything is opt-in, and failing to activate leaves the
app on the CPU rather than broken.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

log = logging.getLogger(__name__)

PACKAGE = "onnxruntime-directml"
PYPI = "https://pypi.org/simple"

PROVIDER = "DmlExecutionProvider"

# The wheel, not what it unpacks to.
APPROXIMATE_DOWNLOAD_MB = 130

# Written last, so a half-extracted directory never counts as installed.
MARKER = "READY"

# Written when the download turns out not to load, so it is never activated again.
# Without it a broken install is retried every start, and the ImportError moves from
# the verification subprocess into the app's own model load — no dictation, no crash,
# no clue.
BROKEN_MARKER = "BROKEN"


def install_dir() -> Path:
    from pywhispr.cuda import _config_or_none
    from pywhispr.storage import directml_dir

    return directml_dir(_config_or_none())


def is_installed() -> bool:
    target = install_dir()
    if (target / BROKEN_MARKER).exists():
        return False
    return (target / MARKER).exists() and (target / "onnxruntime" / "__init__.py").exists()


def interpreter_tag() -> str:
    """The wheel tag this interpreter can load, e.g. "cp312".

    onnxruntime ships one wheel per Python version, and picking the newest by
    version number alone fetched a cp314 wheel for a cp312 app — which installs
    perfectly and then fails with "DLL load failed while importing
    onnxruntime_pybind11_state". The CUDA wheels never needed this: they are
    ``py3-none-win_amd64`` and carry no interpreter tag at all.
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def is_active() -> bool:
    """Is the DirectML build the onnxruntime this process would import?"""
    module = sys.modules.get("onnxruntime")
    if module is None:
        return False
    location = getattr(module, "__file__", "") or ""
    return str(install_dir()) in location


def can_offer() -> tuple[bool, str]:
    """(is DirectML worth offering, why not) for this machine.

    Offered as the *alternative* to CUDA, so anything CUDA can serve is left to
    CUDA: it is faster where it works, and having both installed is two copies of
    onnxruntime for one machine to get confused by.
    """
    if sys.platform != "win32":
        return False, "DirectML is Windows-only"
    if is_installed():
        return False, "DirectML is already installed"

    from pywhispr import cuda

    if cuda.is_installed():
        return False, "the CUDA libraries are already installed"
    cuda_offer, _why = cuda.can_offer()
    if cuda_offer:
        return False, "this GPU can use CUDA, which is faster"
    if not has_direct3d_device():
        return False, "no DirectX 12 GPU was found"
    return True, ""


def has_direct3d_device() -> bool:
    """Is there a GPU at all for DirectML to use?

    Deliberately coarse: WMI lists every display adapter including the Microsoft
    Basic Render Driver, and telling a DX12 device from a DX11 one properly means
    calling D3D12CreateDevice. A machine with no usable device fails at the
    verification step, which is the check that decides anything.
    """
    if sys.platform != "win32":
        return False
    try:
        import subprocess

        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, Exception):
        return False
    if result.returncode != 0:
        return False
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    real = [n for n in names if "Basic Render" not in n and "Basic Display" not in n]
    log.debug("Display adapters found: %d, of which usable: %d", len(names), len(real))
    return bool(real)


# -- fetching -----------------------------------------------------------------

Progress = Callable[[float, str], bool]
"""Called with (fraction done, what is happening); return False to cancel."""


def wheel_url(client) -> str:
    """The newest ``onnxruntime-directml`` wheel *this* interpreter can load.

    PEP 503 simple index, same as the CUDA wheels, but the filter is on the
    interpreter tag as well as the platform — see :func:`interpreter_tag`.
    """
    response = client.get(f"{PYPI}/{PACKAGE}/", follow_redirects=True)
    response.raise_for_status()
    tag = interpreter_tag()
    names = re.findall(r'href="([^"#]+)', response.text)
    usable = [
        url
        for url in names
        if url.endswith(".whl") and "win_amd64" in url and f"-{tag}-" in url
    ]
    if not usable:
        raise RuntimeError(
            f"No {PACKAGE} wheel for {tag} on win_amd64 — DirectML has no build for "
            f"Python {sys.version_info.major}.{sys.version_info.minor}"
        )
    url = sorted(usable, key=_version_key)[-1]
    log.info("Chose %s", url.rsplit("/", 1)[-1])
    return url if url.startswith("http") else f"{PYPI}/{PACKAGE}/{url}"


def _version_key(url: str) -> tuple[int, ...]:
    name = url.rsplit("/", 1)[-1]
    match = re.search(r"-(\d+(?:\.\d+)*)", name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)


def download(progress: Progress | None = None, on_bytes: Callable[[int], None] | None = None) -> Path:
    """Fetch the DirectML build of onnxruntime and unpack it into its own directory.

    The whole package, not just its DLLs — unlike the CUDA wheels, this *is* the
    onnxruntime that will be imported.
    """
    import httpx

    def report(fraction: float, message: str) -> None:
        if progress is not None and not progress(fraction, message):
            raise KeyboardInterrupt("cancelled")

    target = install_dir()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / MARKER
    marker.unlink(missing_ok=True)
    (target / BROKEN_MARKER).unlink(missing_ok=True)  # a fresh attempt gets a fresh verdict

    with httpx.Client(timeout=60.0) as client, TemporaryDirectory() as scratch:
        report(0.0, f"Finding {PACKAGE}…")
        url = wheel_url(client)
        archive_path = Path(scratch) / url.rsplit("/", 1)[-1]

        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with open(archive_path, "wb") as out:
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if on_bytes is not None:
                        on_bytes(done)
                    report(
                        (done / total * 0.9) if total else 0.0,
                        f"Downloading {PACKAGE} ({done >> 20} of {total >> 20} MB)…",
                    )

        report(0.9, f"Installing {PACKAGE}…")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)
        archive_path.unlink(missing_ok=True)

    if not (target / "onnxruntime" / "__init__.py").exists():
        raise RuntimeError(f"{PACKAGE} unpacked without an onnxruntime package in {target}")
    marker.write_text("ok\n", encoding="utf-8")
    log.info("DirectML onnxruntime installed in %s", target)
    return target


def remove() -> None:
    """Delete the download. The next start goes back to the packaged onnxruntime."""
    import shutil

    target = install_dir()
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        log.info("DirectML onnxruntime removed from %s", target)


# -- activation ---------------------------------------------------------------


def activate() -> bool:
    """Make ``import onnxruntime`` find the DirectML build. True if it will.

    Must run before anything imports onnxruntime, so it refuses afterwards rather
    than pretending: a module already in ``sys.modules`` is the one that gets used,
    and a half-switched process is worse than a CPU one.
    """
    if not is_installed():
        return False
    if "onnxruntime" in sys.modules:
        if is_active():
            return True
        log.warning("onnxruntime was imported before DirectML could be activated")
        return False

    target = install_dir()
    sys.path.insert(0, str(target))
    # The native libraries sit beside the extension module, and Python 3.8+ does not
    # search PATH for them.
    capi = target / "onnxruntime" / "capi"
    if capi.is_dir() and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(capi))
        except OSError:
            log.debug("Could not add %s as a DLL directory", capi, exc_info=True)

    # Import it here rather than leaving it to the model load. A wheel that does not
    # load is the failure this has actually had (a cp314 wheel under cp312), and it
    # surfaced deep inside onnx_asr as an ImportError with no way back. Proving it
    # now means one place to undo, and a marker so it is never tried again.
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        log.error("The DirectML onnxruntime in %s does not load: %s", target, exc)
        log.debug("DirectML import failure detail", exc_info=True)
        with contextlib.suppress(OSError):
            sys.path.remove(str(target))
        with contextlib.suppress(OSError):
            (target / BROKEN_MARKER).write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        sys.modules.pop("onnxruntime", None)  # a half-initialised module poisons the retry
        return False

    log.info("DirectML onnxruntime activated from %s", target)
    return True


def activate_if_enabled(cfg) -> bool:
    """Activate when the config says so, or when it is installed and set to auto.

    ``use_directml = None`` means "use it if it is there", which is what answering
    the offer leaves behind — the download would otherwise sit unused.
    """
    if cfg.use_directml is False:
        return False
    if not is_installed():
        return False
    return activate()


__all__ = [
    "APPROXIMATE_DOWNLOAD_MB",
    "PROVIDER",
    "activate",
    "activate_if_enabled",
    "can_offer",
    "download",
    "install_dir",
    "is_active",
    "is_installed",
    "remove",
]
