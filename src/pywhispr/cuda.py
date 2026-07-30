"""Optional GPU acceleration: fetch the CUDA libraries onnxruntime needs.

An NVIDIA GPU transcribes 15s of speech in 0.10s against 0.43s on the CPU, but
``onnxruntime-gpu`` ships none of the CUDA runtime and the libraries are ~1.2 GB —
too much for a 60 MB installer. So they are fetched on demand, like the model, and
only where there is an NVIDIA GPU to use them.

The libraries are pip wheels, which are zip files: this downloads them and
extracts the DLLs into the user's data directory. No pip, no admin rights, no CUDA
Toolkit, so it works from the packaged app.

Installed is not the same as working, and only working is reported: see
:func:`verify`.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from platformdirs import user_data_dir

from pywhispr.config import APP_NAME

log = logging.getLogger(__name__)

PYPI = "https://pypi.org/simple"
# PyPI still ships the CUDA 12 cuFFT (cufft64_11.dll); onnxruntime 1.27 wants
# cufft64_12.dll, which only NVIDIA's own index has.
NVIDIA = "https://pypi.nvidia.com"


@dataclass(frozen=True)
class Wheel:
    package: str
    index: str = PYPI


# What the CUDA provider loads, plus what cuDNN's runtime-compiled engines pull in.
# Small first, so an early failure costs seconds rather than a gigabyte.
WHEELS: tuple[Wheel, ...] = (
    Wheel("nvidia-cuda-runtime"),
    Wheel("nvidia-cuda-nvrtc"),
    Wheel("nvidia-nvjitlink", NVIDIA),
    Wheel("nvidia-cublas"),
    Wheel("nvidia-cufft", NVIDIA),
    Wheel("nvidia-cudnn-cu13"),
)

# The libraries whose absence makes the provider fail. Names only; verify()
# decides whether they work.
REQUIRED_DLLS = (
    "cudart64_13.dll",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cufft64_12.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
)

APPROXIMATE_DOWNLOAD_MB = 1200

# Checked before downloading: a gigabyte spent to be told "driver too old" is the
# worst of both.
MINIMUM_DRIVER = 580


def install_dir() -> Path:
    return Path(user_data_dir(APP_NAME)) / "cuda"


def nvidia_driver_version() -> float | None:
    """The installed NVIDIA driver version, or None if there is no NVIDIA GPU.

    ``nvidia-smi`` ships with the driver, so its absence is the answer.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"(\d+)\.(\d+)", result.stdout)
    return float(f"{match.group(1)}.{match.group(2)}") if match else None


def is_installed() -> bool:
    """Are the libraries already on disk? Says nothing about whether they work."""
    directory = install_dir()
    return all(any(directory.rglob(dll)) for dll in REQUIRED_DLLS)


def can_offer() -> tuple[bool, str]:
    """(is GPU acceleration worth offering, why not) for this machine."""
    if sys.platform not in ("win32", "linux"):
        return False, "GPU acceleration is only available on Windows and Linux"
    if is_installed():
        return False, "the CUDA libraries are already installed"
    driver = nvidia_driver_version()
    if driver is None:
        return False, "no NVIDIA GPU or driver was found"
    if driver < MINIMUM_DRIVER:
        return False, (
            f"the NVIDIA driver is {driver:g}; CUDA 13 needs {MINIMUM_DRIVER} or newer. "
            "Updating the driver and trying again will work."
        )
    return True, ""


# -- fetching -----------------------------------------------------------------

Progress = Callable[[float, str], bool]
"""Called with (fraction done, what is happening); return False to cancel."""


def _wheel_url(client, wheel: Wheel) -> str:
    """The newest Windows/Linux wheel for this package, from its index.

    PEP 503 simple index rather than the JSON API: NVIDIA's serves only the former.
    """
    response = client.get(f"{wheel.index}/{wheel.package}/", follow_redirects=True)
    response.raise_for_status()
    tag = "win_amd64" if sys.platform == "win32" else "manylinux"
    urls = [
        url
        for url in re.findall(r'href="([^"#]+)', response.text)
        if url.endswith(".whl") and tag in url
    ]
    if not urls:
        raise RuntimeError(f"No {tag} wheel for {wheel.package} at {wheel.index}")
    url = sorted(urls, key=_version_key)[-1]
    return url if url.startswith("http") else f"{wheel.index}/{wheel.package}/{url}"


def _version_key(url: str) -> tuple[int, ...]:
    name = url.rsplit("/", 1)[-1]
    match = re.search(r"-(\d+(?:\.\d+)*)", name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)


def _dll_members(archive: zipfile.ZipFile) -> Iterator[zipfile.ZipInfo]:
    for member in archive.infolist():
        if member.filename.lower().endswith((".dll", ".so")) or ".so." in member.filename:
            yield member


def download(progress: Progress | None = None, on_bytes: Callable[[int], None] | None = None) -> Path:
    """Fetch every wheel and extract its libraries into :func:`install_dir`.

    Raises on any failure: a partial install is worse than none.

    ``on_bytes`` receives the running total actually downloaded. A caller showing a
    byte count wants this rather than the size of what has been extracted: the DLLs
    are half again bigger than the wheels they came in, and they only appear at the
    end of each wheel, which makes a bar of six jumps that overshoots its total.
    """
    import httpx

    def report(fraction: float, message: str) -> None:
        if progress is not None and not progress(fraction, message):
            raise KeyboardInterrupt("cancelled")

    downloaded = 0  # across all wheels, so the caller sees one rising number

    target = install_dir()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "READY"  # written last, so a partial download never looks done
    marker.unlink(missing_ok=True)

    with httpx.Client(timeout=60.0) as client, TemporaryDirectory() as scratch:
        for index, wheel in enumerate(WHEELS):
            share = index / len(WHEELS)
            report(share, f"Finding {wheel.package}…")
            url = _wheel_url(client, wheel)
            archive_path = Path(scratch) / url.rsplit("/", 1)[-1]

            with client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                done = 0
                with open(archive_path, "wb") as out:
                    for chunk in response.iter_bytes(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        downloaded += len(chunk)
                        if on_bytes is not None:
                            on_bytes(downloaded)
                        within = done / total if total else 0.0
                        report(
                            share + within / len(WHEELS),
                            f"Downloading {wheel.package} ({done >> 20} of {total >> 20} MB)…",
                        )

            report(share + 0.9 / len(WHEELS), f"Installing {wheel.package}…")
            with zipfile.ZipFile(archive_path) as archive:
                for member in _dll_members(archive):
                    member.filename = Path(member.filename).name  # flatten
                    archive.extract(member, target)
            archive_path.unlink(missing_ok=True)

    missing = [dll for dll in REQUIRED_DLLS if not (target / dll).exists()]
    if missing:
        raise RuntimeError(f"Download finished but these libraries are missing: {missing}")
    marker.write_text("ok\n", encoding="utf-8")
    log.info("CUDA libraries installed in %s", target)
    return target


def remove() -> bool:
    """Delete the downloaded libraries. True if there was anything to delete."""
    from shutil import rmtree

    target = install_dir()
    if not target.exists():
        return False
    rmtree(target)
    log.info("Removed the CUDA libraries from %s", target)
    return True


# -- proving it works ---------------------------------------------------------


def _self_command(*arguments: str) -> list[str]:
    """This program plus arguments: packaged exe, or `python -m pywhispr`."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "pywhispr", *arguments]


# Long enough to build CUDA sessions on a cold cuDNN, short enough that a wedged
# check does not look like a hang. Nothing is downloaded under this timeout: the
# weights are fetched first, where the user can see the bytes arriving.
VERIFY_TIMEOUT = 240.0


def start_verification(quantization: str | None = None) -> subprocess.Popen:
    """Begin the check in a fresh process, returning it so it can be killed.

    Fresh on purpose: onnxruntime resolves providers once per process, so the app
    that just installed the libraries would keep reporting the CPU. Passing here
    means it will work on restart, which is what the user is told.

    ``quantization`` names the variant to check with. Left unset the check picks
    full precision, which is right for running but downloads 2.4 GB inside a step
    the user sees only as "checking…" — and the provider is what is under test,
    not the weights.
    """
    arguments = ["verify-gpu"]
    if quantization is not None:
        arguments += ["--quantization", quantization]
    return subprocess.Popen(
        _self_command(*arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def finish_verification(
    process: subprocess.Popen, timeout: float = VERIFY_TIMEOUT
) -> tuple[bool, str]:
    """Wait for :func:`start_verification` and turn its output into a verdict."""
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return False, "the check did not finish in time"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (out or err or "").strip().splitlines()
    return process.returncode == 0, detail[-1] if detail else "no output"


def verify(timeout: float = VERIFY_TIMEOUT, quantization: str | None = None) -> tuple[bool, str]:
    """Run the check to completion. See :func:`start_verification`."""
    try:
        process = start_verification(quantization)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return finish_verification(process, timeout)
