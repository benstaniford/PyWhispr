"""Parakeet via ONNX Runtime (NVIDIA CUDA, with CPU fallback)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from pywhispr import priority
from pywhispr.stt.base import SAMPLE_RATE, STTBackend

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

CPU_ONLY = ["CPUExecutionProvider"]
# CUDA first where it exists, then DirectML for the cards CUDA 13 will not serve.
# Only one of the two is ever advertised: they come from different builds of
# onnxruntime, and pywhispr.directml decides which one gets imported.
PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

# Fewer threads than cores, by a lot. Parakeet TDT decodes autoregressively, so
# the graph is thousands of small ops and thread synchronisation dominates: on 15s
# of speech, 4 threads 0.43s, 8 threads 0.81s, 16 threads 1.96s. 0 restores
# onnxruntime's own default.
DEFAULT_THREADS = 4

# Same reasoning, one level down: onnxruntime's intra-op pool spin-waits between
# ops, which wins on an idle machine and loses on a busy one — with thousands of
# micro-ops the spinners burn their quantum, get descheduled mid-graph, and every
# decode step pays a scheduler round-trip. On 18s of speech (CUDA, 16 cores, 15
# busy): spinning on 0.37s, off 0.25s. Idle it costs (0.27s against 0.42s), which
# the priority boost in transcribe() more than pays back (0.22s) — the pair is
# what was measured, and stt_allow_spinning = true undoes half of it.
SPINNING_OPTION = "session.intra_op.allow_spinning"

# int8 is ~2x on the CPU and ~4x *slower* on the GPU (1.60s against 0.12s), where
# the quantised ops have no CUDA kernels. So it is chosen, not defaulted.
CPU_QUANTIZATION = "int8"

# The variants are separate downloads, and picking wrong costs the user gigabytes:
# full precision is 2.4 GB, int8 is 0.65 GB. Hence cuda_libraries_load(), which
# decides *before* anything is fetched rather than after.
DOWNLOAD_MB = {None: 2450, "int8": 650}

# What the CUDA provider needs at load time. Loading them by name proves the
# provider will work without downloading a model to find out.
CUDA_PROBE_DLLS = ("cudart64_13.dll", "cublasLt64_13.dll", "cufft64_12.dll", "cudnn64_9.dll")


def add_cuda_dll_directories() -> list[str]:
    """Put the pip-installed CUDA/cuDNN DLLs on Windows' DLL search path.

    The wheels install them under ``site-packages/nvidia/<lib>/bin[/x86_64]``,
    which nothing searches, and ``onnxruntime.preload_dlls()`` knows the CUDA 12
    layout only. Without this the provider fails to create and everything runs on
    the CPU, with only a log warning to say so.
    """
    if sys.platform != "win32":
        return []  # ELF rpath handles this on Linux
    from pywhispr.cuda import install_dir

    spec = importlib.util.find_spec("nvidia")
    roots = list(spec.submodule_search_locations or ()) if spec is not None else []
    added = []
    # `pywhispr enable-gpu` flattens its DLLs into one directory; pip nests them.
    candidates = [install_dir()] if install_dir().is_dir() else []
    for root in roots:
        candidates.extend(sorted(Path(root).glob("*/bin*/**/")))
    for path in candidates:
        if any(path.glob("*.dll")):
            try:
                os.add_dll_directory(str(path))
            except OSError:  # vanished, or not a directory after all
                continue
            added.append(str(path))
    return added


def cuda_libraries_load() -> bool:
    """Can the CUDA libraries actually be loaded in this process?

    onnxruntime reports CUDAExecutionProvider as available whether or not they are
    installed, and the truth only comes out when a session is built — by which
    time a 2.4 GB model has been downloaded. Loading the DLLs by name is the same
    question asked for free.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    add_cuda_dll_directories()
    for name in CUDA_PROBE_DLLS:
        try:
            ctypes.WinDLL(name)
        except OSError:
            log.debug("CUDA probe: %s could not be loaded", name)
            return False
    return True


def directml_is_active() -> bool:
    """Is this process running the DirectML build of onnxruntime?

    Asked without importing onnxruntime: the answer decides which model variant to
    download, and that happens before the backend loads anything.
    """
    try:
        from pywhispr.directml import is_active

        return is_active()
    except Exception:  # pragma: no cover - directml is optional at runtime
        log.debug("Could not tell whether DirectML is active", exc_info=True)
        return False


def session_providers(model) -> set[str]:
    """Which execution providers the loaded model's sessions are *actually* using.

    onnxruntime accepts CUDAExecutionProvider, silently drops it when the CUDA
    libraries are missing, and reports back the list it was given — so this is the
    only honest answer. The sessions are private attributes of the adapter, hence
    the walk.
    """
    found: set[str] = set()
    seen: set[int] = set()

    def walk(obj, depth: int) -> None:
        if depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                attribute = getattr(obj, name)
            except Exception:  # properties can raise before the model is ready
                continue
            providers = getattr(attribute, "get_providers", None)
            if callable(providers):
                try:
                    found.update(providers())
                except Exception:
                    continue
            elif hasattr(attribute, "__dict__") and not callable(attribute):
                walk(attribute, depth + 1)  # the sessions hide inside the adapter

    walk(model, 0)
    return found


class OnnxBackend(STTBackend):
    def __init__(
        self,
        model_id: str | None = None,
        quantization: str | None = None,
        threads: int | None = None,
        allow_spinning: bool = False,
    ):
        self._model_id = model_id or DEFAULT_MODEL
        self._quantization = quantization
        self._threads = DEFAULT_THREADS if threads is None else threads
        self._allow_spinning = allow_spinning
        self._model = None
        self._providers: list[str] = []

    @property
    def name(self) -> str:
        variant = f", {self._quantization}" if self._quantization else ""
        return f"onnx-asr ({self._model_id}{variant})"

    @property
    def quantization(self) -> str | None:
        """The variant in use, once chosen: None is full precision."""
        return self._quantization

    @property
    def download_mb(self) -> int:
        """Roughly what a first run will fetch, for the progress window."""
        return DOWNLOAD_MB.get(self._quantization, DOWNLOAD_MB[None])

    def choose_quantization(self) -> None:
        """Pick the variant before it is downloaded, unless the user set one.

        Any GPU means full precision, not just CUDA. Measured on a GTX 1080 under
        DirectML: int8 transcribed 19.8s of speech in 1.3s, full precision did 22.2s
        in 0.8s — the same ~2x the other way round that CUDA shows, because the
        quantised ops have no GPU kernels either way and fall back to the CPU.
        """
        if self._quantization is not None:
            return
        if cuda_libraries_load():
            return
        if directml_is_active():
            return
        self._quantization = CPU_QUANTIZATION
        log.info("No usable GPU runtime: loading the quantised model (%s)", CPU_QUANTIZATION)

    def load(self) -> None:
        import onnx_asr
        import onnxruntime

        self.choose_quantization()
        try:
            found = add_cuda_dll_directories()
            log.debug("CUDA DLL directories added: %s", found or "none found")
            # Adding the directories is not enough on Windows: onnxruntime loads
            # the CUDA libraries by bare name from its own module directory, so
            # they have to be pulled into the process first. Both steps are
            # needed — the search path for the dependencies, this for the loads.
            if found and hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()
        except Exception:  # never let a GPU nicety stop the app loading
            log.debug("Could not preload the CUDA libraries", exc_info=True)

        advertised = onnxruntime.get_available_providers()
        log.info(
            "onnxruntime %s on %s; providers advertised: %s",
            getattr(onnxruntime, "__version__", "?"),
            onnxruntime.get_device(),
            ", ".join(advertised),
        )

        providers = [p for p in PREFERRED_PROVIDERS if p in advertised] or CPU_ONLY
        if not any(p != "CPUExecutionProvider" for p in providers):
            log.warning(
                "No GPU provider available (found: %s) — transcription will run on CPU. "
                "For an RTX GPU, install onnxruntime-gpu>=1.22 with a CUDA 12.8+ runtime "
                "and driver >=570; for anything older, or an AMD or Intel GPU, "
                "\"pywhispr enable-directml\".",
                ", ".join(advertised),
            )

        log.info(
            "Loading %s with providers %s, %s intra-op thread(s), spinning %s "
            "(a first run downloads about %d MB)",
            self.name,
            providers,
            self._threads or "onnxruntime's default",
            "on" if self._allow_spinning else "off",
            self.download_mb,
        )
        started = time.monotonic()
        try:
            self._load_with(onnx_asr, providers)
        except Exception as exc:
            # onnxruntime-gpu advertises CUDAExecutionProvider even when the
            # machine has no CUDA runtime; the failure only surfaces here, when
            # a session is created. Falling back keeps a GPU-less Windows box
            # working instead of leaving the app stuck on "Loading model…".
            if providers == CPU_ONLY:
                raise
            log.warning(
                "Loading with %s failed (%s: %s) — retrying on CPU only",
                providers,
                type(exc).__name__,
                exc,
            )
            log.debug("Provider failure detail", exc_info=True)
            self._load_with(onnx_asr, CPU_ONLY)

        in_use = session_providers(self._model)
        on_gpu = any(p != "CPUExecutionProvider" for p in in_use)
        wanted_gpu = [p for p in providers if p != "CPUExecutionProvider"]
        if wanted_gpu and not on_gpu:
            # The trap this whole dance exists for: onnxruntime accepts the
            # provider, drops it when the session is built, and says nothing, so
            # the app looks GPU-accelerated while every transcription is on the
            # CPU. Ask the sessions, not the list we passed in.
            log.warning(
                "%s was requested but the sessions run on %s — transcription is on the CPU. "
                "For CUDA, onnxruntime needs a full CUDA 13 + cuDNN 9 runtime: the pip wheels "
                "are nvidia-cuda-runtime, nvidia-cublas, nvidia-cudnn-cu13 and nvidia-cufft "
                "(the last from https://pypi.nvidia.com). For DirectML, the GPU has to support "
                "DirectX 12.",
                ", ".join(wanted_gpu),
                ", ".join(sorted(in_use)) or "the CPU",
            )

        if self._quantization is None and not on_gpu and CPU_QUANTIZATION:
            # Nothing accelerates this but quantisation now, and it is worth ~2x.
            log.info("Reloading as %s: the CPU path is much faster quantised", CPU_QUANTIZATION)
            self._quantization = CPU_QUANTIZATION
            try:
                self._load_with(onnx_asr, self._providers)
            except Exception:
                # No network on a first run, most likely: the quantised weights
                # are a separate download. The model we already have works.
                log.warning("Could not load the quantised model; keeping full precision")
                log.debug("Quantised load failure", exc_info=True)
                self._quantization = None

        log.info(
            "Loaded %s in %.1fs using %s",
            self.name,
            time.monotonic() - started,
            ", ".join(sorted(session_providers(self._model))) or self._providers,
        )

    def _load_with(self, onnx_asr, providers: list[str]) -> None:
        import onnxruntime

        options = None
        if self._threads or not self._allow_spinning:
            options = onnxruntime.SessionOptions()
            if self._threads:
                options.intra_op_num_threads = self._threads
            if not self._allow_spinning:
                options.add_session_config_entry(SPINNING_OPTION, "0")
        self._model = onnx_asr.load_model(
            self._model_id,
            providers=providers,
            quantization=self._quantization,
            sess_options=options,
        )
        self._providers = providers

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")
        started = time.monotonic()
        # Both callers (dictation and the network API) come through here on the
        # one STT worker, so this is the single place the boost belongs.
        with priority.boosted():
            text = self._model.recognize(audio.astype(np.float32), sample_rate=sample_rate)
        log.debug(
            "Transcribed %.1fs of audio in %.2fs",
            len(audio) / sample_rate,
            time.monotonic() - started,
        )
        return text.strip()
