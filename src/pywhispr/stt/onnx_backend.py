"""Parakeet via ONNX Runtime (NVIDIA CUDA, with CPU fallback)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from pywhispr.stt.base import SAMPLE_RATE, STTBackend

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

CPU_ONLY = ["CPUExecutionProvider"]
PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Fewer threads than cores, by a lot. Parakeet TDT decodes autoregressively, so
# the graph is thousands of small ops and thread synchronisation dominates: on 15s
# of speech, 4 threads 0.43s, 8 threads 0.81s, 16 threads 1.96s. 0 restores
# onnxruntime's own default.
DEFAULT_THREADS = 4

# int8 is ~2x on the CPU and ~4x *slower* on the GPU (1.60s against 0.12s), where
# the quantised ops have no CUDA kernels. So it is chosen, not defaulted.
CPU_QUANTIZATION = "int8"


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
    ):
        self._model_id = model_id or DEFAULT_MODEL
        self._quantization = quantization
        self._threads = DEFAULT_THREADS if threads is None else threads
        self._model = None
        self._providers: list[str] = []

    @property
    def name(self) -> str:
        variant = f", {self._quantization}" if self._quantization else ""
        return f"onnx-asr ({self._model_id}{variant})"

    def load(self) -> None:
        import onnx_asr
        import onnxruntime

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
        if "CUDAExecutionProvider" not in providers:
            log.warning(
                "CUDAExecutionProvider not available (found: %s) — transcription will run "
                "on CPU. For an RTX GPU, install onnxruntime-gpu>=1.22 with a CUDA 12.8+ "
                "runtime and driver >=570.",
                ", ".join(advertised),
            )

        log.info(
            "Loading %s with providers %s, %s intra-op thread(s) (first run downloads ~600 MB)",
            self.name,
            providers,
            self._threads or "onnxruntime's default",
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
        if "CUDAExecutionProvider" in providers and not on_gpu:
            # The trap this whole dance exists for: onnxruntime accepts the
            # provider, drops it when the session is built, and says nothing, so
            # the app looks GPU-accelerated while every transcription is on the
            # CPU. Ask the sessions, not the list we passed in.
            log.warning(
                "CUDA was requested but the sessions run on %s — transcription is on the CPU. "
                "onnxruntime needs a full CUDA 13 + cuDNN 9 runtime; the pip wheels are "
                "nvidia-cuda-runtime, nvidia-cublas, nvidia-cudnn-cu13 and nvidia-cufft "
                "(the last from https://pypi.nvidia.com).",
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
        if self._threads:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self._threads
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
        text = self._model.recognize(audio.astype(np.float32), sample_rate=sample_rate)
        log.debug(
            "Transcribed %.1fs of audio in %.2fs",
            len(audio) / sample_rate,
            time.monotonic() - started,
        )
        return text.strip()
