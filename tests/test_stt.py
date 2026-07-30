import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pywhispr.config import Config
from pywhispr.stt import create_backend
from pywhispr.stt.wav import read_wav_mono_16k

FIXTURE = Path(__file__).parent / "fixtures" / "hello_world.wav"


def fake_model(*providers: str):
    """A stand-in for a loaded onnx_asr model whose sessions use `providers`.

    The sessions are what the backend interrogates, because the provider list
    handed to onnxruntime is not evidence that it kept any of them.
    """
    model = MagicMock()
    model._encoder.get_providers.return_value = list(providers)
    model._decoder_joint.get_providers.return_value = list(providers)
    return model


class TestBackendSelection:
    def test_apple_silicon_gets_mlx(self):
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            backend = create_backend(Config())
        assert "parakeet-mlx" in backend.name

    def test_windows_gets_onnx(self):
        with patch("sys.platform", "win32"):
            backend = create_backend(Config())
        assert "onnx-asr" in backend.name

    def test_model_override_is_passed_through(self):
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            backend = create_backend(Config(model_override="someone/custom-model"))
        assert "someone/custom-model" in backend.name


class TestOnnxProviderFallback:
    """onnxruntime-gpu advertises CUDAExecutionProvider even with no CUDA runtime,
    so a load can only be trusted once it has actually succeeded."""

    @pytest.fixture
    def modules(self, monkeypatch):
        onnxruntime = MagicMock()
        onnxruntime.__version__ = "1.22.0"
        onnxruntime.get_device.return_value = "GPU"
        onnxruntime.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        onnx_asr = MagicMock()
        # A loaded model reports the providers its sessions really use. Default to
        # the CPU: that is what onnxruntime does when the CUDA libraries are
        # missing, however cheerfully it accepted CUDAExecutionProvider.
        onnx_asr.load_model.return_value = fake_model("CPUExecutionProvider")
        monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
        monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)
        # Otherwise the real probe runs and the result depends on whether this
        # developer's machine has CUDA — which made these tests fail differently on
        # each run while enable-gpu was mid-download. Tests wanting it override.
        from pywhispr.stt import onnx_backend

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: False)
        return onnx_asr, onnxruntime

    def _backend(self):
        from pywhispr.stt.onnx_backend import OnnxBackend

        return OnnxBackend()

    def test_prefers_cuda_when_it_works(self, modules, monkeypatch):
        from pywhispr.stt import onnx_backend

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: True)
        onnx_asr, _ = modules
        onnx_asr.load_model.return_value = fake_model("CUDAExecutionProvider")
        backend = self._backend()
        backend.load()
        assert backend._providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert onnx_asr.load_model.call_count == 1  # no quantised reload on the GPU

    def test_warns_when_cuda_was_accepted_but_dropped(self, modules, caplog):
        """The silent fallback this whole dance exists for."""
        backend = self._backend()  # fixture's model runs on the CPU
        with caplog.at_level("WARNING"):
            backend.load()
        assert "CUDA was requested" in caplog.text

    def test_falls_back_to_cpu_when_cuda_session_fails(self, modules):
        onnx_asr, _ = modules
        sentinel = object()
        onnx_asr.load_model.side_effect = [RuntimeError("libcudart not found"), sentinel]

        backend = self._backend()
        backend.load()

        assert backend._providers == ["CPUExecutionProvider"]
        assert backend._model is sentinel
        assert onnx_asr.load_model.call_args_list[-1].kwargs["providers"] == [
            "CPUExecutionProvider"
        ]

    def test_quantization_is_passed_through(self, modules):
        """int8 is the one lever that speeds up the CPU path, so it must arrive."""
        from pywhispr.stt.onnx_backend import OnnxBackend

        onnx_asr, _ = modules
        backend = OnnxBackend(quantization="int8")
        assert "int8" in backend.name
        backend.load()
        assert onnx_asr.load_model.call_args.kwargs["quantization"] == "int8"

    def test_quantises_from_the_first_load_without_a_usable_cuda(self, modules, monkeypatch):
        """Deciding late would download full precision (2.4GB) and then int8 too."""
        from pywhispr.stt import onnx_backend
        from pywhispr.stt.onnx_backend import CPU_QUANTIZATION

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: False)
        onnx_asr, _ = modules
        backend = self._backend()
        backend.load()
        assert onnx_asr.load_model.call_args_list[0].kwargs["quantization"] == CPU_QUANTIZATION
        assert onnx_asr.load_model.call_count == 1  # nothing downloaded twice

    def test_quantises_late_if_cuda_looked_usable_but_was_not(self, modules, monkeypatch):
        """The libraries loaded, so full precision was the right bet — but the
        sessions still came back on the CPU, where int8 is worth the extra fetch."""
        from pywhispr.stt import onnx_backend
        from pywhispr.stt.onnx_backend import CPU_QUANTIZATION

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: True)
        onnx_asr, _ = modules  # fixture's model runs on the CPU
        backend = self._backend()
        backend.load()
        assert onnx_asr.load_model.call_args_list[0].kwargs["quantization"] is None
        assert onnx_asr.load_model.call_args_list[-1].kwargs["quantization"] == CPU_QUANTIZATION
        assert backend._quantization == CPU_QUANTIZATION

    def test_a_failed_quantised_reload_keeps_the_working_model(self, modules, monkeypatch):
        """The quantised weights are a separate download; no network, no problem."""
        from pywhispr.stt import onnx_backend

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: True)
        onnx_asr, _ = modules
        full_precision = fake_model("CPUExecutionProvider")
        onnx_asr.load_model.side_effect = [full_precision, RuntimeError("offline")]

        backend = self._backend()
        backend.load()

        assert backend._model is full_precision
        assert backend._quantization is None

    def test_intra_op_threads_are_capped_by_default(self, modules):
        """onnxruntime's one-thread-per-core default is several times slower."""
        from pywhispr.stt.onnx_backend import DEFAULT_THREADS, OnnxBackend

        onnx_asr, onnxruntime = modules
        OnnxBackend().load()
        options = onnx_asr.load_model.call_args.kwargs["sess_options"]
        assert options is onnxruntime.SessionOptions.return_value
        assert options.intra_op_num_threads == DEFAULT_THREADS

    def test_zero_threads_hands_the_choice_back_to_onnxruntime(self, modules):
        from pywhispr.stt.onnx_backend import OnnxBackend

        onnx_asr, _ = modules
        OnnxBackend(threads=0).load()
        assert onnx_asr.load_model.call_args.kwargs["sess_options"] is None

    def test_cuda_dll_directories_are_added_from_the_nvidia_wheels(self, tmp_path, monkeypatch):
        """The pip CUDA wheels hide their DLLs where nothing searches."""
        import pywhispr.cuda
        from pywhispr.stt import onnx_backend

        libs = tmp_path / "cu13" / "bin" / "x86_64"
        libs.mkdir(parents=True)
        (libs / "cublas64_13.dll").touch()
        (tmp_path / "empty" / "bin").mkdir(parents=True)
        spec = MagicMock()
        spec.submodule_search_locations = [str(tmp_path)]
        added = []
        monkeypatch.setattr(onnx_backend.sys, "platform", "win32")
        monkeypatch.setattr(onnx_backend.importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(onnx_backend.os, "add_dll_directory", added.append, raising=False)
        # Otherwise a developer who has run enable-gpu gets their own CUDA
        # directory in the result and the test fails on their machine only.
        monkeypatch.setattr(pywhispr.cuda, "install_dir", lambda: tmp_path / "not-installed")

        assert onnx_backend.add_cuda_dll_directories() == [str(libs)]
        assert added == [str(libs)]  # only the directory that has DLLs in it

    def test_cpu_only_failure_propagates(self, modules):
        onnx_asr, onnxruntime = modules
        onnxruntime.get_available_providers.return_value = ["CPUExecutionProvider"]
        onnx_asr.load_model.side_effect = RuntimeError("model download failed")

        with pytest.raises(RuntimeError, match="model download failed"):
            self._backend().load()
        assert onnx_asr.load_model.call_count == 1  # no pointless retry


class TestWavReading:
    def test_fixture_reads_as_mono_float32_16k(self):
        audio = read_wav_mono_16k(FIXTURE)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) > 16000  # fixture is longer than 1 second
        assert np.abs(audio).max() <= 1.0

    def test_resamples_non_16k(self, tmp_path):
        import wave

        path = tmp_path / "44k.wav"
        rate = 44100
        t = np.linspace(0, 1.0, rate, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(samples.tobytes())

        audio = read_wav_mono_16k(path)
        assert abs(len(audio) - 16000) <= 1


@pytest.mark.model
def test_real_model_transcribes_fixture():
    """Downloads and runs the actual Parakeet model. Run with: pytest -m model"""
    backend = create_backend(Config())
    backend.load()
    audio = read_wav_mono_16k(FIXTURE)
    text = backend.transcribe(audio).lower()
    assert "hello world" in text
    assert "transcription" in text


class TestVariantChosenBeforeDownloading:
    """The variants are separate downloads: fp32 2.4GB, int8 0.7GB. Loading the
    wrong one first and correcting afterwards costs the user both."""

    def test_no_usable_cuda_means_int8_before_any_download(self, monkeypatch):
        from pywhispr.stt import onnx_backend
        from pywhispr.stt.onnx_backend import CPU_QUANTIZATION, OnnxBackend

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: False)
        backend = OnnxBackend()
        backend.choose_quantization()
        assert backend._quantization == CPU_QUANTIZATION
        assert backend.download_mb < 1000

    def test_usable_cuda_keeps_full_precision(self, monkeypatch):
        from pywhispr.stt import onnx_backend
        from pywhispr.stt.onnx_backend import OnnxBackend

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: True)
        backend = OnnxBackend()
        backend.choose_quantization()
        assert backend._quantization is None
        assert backend.download_mb > 2000

    def test_an_explicit_choice_is_never_overridden(self, monkeypatch):
        from pywhispr.stt import onnx_backend
        from pywhispr.stt.onnx_backend import OnnxBackend

        monkeypatch.setattr(onnx_backend, "cuda_libraries_load", lambda: False)
        backend = OnnxBackend(quantization="")
        backend.choose_quantization()
        assert backend._quantization == ""

    def test_the_probe_needs_every_library(self, monkeypatch, tmp_path):
        import ctypes

        from pywhispr.stt import onnx_backend

        monkeypatch.setattr(onnx_backend.sys, "platform", "win32")
        monkeypatch.setattr(onnx_backend, "add_cuda_dll_directories", lambda: [])
        loaded = []

        def fake_windll(name):
            loaded.append(name)
            if name == onnx_backend.CUDA_PROBE_DLLS[-1]:
                raise OSError("missing")
            return object()

        monkeypatch.setattr(ctypes, "WinDLL", fake_windll, raising=False)
        assert onnx_backend.cuda_libraries_load() is False
        assert loaded  # it stopped at the missing one rather than assuming
