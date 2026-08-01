"""User configuration: TOML file in the platform config directory."""

from __future__ import annotations

import dataclasses
import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w
from platformdirs import user_config_dir

log = logging.getLogger(__name__)

APP_NAME = "PyWhispr"

DEFAULT_HOTKEY_MAC = "<cmd>+<shift>+<space>"
DEFAULT_HOTKEY_OTHER = "<ctrl>+<alt>+<space>"


def default_hotkey() -> str:
    return DEFAULT_HOTKEY_MAC if sys.platform == "darwin" else DEFAULT_HOTKEY_OTHER


@dataclass
class Config:
    hotkey: str = dataclasses.field(default_factory=default_hotkey)
    input_device: int | None = None  # None = system default microphone
    model_override: str | None = None  # HuggingFace repo id; None = platform default
    # ONNX (Windows/Linux) only. None (the default) chooses: "int8" when the model
    # ends up on the CPU, where it is ~2x faster, and full precision on the GPU,
    # where int8 is *slower*. Set it explicitly ("int8" / "") to override.
    model_quantization: str | None = None
    # How many CPU threads a transcription may use. Fewer is faster here than
    # onnxruntime's default of one per core — see stt/onnx_backend.py. None uses
    # PyWhispr's tuned default; 0 restores onnxruntime's.
    stt_threads: int | None = None
    # Offer the one-time CUDA download when there is an NVIDIA GPU going unused.
    # Set false (or answer "Never") to stop asking; the tray menu still has it.
    offer_gpu_setup: bool = True
    # Where the gigabytes go. None keeps the platform defaults (the Hugging Face
    # cache and the app's data directory, both on the system drive); set either to
    # a path on another drive when C: has no room. See storage.py.
    model_cache_dir: str | None = None
    cuda_dir: str | None = None
    max_recording_seconds: int = 120
    play_sounds: bool = True
    paste_delay_ms: int = 150
    clipboard_restore_delay_ms: int = 300
    # Continuation joining: make consecutive dictations read as one passage.
    # Only ever adds a space or lower-cases the new text — never edits what is
    # already in the document.
    join_continuations: bool = True
    lowercase_continuations: bool = True
    context_chars: int = 64
    context_memory_seconds: int = 90
    # Filler removal: "um"/"uh"-style hesitations. extra_filler_words adds terms
    # of your own; keep_filler_words spares a built-in.
    remove_fillers: bool = True
    extra_filler_words: list[str] = dataclasses.field(default_factory=list)
    keep_filler_words: list[str] = dataclasses.field(default_factory=list)
    # Custom vocabulary: correct the transcript's spelling of terms listed in
    # vocabulary.txt (tray menu → Vocabulary…). vocabulary_fuzzy also fixes a
    # near miss on longer terms; turn it off to only ever match them exactly.
    vocabulary_enabled: bool = True
    vocabulary_fuzzy: bool = True
    # Network transcription API. Open to the LAN with no authentication:
    # set api_host to "127.0.0.1" to keep it on this machine only.
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 9149
    api_max_audio_seconds: int = 300
    api_max_queue: int = 4


def config_path() -> Path:
    return Path(user_config_dir(APP_NAME)) / "config.toml"


def load_config(path: Path | None = None) -> Config:
    """Load config, creating a default file on first run.

    Unknown keys are ignored so old configs survive upgrades.
    """
    path = path or config_path()
    if not path.exists():
        cfg = Config()
        save_config(cfg, path)
        log.info("Created default config at %s", path)
        return cfg

    with open(path, "rb") as f:
        data = tomllib.load(f)

    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - known
    if unknown:
        log.warning("Ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))
    return Config(**{k: v for k, v in data.items() if k in known})


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # TOML has no null: drop None values; load_config restores them as defaults.
    data = {k: v for k, v in dataclasses.asdict(cfg).items() if v is not None}
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
