"""Generate the start/stop sound cues in src/pywhispr/assets.

Run with: uv run python scripts/make_assets.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent.parent / "src" / "pywhispr" / "assets"
RATE = 44100


def blip(freqs: list[float], duration: float = 0.09, volume: float = 0.25) -> np.ndarray:
    """Short sine sweep through the given frequencies with a fade envelope."""
    chunks = []
    for freq in freqs:
        t = np.linspace(0, duration, int(RATE * duration), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        envelope = np.minimum(1.0, np.minimum(t / 0.01, (duration - t) / 0.02))
        chunks.append(tone * envelope)
    return (np.concatenate(chunks) * volume * 32767).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(samples.tobytes())
    print(f"wrote {path} ({len(samples) / RATE:.2f}s)")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    write_wav(ASSETS / "start.wav", blip([660.0, 880.0]))  # rising: recording started
    write_wav(ASSETS / "stop.wav", blip([880.0, 660.0]))  # falling: recording stopped


if __name__ == "__main__":
    main()
