import numpy as np

from pywhispr.audio import rms_level


def test_silence_is_zero():
    assert rms_level(np.zeros(1600, dtype=np.float32)) == 0.0


def test_full_scale_sine_is_near_one():
    t = np.arange(1600) / 16000
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert rms_level(sine) > 0.9


def test_quiet_speech_level_is_mid_range():
    t = np.arange(1600) / 16000
    sine = (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    level = rms_level(sine)
    assert 0.2 < level < 0.8


def test_level_is_monotonic_in_amplitude():
    t = np.arange(1600) / 16000
    base = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    levels = [rms_level(a * base) for a in (0.01, 0.05, 0.2, 0.8)]
    assert levels == sorted(levels)
    assert all(0.0 <= lv <= 1.0 for lv in levels)
