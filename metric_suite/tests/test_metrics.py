import math

import numpy as np
import pytest

from msuite.metrics import pesq_wb, score_file, si_sdr, snr, stoi

SR = 16000


def speech_like(sr=SR, seconds=3.0, seed=0):
    """Harmonic 'voice' with a syllable envelope: enough for PESQ's voice-activity detection."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.5
    return (0.2 * voice * env + 0.002 * rng.normal(size=t.size)).astype(np.float32)


def test_snr_and_si_sdr_known_values():
    x = speech_like()
    noise = np.random.default_rng(1).normal(size=x.size).astype(np.float32)
    noise *= np.sqrt(np.sum(x**2) / np.sum(noise**2)) / 10 ** (10 / 20)  # exactly 10 dB SNR
    assert snr(x + noise, x) == pytest.approx(10.0, abs=1e-3)
    assert si_sdr(0.5 * x, x) > 100  # scale invariant
    assert snr(0.5 * x, x) == pytest.approx(6.02, abs=0.01)  # plain SNR is not


def test_pesq_wb_and_stoi_rank_noisier_lower():
    x = speech_like()
    rng = np.random.default_rng(2)
    light, heavy = x + 0.01 * rng.normal(size=x.size), x + 0.1 * rng.normal(size=x.size)
    assert pesq_wb(light, x, SR) > pesq_wb(heavy, x, SR)
    assert stoi(light, x, SR) > stoi(heavy, x, SR)
    assert 1.0 <= pesq_wb(heavy, x, SR) <= 4.65


def test_pesq_resamples_48k_like_16k():
    from msuite.metrics import resample
    x16 = speech_like(SR)
    n16 = x16 + 0.03 * np.random.default_rng(3).normal(size=x16.size)
    x48, n48 = resample(x16, SR, 48000), resample(n16, SR, 48000)
    assert pesq_wb(n48, x48, 48000) == pytest.approx(pesq_wb(n16, x16, SR), abs=0.05)


def test_silence_gives_nan_not_crash():
    z = np.zeros(SR * 2, dtype=np.float32)
    assert math.isnan(pesq_wb(z, z, SR))


def test_score_file_deltas():
    x = speech_like()
    rng = np.random.default_rng(4)
    noisy = x + 0.1 * rng.normal(size=x.size)
    s = score_file(x, noisy, x + 0.01 * rng.normal(size=x.size), SR)
    for m in ("pesq_wb", "stoi", "snr_db", "si_sdr_db"):
        assert s[f"{m}_delta"] == pytest.approx(s[f"{m}_processed"] - s[f"{m}_noisy"])
        assert s[f"{m}_delta"] > 0
