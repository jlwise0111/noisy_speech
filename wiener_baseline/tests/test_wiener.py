import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wiener import WienerParams, wiener_filter, wiener_gain, wiener_oracle  # noqa: E402
from wiener.wiener import analyze, noise_psd_leading  # noqa: E402

SR = 16000


def snr_db(ref, est):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2))


@pytest.fixture
def pair():
    """Speech-like harmonic bursts after a 0.5 s pause, plus white noise at ~5 dB SNR."""
    rng = np.random.default_rng(0)
    t = np.arange(int(2.5 * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 8))
    bursts = ((t > 0.5) & (np.floor((t - 0.5) / 0.25) % 2 == 0)).astype(float)
    clean = 0.3 * tone * bursts / np.abs(tone).max()
    noise = rng.normal(size=t.size)
    noise *= np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2) / 10 ** (5 / 10))
    return clean, clean + noise


def test_output_length_matches_input(pair):
    _, noisy = pair
    for n in (len(noisy), len(noisy) - 37, 1000):
        assert len(wiener_filter(noisy[:n], SR)) == n


def test_gain_bounded(pair):
    _, noisy = pair
    p = WienerParams()
    Y = analyze(noisy, SR, p)
    G = wiener_gain(Y, noise_psd_leading(np.abs(Y) ** 2, SR, p), p)
    assert G.min() >= 10 ** (p.gain_floor_db / 20) - 1e-9
    assert G.max() <= 1.0


@pytest.mark.parametrize("method", ["leading", "minstats"])
def test_improves_snr(pair, method):
    clean, noisy = pair
    out = wiener_filter(noisy, SR, method)
    assert snr_db(clean, out) > snr_db(clean, noisy) + 3


def test_oracle_beats_blind(pair):
    clean, noisy = pair
    blind = wiener_filter(noisy, SR)
    oracle = wiener_oracle(noisy, clean, SR)
    assert snr_db(clean, oracle) > snr_db(clean, blind)


def test_silence_is_finite():
    out = wiener_filter(np.zeros(SR), SR)
    assert np.all(np.isfinite(out)) and np.all(out == 0)


def test_output_in_range(pair):
    _, noisy = pair
    out = wiener_filter(noisy * 5, SR)
    assert np.abs(out).max() <= 1.0
