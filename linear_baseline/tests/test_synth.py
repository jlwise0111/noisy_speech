import numpy as np
import soundfile as sf

from lrb.ridge import Model
from lrb.synth import enhance, file_stats, write_triplet

SR, N_FFT, HOP = 16000, 512, 128


def identity_model():
    F = N_FFT // 2 + 1
    W = np.zeros((F, 9))
    W[:, 4] = 1.0
    return Model(W=W, b=np.zeros(F), alpha=np.zeros(F))


def test_identity_model_reconstructs_input():
    x = (0.1 * np.random.default_rng(0).normal(size=SR)).astype(np.float32)
    y = enhance(x, identity_model(), N_FFT, HOP)
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1e-4


def test_output_length_matches_odd_lengths():
    for n in (1001, 16000 + 37):
        x = np.random.default_rng(n).normal(size=n).astype(np.float32) * 0.1
        assert len(enhance(x, identity_model(), N_FFT, HOP)) == n


def test_write_triplet_matches_lengths(tmp_path):
    clean = np.zeros(1000, np.float32)
    write_triplet(tmp_path, "p000_001", clean, np.zeros(1003, np.float32), np.zeros(990, np.float32), SR)
    infos = [sf.info(str(tmp_path / r / "p000_001.wav")) for r in ("clean", "noisy", "enhanced")]
    assert {(i.samplerate, i.channels, i.frames) for i in infos} == {(SR, 1, 1000)}


def test_file_stats_from_48k_wavs(tmp_path):
    rng = np.random.default_rng(0)
    clean = 0.1 * rng.normal(size=48000)
    noisy = clean + 0.05 * rng.normal(size=48000)
    sf.write(str(tmp_path / "c.wav"), clean, 48000)
    sf.write(str(tmp_path / "n.wav"), noisy, 48000)
    s = file_stats(tmp_path / "c.wav", tmp_path / "n.wav", SR, N_FFT, HOP)
    assert s.sxx.shape == (N_FFT // 2 + 1, 9, 9)
    assert s.n[0] > 0
