"""The shared preprocessing must match the Ridge baseline exactly, so the comparison is fair."""
import sys

import numpy as np

from conftest import REPO
from speechprep import (N_BINS, WaveCache, istft, load_audio, log_mag, split_by_speaker, stft,
                        train_pairs)
from speechprep import test_pairs as valentini_test_pairs

sys.path.insert(0, str(REPO / "linear_baseline"))
from lrb import data as lrb_data  # noqa: E402
from lrb import features as lrb_features  # noqa: E402


def test_stft_matches_ridge_baseline():
    x = np.random.default_rng(1).standard_normal(16000 * 2 + 37).astype(np.float32)
    Z = stft(x)
    assert Z.shape[0] == N_BINS == 257
    np.testing.assert_array_equal(Z, lrb_features.stft(x, 512, 128).T)


def test_istft_round_trip_exact_length():
    x = np.random.default_rng(2).standard_normal(12345).astype(np.float32)
    y = istft(stft(x), len(x))
    assert len(y) == len(x)
    np.testing.assert_allclose(y, x, atol=1e-5)


def test_log_mag_matches_ridge_baseline():
    Z = stft(np.random.default_rng(3).standard_normal(4000))
    np.testing.assert_allclose(log_mag(Z), lrb_features.log_mag(Z), rtol=1e-6)


def test_audio_and_split_match_ridge_baseline(tiny_dataset):
    pairs = train_pairs(tiny_dataset, "28spk")
    assert [p[0] for p in pairs] == [p[0] for p in lrb_data.train_pairs(tiny_dataset, "28spk")]
    for seed in range(5):
        assert split_by_speaker(pairs, seed=seed, val_frac=0.25)[2] == \
            lrb_data.split_by_speaker(pairs, seed=seed, val_frac=0.25)[2]
    _, clean, _ = pairs[0]
    np.testing.assert_array_equal(load_audio(clean), lrb_data.load_audio(clean, 16000))


def test_split_is_speaker_disjoint_and_excludes_test(tiny_dataset):
    train, val, held = split_by_speaker(train_pairs(tiny_dataset, "28spk"), val_frac=0.25)
    train_spk = {s.split("_")[0] for s, _, _ in train}
    assert held and not train_spk & set(held)
    test_spk = {s.split("_")[0] for s, _, _ in valentini_test_pairs(tiny_dataset)}
    assert not test_spk & (train_spk | set(held))


def test_cache_is_lossless_and_reused(tiny_dataset, tmp_path):
    cache = WaveCache(tmp_path)
    _, clean, noisy = valentini_test_pairs(tiny_dataset)[0]
    assert cache.warm([clean, noisy], workers=2) == 2
    assert cache.warm([clean, noisy], workers=2) == 0
    np.testing.assert_array_equal(cache.load(clean), load_audio(clean))
    assert cache.path_for(clean).parent.name == "clean_testset_wav"
