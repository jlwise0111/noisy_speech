"""Valentini dataset discovery, speaker splits and audio loading (read-only)."""

import random
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TRAIN_SETS = {
    "28spk": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"),
    "56spk": ("clean_trainset_56spk_wav", "noisy_trainset_56spk_wav"),
}
TEST_SET = ("clean_testset_wav", "noisy_testset_wav")


def list_pairs(root, clean_dir, noisy_dir):
    """Return sorted (stem, clean_path, noisy_path) for stems present in both folders."""
    clean_root, noisy_root = Path(root) / clean_dir, Path(root) / noisy_dir
    if not clean_root.is_dir() or not noisy_root.is_dir():
        raise FileNotFoundError(f"Missing {clean_root} or {noisy_root}")

    def stems(d):
        return {p.stem: p for p in d.glob("*.wav") if not p.name.startswith("._")}

    clean, noisy = stems(clean_root), stems(noisy_root)
    return [(s, clean[s], noisy[s]) for s in sorted(clean.keys() & noisy.keys())]


def train_pairs(root, train_set):
    names = list(TRAIN_SETS) if train_set == "both" else [train_set]
    pairs = []
    for name in names:
        pairs.extend(list_pairs(root, *TRAIN_SETS[name]))
    return pairs


def speaker(stem):
    return stem.split("_")[0]


def split_by_speaker(pairs, val_speakers=None, val_frac=0.1, seed=0):
    """Hold out whole speakers for validation. Returns (train, val, sorted val speaker ids)."""
    speakers = sorted({speaker(s) for s, _, _ in pairs})
    if val_speakers:
        held = set(val_speakers)
        unknown = held - set(speakers)
        if unknown:
            raise ValueError(f"Validation speakers not in training set: {sorted(unknown)}")
    else:
        n_val = max(1, round(len(speakers) * val_frac))
        held = set(random.Random(seed).sample(speakers, n_val))
    train = [p for p in pairs if speaker(p[0]) not in held]
    val = [p for p in pairs if speaker(p[0]) in held]
    return train, val, sorted(held)


def load_audio(path, sr):
    """Read a wav (read-only), downmix to mono, resample to `sr`. Returns float32."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x
