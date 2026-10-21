"""Valentini pair discovery and the speaker-disjoint train/validation split.

The split is the same function as linear_baseline/lrb/data.py, so with the same seed every model
holds out the same speakers. The test set (speakers p232, p257) is a separate folder and is
never part of training or model selection.
"""
from __future__ import annotations

import random
from pathlib import Path

TRAIN_SETS = {
    "28spk": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"),
    "56spk": ("clean_trainset_56spk_wav", "noisy_trainset_56spk_wav"),
}
TEST_SET = ("clean_testset_wav", "noisy_testset_wav")


def list_pairs(root, clean_dir: str, noisy_dir: str) -> list[tuple[str, Path, Path]]:
    """Sorted (stem, clean_path, noisy_path) for stems present in both folders."""
    clean_root, noisy_root = Path(root) / clean_dir, Path(root) / noisy_dir
    if not clean_root.is_dir() or not noisy_root.is_dir():
        raise FileNotFoundError(f"Missing {clean_root} or {noisy_root}")

    def stems(d):
        return {p.stem: p for p in d.glob("*.wav") if not p.name.startswith("._")}

    clean, noisy = stems(clean_root), stems(noisy_root)
    return [(s, clean[s], noisy[s]) for s in sorted(clean.keys() & noisy.keys())]


def train_pairs(root, train_set: str) -> list[tuple[str, Path, Path]]:
    """`train_set` is "28spk", "56spk" or "both"."""
    names = list(TRAIN_SETS) if train_set == "both" else [train_set]
    pairs = []
    for name in names:
        pairs.extend(list_pairs(root, *TRAIN_SETS[name]))
    return pairs


def test_pairs(root) -> list[tuple[str, Path, Path]]:
    return list_pairs(root, *TEST_SET)


def speaker(stem: str) -> str:
    return stem.split("_")[0]


def split_by_speaker(pairs, val_speakers=None, val_frac: float = 0.1, seed: int = 0):
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
