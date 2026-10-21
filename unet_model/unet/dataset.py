"""PyTorch datasets over cached 16 kHz Valentini pairs.

Each item is computed from the *whole* utterance (STFT, mask target, VAD whose threshold is
relative to the utterance peak) and only then cropped, so a crop's labels are the same as the
full file's labels at those frames.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from speechprep import N_BINS, WaveCache, log_mag, stft

from .targets import SILENCE_DB, mask_target, speech_frames


def utterance_features(clean: np.ndarray, noisy: np.ndarray, target: str, silence_db: float):
    """Full-utterance arrays: noisy log|Y| (F, T), target mask (F, T), speech frames (T,)."""
    n = min(len(clean), len(noisy))
    C, Y = stft(clean[:n]), stft(noisy[:n])
    return log_mag(Y), mask_target(C, Y, target), speech_frames(C, silence_db)


class PairDataset(Dataset):
    """Clean/noisy pairs -> dict(x=log|Y|, y=mask target, speech, valid, stem).

    crop_frames=None gives whole utterances (validation). Otherwise every item is a random crop of
    `crop_frames` frames; shorter utterances are zero-padded and the padding is marked valid=False.
    """

    def __init__(self, pairs, cache: WaveCache, crop_frames: int | None = 256,
                 target: str = "irm", silence_db: float = SILENCE_DB):
        self.pairs = list(pairs)
        self.cache = cache
        self.crop_frames = crop_frames
        self.target = target
        self.silence_db = silence_db

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        stem, clean_path, noisy_path = self.pairs[i]
        x, y, speech = utterance_features(self.cache.load(clean_path), self.cache.load(noisy_path),
                                          self.target, self.silence_db)
        valid = np.ones(x.shape[1], dtype=bool)
        if self.crop_frames is not None:
            x, y, speech, valid = _crop(x, y, speech, valid, self.crop_frames)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y),
                "speech": torch.from_numpy(speech), "valid": torch.from_numpy(valid), "stem": stem}


def _crop(x, y, speech, valid, n):
    t = x.shape[1]
    if t >= n:
        # torch's RNG is seeded per DataLoader worker, so crops are reproducible for a given seed.
        s = int(torch.randint(0, t - n + 1, ()).item())
        return x[:, s:s + n], y[:, s:s + n], speech[s:s + n], valid[s:s + n]
    pad = n - t
    return (np.pad(x, ((0, 0), (0, pad)), constant_values=x.min()), np.pad(y, ((0, 0), (0, pad))),
            np.pad(speech, (0, pad)), np.pad(valid, (0, pad)))


def input_stats(pairs, cache: WaveCache, max_files: int = 1000, seed: int = 0,
                silence_db: float = SILENCE_DB) -> dict:
    """Per-bin mean/std of noisy log|Y| and the silent-frame fraction, from training pairs only."""
    pairs = list(pairs)
    if len(pairs) > max_files:
        pairs = random.Random(seed).sample(pairs, max_files)
    s, s2, n_frames, n_silent = np.zeros(N_BINS), np.zeros(N_BINS), 0, 0
    for _, clean_path, noisy_path in pairs:
        x, _, speech = utterance_features(cache.load(clean_path), cache.load(noisy_path), "irm", silence_db)
        s += x.sum(axis=1)
        s2 += (x.astype(np.float64) ** 2).sum(axis=1)
        n_frames += x.shape[1]
        n_silent += int((~speech).sum())
    mean = s / n_frames
    std = np.sqrt(np.maximum(s2 / n_frames - mean ** 2, 0.0))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "silent_frac": n_silent / n_frames, "n_files": len(pairs), "n_frames": n_frames}
