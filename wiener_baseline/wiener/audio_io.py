"""Valentini dataset pairing and WAV I/O.

The dataset is only ever read from; outputs are written elsewhere.
"""
from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .wiener import SR  # Valentini is 48 kHz; the filter runs at 16 kHz (PESQ-WB's rate, 3x less work)

# split name -> (noisy folder, clean folder)
SPLITS = {
    "test": ("noisy_testset_wav", "clean_testset_wav"),
    "train28": ("noisy_trainset_28spk_wav", "clean_trainset_28spk_wav"),
    "train56": ("noisy_trainset_56spk_wav", "clean_trainset_56spk_wav"),
}


def _wavs(folder: Path) -> dict[str, Path]:
    return {
        f.stem: f
        for f in folder.iterdir()
        if f.suffix.lower() == ".wav" and not f.name.startswith("._")
    }


def find_pairs(dataset_root: Path, split: str) -> list[tuple[str, Path, Path]]:
    """Return sorted ``(stem, noisy_path, clean_path)`` for every matched file."""
    noisy_dir, clean_dir = (Path(dataset_root) / d for d in SPLITS[split])
    for d in (noisy_dir, clean_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"missing dataset folder: {d}")
    noisy, clean = _wavs(noisy_dir), _wavs(clean_dir)
    unmatched = noisy.keys() ^ clean.keys()
    if unmatched:
        print(f"warning: {len(unmatched)} unmatched file(s) skipped, "
              f"e.g. {sorted(unmatched)[:3]}", file=sys.stderr)
    return [(s, noisy[s], clean[s]) for s in sorted(noisy.keys() & clean.keys())]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV as float32 in [-1, 1], resampled to 16 kHz. Returns (signal, SR).

    Resampling is the same polyphase filter as speechprep.load_audio, so a waveform read here is
    identical to the one in the shared 16 kHz cache."""
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] != 1:
        raise ValueError(f"expected mono, got {x.shape[1]} channels: {path}")
    x = x[:, 0]
    if file_sr == SR:
        return x, SR
    g = gcd(SR, file_sr)
    return resample_poly(x, SR // g, file_sr // g).astype(np.float32), SR


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """Write mono 16-bit PCM, clipping to [-1, 1] first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")
