"""Disk cache of 16 kHz waveforms.

Reading a 48 kHz wav and resampling it is the slow part of loading. The STFT is cheap. So the cache
stores the resampled waveform (float32 .npy, lossless), not spectrograms. That is ~4x less disk
than caching magnitude + phase, and the noisy phase is recomputed exactly from the cached
waveform at inference.

Layout: <cache_root>/sr16000/<dataset folder>/<stem>.npy, e.g. sr16000/noisy_testset_wav/p232_001.npy.
Writes are atomic (temp file + rename), so parallel DataLoader workers can fill the cache safely.
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .audio import SR, load_audio


class WaveCache:
    def __init__(self, root, sr: int = SR):
        self.root = Path(root) / f"sr{sr}"
        self.sr = sr

    def path_for(self, wav_path) -> Path:
        wav_path = Path(wav_path)
        return self.root / wav_path.parent.name / f"{wav_path.stem}.npy"

    def load(self, wav_path) -> np.ndarray:
        """The cached 16 kHz waveform, created from `wav_path` on first use."""
        p = self.path_for(wav_path)
        try:
            return np.load(p)
        except (FileNotFoundError, ValueError, EOFError):
            pass
        x = load_audio(wav_path, self.sr)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, x)
        os.replace(tmp, p)
        return x

    def warm(self, wav_paths, workers: int | None = None) -> int:
        """Fill the cache for every path not cached yet (in parallel). Returns how many were added."""
        todo = [w for w in wav_paths if not self.path_for(w).exists()]
        if todo:
            with Pool(workers or os.cpu_count()) as pool:
                for _ in pool.imap_unordered(self.load, todo, chunksize=16):
                    pass
        return len(todo)
