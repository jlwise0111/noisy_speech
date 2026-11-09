"""Where the audio comes from: an uploaded file, or a train/test file of the Valentini dataset."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from . import DATASET_ROOT  # noqa: F401  (also sets sys.path for speechprep)
from speechprep import SR, TEST_SET, TRAIN_SETS, list_pairs, load_audio

MAX_SECONDS = 60.0
SPLITS = {  # menu label -> (clean folder, noisy folder)
    "Test set": TEST_SET,
    "Train 28spk": TRAIN_SETS["28spk"],
    "Train 56spk": TRAIN_SETS["56spk"],
}
# Which trained models may have seen a training file (validation speakers aside).
SEEN_BY = {"Train 28spk": ("28spk", "both"), "Train 56spk": ("56spk", "both")}


def decode_upload(data: bytes, suffix: str = ".wav") -> tuple[np.ndarray, int]:
    """Uploaded bytes -> (16 kHz mono float32, original sample rate), through the project's loader."""
    import soundfile as sf

    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        file_sr = sf.info(tmp).samplerate
        return load_audio(tmp, SR), int(file_sr)
    finally:
        os.unlink(tmp)


def cap_length(x: np.ndarray, max_s: float = MAX_SECONDS) -> tuple[np.ndarray, bool]:
    n = int(max_s * SR)
    return (x[:n], True) if len(x) > n else (x, False)


def list_split(root: str | Path, split: str) -> list[tuple[str, Path, Path]]:
    """Sorted (stem, clean_path, noisy_path) of one split; [] if the folders are not there."""
    try:
        return list_pairs(root, *SPLITS[split])
    except FileNotFoundError:
        return []


def available_splits(root: str | Path) -> list[str]:
    if not root or not Path(root).is_dir():
        return []
    return [s for s, (c, n) in SPLITS.items() if (Path(root) / c).is_dir() and (Path(root) / n).is_dir()]


def load_pair(clean_path: Path, noisy_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy) at 16 kHz, trimmed to a common length."""
    c, n = load_audio(clean_path, SR), load_audio(noisy_path, SR)
    m = min(len(c), len(n))
    return c[:m], n[:m]
