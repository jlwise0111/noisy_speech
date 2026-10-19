"""Loading .wav files and strict validation of a triplet."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class TripletError(ValueError):
    """Raised when the files of a triplet cannot be compared sample-for-sample."""


@dataclass
class Audio:
    path: Path
    data: np.ndarray  # float32, shape (n_samples, n_channels)
    sr: int
    subtype: str

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sr

    def channel(self, ch: int) -> np.ndarray:
        return self.data[:, ch]


def load_wav(path: Path | str) -> Audio:
    path = Path(path)
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return Audio(path=path, data=data, sr=int(sr), subtype=info.subtype)


def validate_strict(tracks: dict[str, Audio]) -> None:
    """Raise TripletError listing every mismatch against the first track."""
    if not tracks:
        raise TripletError("No tracks to compare.")
    (ref_name, ref), *others = tracks.items()
    problems = []
    for name, a in others:
        if a.sr != ref.sr:
            problems.append(f"sample rate — {name}: {a.sr} Hz ≠ {ref_name}: {ref.sr} Hz")
        if a.n_channels != ref.n_channels:
            problems.append(
                f"channels — {name}: {a.n_channels} ≠ {ref_name}: {ref.n_channels}"
            )
        if a.n_samples != ref.n_samples:
            problems.append(
                f"length — {name}: {a.n_samples} samples ({a.duration:.3f} s) ≠ "
                f"{ref_name}: {ref.n_samples} samples ({ref.duration:.3f} s)"
            )
    if problems:
        raise TripletError("Files are not directly comparable:\n" + "\n".join(f"• {p}" for p in problems))


def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    """Encode a mono float signal as 16-bit PCM WAV for browser playback."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
