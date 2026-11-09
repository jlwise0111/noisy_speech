"""Loudness normalization applied to every track before it reaches the player.

Every track is brought to the same integrated loudness (ITU-R BS.1770 with EBU R128 gating, the
diagnostic viewer's implementation), so solo and diff comparisons hear processing differences, not
level differences. If any track's peak would go above −1 dBFS, all tracks are lowered together by
the same amount, so they all stay at one common loudness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CODE_ROOT  # noqa: F401  (sets sys.path for viewer)
from viewer import dsp

TARGET_LUFS = -20.0
CEILING_DBFS = -1.0


@dataclass
class Normalized:
    signals: dict[str, np.ndarray]   # gain applied, float32
    gain_db: dict[str, float]
    lufs_in: dict[str, float]        # before normalization
    peak_in: dict[str, float]        # sample peak before normalization, dBFS
    achieved_lufs: float             # common loudness of every (non-silent) track

    def lufs_out(self, k: str) -> float:
        return self.lufs_in[k] + self.gain_db[k]

    def peak_out(self, k: str) -> float:
        return self.peak_in[k] + self.gain_db[k]


def measure(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(integrated loudness in LUFS, sample peak in dBFS)."""
    return dsp.integrated_loudness(x, sr), float(20 * np.log10(np.max(np.abs(x)) + dsp.EPS))


def normalize(signals: dict[str, np.ndarray], sr: int, target_lufs: float = TARGET_LUFS,
              measured: dict[str, tuple[float, float]] | None = None) -> Normalized:
    """Bring every signal to one integrated loudness. `measured` reuses (lufs, peak) already computed."""
    keys = list(signals)
    m = measured or {k: measure(signals[k], sr) for k in keys}
    gains, achieved = dsp.loudness_match_gains([m[k][0] for k in keys], [m[k][1] for k in keys],
                                               target_lufs, CEILING_DBFS)
    gain_db = {k: float(g) for k, g in zip(keys, gains)}
    out = {k: (np.asarray(signals[k], np.float32) * np.float32(10 ** (gain_db[k] / 20))) for k in keys}
    return Normalized(out, gain_db, {k: m[k][0] for k in keys}, {k: m[k][1] for k in keys}, achieved)
