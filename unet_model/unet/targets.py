"""Mask targets and the voice-activity (silence) labels, computed from a clean/noisy STFT pair.

Both use the same definitions as eda/simple_eda.py. The silence threshold is the exception: the EDA
uses 40 dB, but on the hop-128 grid that marks only ~5% of training frames silent (30 dB: ~28%), too
few for a silence-weighted loss to matter. So training defaults to 30 dB (TrainConfig.silence_db).
"""
from __future__ import annotations

import numpy as np

SILENCE_DB = 30.0   # a frame is silent if its clean energy is this far below the utterance peak
MASK_EPS = 1e-8


def mask_target(C: np.ndarray, Y: np.ndarray, kind: str = "irm") -> np.ndarray:
    """(F, T) target mask in [0, 1] from clean STFT C and noisy STFT Y.

    irm: clip(|C| / |Y|, 0, 1), the magnitude ratio mask.
    psm: clip(|C| / |Y| * cos(angle C - angle Y), 0, 1), the phase-sensitive mask. It is lower where
         the noisy phase is wrong, so it is the best mask to apply *given* that we keep the noisy phase.
    """
    ratio = np.abs(C) / (np.abs(Y) + MASK_EPS)
    if kind == "psm":
        ratio = ratio * np.cos(np.angle(C) - np.angle(Y))
    elif kind != "irm":
        raise ValueError(f"unknown mask target {kind!r} (irm or psm)")
    return np.clip(ratio, 0.0, 1.0).astype(np.float32)


def speech_frames(C: np.ndarray, silence_db: float = SILENCE_DB) -> np.ndarray:
    """(T,) bool: True where the clean frame energy is within `silence_db` of the utterance peak."""
    energy = (np.abs(C) ** 2).sum(axis=0)
    frame_db = 10.0 * np.log10(np.maximum(energy, 1e-10))
    return frame_db >= frame_db.max() - silence_db
