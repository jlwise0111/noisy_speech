"""Objective speech-enhancement metrics, each scored against the clean reference.

Extracted from diagnostic_viewer/viewer/metrics.py so batch numbers match what the viewer shows
for a single file: same SNR / SI-SDR definitions, PESQ in wide-band mode at 16 kHz (other rates
are resampled for scoring only), STOI via pystoi (which resamples to 10 kHz internally).
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

EPS = 1e-12
PESQ_FS = 16000


def snr(est: np.ndarray, ref: np.ndarray) -> float:
    """Plain SNR: reference power over residual (est − ref) power, in dB."""
    ref = ref.astype(np.float64)
    noise = est.astype(np.float64) - ref
    return float(10 * np.log10((np.sum(ref**2) + EPS) / (np.sum(noise**2) + EPS)))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al., 2019), zero-mean signals, in dB."""
    est = est.astype(np.float64) - np.mean(est)
    ref = ref.astype(np.float64) - np.mean(ref)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + EPS) / (np.sum(noise**2) + EPS)))


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g)


def pesq_wb(est: np.ndarray, ref: np.ndarray, sr: int) -> float:
    """Wide-band PESQ (ITU-T P.862.2). NaN if PESQ cannot score the file (e.g. no speech)."""
    from pesq import pesq

    try:
        # On silent input pesq divides by zero internally before raising NoUtterancesError.
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(pesq(PESQ_FS, resample(ref, sr, PESQ_FS), resample(est, sr, PESQ_FS), "wb"))
    except Exception:
        return float("nan")


def stoi(est: np.ndarray, ref: np.ndarray, sr: int) -> float:
    """Short-time objective intelligibility, 0–1. NaN if it cannot be computed."""
    from pystoi import stoi as _stoi

    try:
        return float(_stoi(ref, est, sr, extended=False))
    except Exception:
        return float("nan")


def score_file(clean: np.ndarray, noisy: np.ndarray, processed: np.ndarray, sr: int) -> dict:
    """All metrics for one triplet: noisy and processed vs clean, plus the change."""
    out = {}
    for name, f in (("pesq_wb", pesq_wb), ("stoi", stoi), ("snr_db", snr), ("si_sdr_db", si_sdr)):
        args = (sr,) if name in ("pesq_wb", "stoi") else ()
        n, p = f(noisy, clean, *args), f(processed, clean, *args)
        out[f"{name}_noisy"], out[f"{name}_processed"], out[f"{name}_delta"] = n, p, p - n
    return out
