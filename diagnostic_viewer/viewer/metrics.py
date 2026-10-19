"""Objective speech-enhancement metrics, each scored against the clean reference."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

EPS = 1e-12


@dataclass
class MetricResult:
    value: float | None
    note: str = ""


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


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g)


def pesq_score(est: np.ndarray, ref: np.ndarray, sr: int) -> MetricResult:
    from pesq import pesq

    note = ""
    if sr == 8000:
        mode, fs = "nb", 8000
    else:
        mode, fs = "wb", 16000
        if sr != 16000:
            ref, est = _resample(ref, sr, fs), _resample(est, sr, fs)
            note = f"resampled {sr}→{fs} Hz for scoring"
    try:
        # On silent input pesq divides by zero internally before raising NoUtterancesError.
        with np.errstate(invalid="ignore", divide="ignore"):
            return MetricResult(float(pesq(fs, ref, est, mode)), f"{mode}" + (f"; {note}" if note else ""))
    except Exception as e:  # e.g. NoUtterancesError on silent input
        return MetricResult(None, f"PESQ failed: {e}")


def stoi_score(est: np.ndarray, ref: np.ndarray, sr: int, extended: bool = False) -> MetricResult:
    from pystoi import stoi

    try:
        return MetricResult(float(stoi(ref, est, sr, extended=extended)), "internally resampled to 10 kHz")
    except Exception as e:
        return MetricResult(None, f"STOI failed: {e}")


METRIC_INFO = {
    "SNR (dB)": "Clean power / (estimate − clean) power. Sensitive to gain and phase.",
    "SI-SDR (dB)": "Scale-invariant SDR. Standard separation metric; ignores overall gain.",
    "PESQ": "ITU-T P.862 perceptual quality, 1.0–4.5 (wb: 1.04–4.64).",
    "STOI": "Short-time objective intelligibility, 0–1.",
}


def compute_all(clean: np.ndarray, noisy: np.ndarray, processed: np.ndarray, sr: int) -> pd.DataFrame:
    """Table with one row per metric: noisy vs clean, processed vs clean, and Δ."""
    rows = []

    def add(name, f_noisy: MetricResult, f_proc: MetricResult):
        delta = None
        if f_noisy.value is not None and f_proc.value is not None:
            delta = f_proc.value - f_noisy.value
        notes = "; ".join(dict.fromkeys(n for n in (f_noisy.note, f_proc.note) if n))
        rows.append({"metric": name, "noisy": f_noisy.value, "processed": f_proc.value,
                     "Δ (processed − noisy)": delta, "notes": notes})

    add("SNR (dB)", MetricResult(snr(noisy, clean)), MetricResult(snr(processed, clean)))
    add("SI-SDR (dB)", MetricResult(si_sdr(noisy, clean)), MetricResult(si_sdr(processed, clean)))
    add("PESQ", pesq_score(noisy, clean, sr), pesq_score(processed, clean, sr))
    add("STOI", stoi_score(noisy, clean, sr), stoi_score(processed, clean, sr))
    return pd.DataFrame(rows)


SCORECARD = (
    # (plain label, metric row, technical tooltip, decimals)
    ("Separation", "SI-SDR (dB)", "SI-SDR in dB: how cleanly speech is separated from everything else", 1),
    ("Quality", "PESQ", "PESQ (ITU-T P.862): predicted listening quality, ~1 (bad) to 4.5 (excellent)", 2),
    ("Intelligibility", "STOI", "STOI: predicted word intelligibility, 0 to 1", 2),
)


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def scorecard(df: pd.DataFrame) -> list[dict]:
    """The three headline metrics in plain language, for the top-right panel. Higher is better for all."""
    rows = df.set_index("metric")
    out = []
    for label, metric, tip, decimals in SCORECARD:
        r = rows.loc[metric]
        out.append({"label": label, "tooltip": tip, "decimals": decimals, "noisy": _num(r["noisy"]),
                    "processed": _num(r["processed"]), "delta": _num(r["Δ (processed − noisy)"])})
    return out


def flatten(df: pd.DataFrame) -> dict:
    """One-row dict (for batch CSV): e.g. {'SI-SDR (dB) noisy': .., 'SI-SDR (dB) processed': ..}."""
    out = {}
    for _, r in df.iterrows():
        for col in ("noisy", "processed", "Δ (processed − noisy)"):
            out[f"{r['metric']} {col}"] = r[col]
    return out
