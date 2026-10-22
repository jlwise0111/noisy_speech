"""STFT-domain Wiener filter for single-channel speech enhancement.

Pipeline (blind mode):
    noisy x  ->  STFT Y  ->  noise PSD estimate N  ->  decision-directed a priori
    SNR xi  ->  Wiener gain G = xi / (1 + xi)  ->  S_hat = G * Y  ->  ISTFT

The Wiener gain is the per-bin MMSE-optimal linear estimator of the clean
spectrum given the speech and noise powers; everything else here is about
estimating those powers from the noisy signal alone.

``wiener_oracle`` uses the true clean signal to build the ideal Wiener mask.
It is not a real baseline, only an upper bound on what this filter family
can reach.

The filter runs at 16 kHz only (SR): 32 ms frames are 512 samples with a
128-sample hop, the same grid as every other model in the project. Resample
before calling it (wiener.audio_io.read_wav does); any other rate is an error.

Algorithm details and tuning notes: wiener_baseline/docs/wiener_baseline.md (section 6).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import istft, stft

EPS = 1e-12
SR = 16000  # the only sample rate the filter runs at


@dataclass
class WienerParams:
    frame_ms: float = 32.0        # STFT window length
    overlap: float = 0.75         # fraction of window overlap
    alpha: float = 0.98           # decision-directed smoothing (Ephraim-Malah)
    gain_floor_db: float = -20.0  # lower bound on G; limits musical noise
    xi_min_db: float = -25.0      # lower bound on a priori SNR
    noise_init_s: float = 0.25    # "leading": length of the assumed speech-free lead-in
    ms_window_s: float = 1.5      # "minstats": minimum-search window
    ms_smooth: float = 0.85       # "minstats": periodogram smoothing factor
    ms_bias: float = 2.0          # "minstats": compensates the downward bias of a minimum

    def to_dict(self) -> dict:
        return asdict(self)


def check_sr(sr: int) -> None:
    """The filter is tuned and documented for 16 kHz; refuse anything else instead of silently running at it."""
    if sr != SR:
        raise ValueError(f"the Wiener filter runs at {SR} Hz only, got {sr} Hz: resample first "
                         "(wiener.audio_io.read_wav does)")


def _stft_cfg(sr: int, p: WienerParams) -> tuple[int, int]:
    check_sr(sr)
    nperseg = int(round(sr * p.frame_ms / 1000.0))
    noverlap = int(round(nperseg * p.overlap))
    return nperseg, noverlap


def _hop_s(sr: int, p: WienerParams) -> float:
    nperseg, noverlap = _stft_cfg(sr, p)
    return (nperseg - noverlap) / sr


def analyze(x: np.ndarray, sr: int, p: WienerParams) -> np.ndarray:
    """Complex STFT, shape (freq, frames)."""
    nperseg, noverlap = _stft_cfg(sr, p)
    _, _, Y = stft(x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    return Y


def synthesize(S: np.ndarray, sr: int, p: WienerParams, length: int) -> np.ndarray:
    """Inverse STFT, trimmed/zero-padded to exactly ``length`` samples."""
    nperseg, noverlap = _stft_cfg(sr, p)
    _, y = istft(S, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
    if len(y) >= length:
        y = y[:length]
    else:
        y = np.pad(y, (0, length - len(y)))
    return y.astype(np.float32)


def noise_psd_leading(P: np.ndarray, sr: int, p: WienerParams) -> np.ndarray:
    """Mean power over the first ``noise_init_s`` seconds. Shape (freq, 1)."""
    n = max(1, int(p.noise_init_s / _hop_s(sr, p)))
    return P[:, :n].mean(axis=1, keepdims=True)


def noise_psd_minstats(P: np.ndarray, sr: int, p: WienerParams) -> np.ndarray:
    """Simplified minimum statistics (after Martin, 2001). Shape (freq, frames).

    Smooth the periodogram over time, take a sliding minimum (speech pauses
    pull it down to the noise floor) and scale up to undo the minimum's bias.
    Non-causal (centred window), which is fine for offline processing.
    """
    smoothed = np.empty_like(P)
    acc = P[:, 0]
    for t in range(P.shape[1]):
        acc = p.ms_smooth * acc + (1.0 - p.ms_smooth) * P[:, t]
        smoothed[:, t] = acc
    win = max(1, int(p.ms_window_s / _hop_s(sr, p)))
    return p.ms_bias * minimum_filter1d(smoothed, size=win, axis=1, mode="nearest")


def wiener_gain(Y: np.ndarray, noise_psd: np.ndarray, p: WienerParams) -> np.ndarray:
    """Wiener gain with decision-directed a priori SNR. Same shape as ``Y``."""
    P = np.abs(Y) ** 2
    N = np.broadcast_to(np.maximum(noise_psd, EPS), P.shape)
    gamma = P / N                                   # a posteriori SNR
    xi_min = 10.0 ** (p.xi_min_db / 10.0)
    g_min = 10.0 ** (p.gain_floor_db / 20.0)

    G = np.empty_like(P)
    prev_s2 = N[:, 0]                               # gives xi = alpha + ... on frame 0
    for t in range(P.shape[1]):
        xi = p.alpha * prev_s2 / N[:, t] + (1.0 - p.alpha) * np.maximum(gamma[:, t] - 1.0, 0.0)
        xi = np.maximum(xi, xi_min)
        g = np.maximum(xi / (1.0 + xi), g_min)
        G[:, t] = g
        prev_s2 = (g ** 2) * P[:, t]                # |S_hat|^2 for the next frame
    return G


NOISE_ESTIMATORS = {
    "leading": noise_psd_leading,
    "minstats": noise_psd_minstats,
}


def wiener_filter(
    x: np.ndarray,
    sr: int,
    noise_method: str = "leading",
    params: WienerParams | None = None,
) -> np.ndarray:
    """Enhance a mono float signal. Output has the same length, clipped to [-1, 1]."""
    p = params or WienerParams()
    x = np.asarray(x, dtype=np.float64)
    Y = analyze(x, sr, p)
    noise = NOISE_ESTIMATORS[noise_method](np.abs(Y) ** 2, sr, p)
    G = wiener_gain(Y, noise, p)
    return np.clip(synthesize(G * Y, sr, p, len(x)), -1.0, 1.0)


def wiener_oracle(
    x: np.ndarray,
    clean: np.ndarray,
    sr: int,
    params: WienerParams | None = None,
) -> np.ndarray:
    """Ideal Wiener mask |S|^2 / (|S|^2 + |N|^2) from the true clean signal (upper bound)."""
    p = params or WienerParams()
    x = np.asarray(x, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    Y = analyze(x, sr, p)
    S2 = np.abs(analyze(clean, sr, p)) ** 2
    N2 = np.abs(analyze(x - clean, sr, p)) ** 2
    G = S2 / (S2 + N2 + EPS)
    return np.clip(synthesize(G * Y, sr, p, len(x)), -1.0, 1.0)
