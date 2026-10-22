"""Signal analysis: spectrograms, LTAS, level envelope, stats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

EPS = 1e-12
WINDOWS = ("hann", "hamming", "blackman", "blackmanharris", "boxcar")
SCALES = ("linear", "log", "mel")


@dataclass(frozen=True)
class StftParams:
    n_fft: int = 1024
    hop: int = 256
    window: str = "hann"
    scale: str = "linear"  # linear | log | mel
    n_mels: int = 128
    fmin: float = 40.0  # lower edge for log / mel axes
    max_frames: int = 1500  # display decimation in time
    max_bins: int = 384  # display decimation in frequency (linear only)


@dataclass
class Spectrogram:
    times: np.ndarray  # (n_frames,) seconds, frame centres
    freqs: np.ndarray  # (n_bins,) Hz (bin centre on the displayed axis)
    db: np.ndarray  # (n_bins, n_frames) power in dB re. full-scale
    scale: str


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangular HTK-mel filterbank, area-normalised. Returns (weights, centre_hz)."""
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    weights = np.zeros((n_mels, fft_freqs.size))
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i : i + 3]
        up = (fft_freqs - lo) / max(ctr - lo, EPS)
        down = (hi - fft_freqs) / max(hi - ctr, EPS)
        weights[i] = np.maximum(0.0, np.minimum(up, down))
    weights *= (2.0 / (hz_pts[2:] - hz_pts[:-2]))[:, None]
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    return weights, hz_pts[1:-1]


def _bin_mean(power: np.ndarray, axis: int, max_n: int) -> np.ndarray:
    """Average groups of rows/columns (in the power domain) so size ≤ max_n."""
    n = power.shape[axis]
    if n <= max_n:
        return power
    group = int(np.ceil(n / max_n))
    pad = (-n) % group
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (0, pad)
    p = np.pad(power, pad_width, mode="edge")
    if axis == 1:
        return p.reshape(p.shape[0], -1, group).mean(axis=2)
    return p.reshape(-1, group, p.shape[1]).mean(axis=1)


def _bin_axis(values: np.ndarray, max_n: int) -> np.ndarray:
    return _bin_mean(values[None, :], 1, max_n)[0]


def power_spectrogram(x: np.ndarray, sr: int, p: StftParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-resolution power STFT (no display decimation). Returns (freqs, times, power)."""
    freqs, times, z = signal.stft(
        x, fs=sr, window=p.window, nperseg=p.n_fft, noverlap=p.n_fft - p.hop,
        boundary="zeros", padded=True,
    )
    # scipy's default 'spectrum' scaling: a full-scale sine peaks at ~-6 dB.
    return freqs, times, np.abs(z) ** 2


def spectrogram(x: np.ndarray, sr: int, p: StftParams) -> Spectrogram:
    freqs, times, power = power_spectrogram(x, sr, p)
    if p.scale == "mel":
        w, centres = mel_filterbank(sr, p.n_fft, p.n_mels, p.fmin, sr / 2)
        power = w @ power
        freqs = centres
    elif p.scale == "log":
        n_out = min(p.max_bins, freqs.size)
        log_f = np.geomspace(max(p.fmin, freqs[1]), freqs[-1], n_out)
        pos = np.interp(log_f, freqs, np.arange(freqs.size))  # fractional bin index
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, freqs.size - 1)
        frac = (pos - lo)[:, None]
        power = power[lo] * (1 - frac) + power[hi] * frac
        freqs = log_f
    else:
        power = _bin_mean(power, 0, p.max_bins)
        freqs = _bin_axis(freqs, p.max_bins)
    power = _bin_mean(power, 1, p.max_frames)
    times = _bin_axis(times, p.max_frames)
    db = 10.0 * np.log10(power + EPS)
    return Spectrogram(times=times, freqs=freqs, db=db.astype(np.float32), scale=p.scale)


def ltas(x: np.ndarray, sr: int, n_fft: int = 2048, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Long-term average spectrum (Welch PSD) in dB/Hz."""
    nper = min(n_fft, len(x))
    f, pxx = signal.welch(x, fs=sr, window=window, nperseg=nper, noverlap=nper // 2)
    return f, 10.0 * np.log10(pxx + EPS)


def rms_envelope(x: np.ndarray, sr: int, win_ms: float = 20.0, hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Short-time RMS level in dBFS (full-scale sine = −3 dBFS)."""
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    times = (np.arange(len(rms)) * hop + win / 2) / sr
    return times, 20.0 * np.log10(rms + EPS)


def signal_stats(x: np.ndarray, sr: int, clip_threshold: float = 0.999) -> dict:
    x64 = x.astype(np.float64)
    peak = float(np.max(np.abs(x64))) if x64.size else 0.0
    rms = float(np.sqrt(np.mean(x64**2))) if x64.size else 0.0
    _, env_db = rms_envelope(x, sr)
    return {
        "duration_s": len(x) / sr,
        "sample_rate_hz": sr,
        "peak_dbfs": 20 * np.log10(peak + EPS),
        "rms_dbfs": 20 * np.log10(rms + EPS),
        "max_rms_dbfs": float(np.max(env_db)),
        # Level of the quietest 10% of 20 ms windows: a plain proxy for "how quiet are the pauses".
        "noise_floor_dbfs": float(np.percentile(env_db, 10)),
        "crest_factor_db": 20 * np.log10((peak + EPS) / (rms + EPS)),
        "clipped_samples": int(np.sum(np.abs(x64) >= clip_threshold)),
        "dc_offset": float(np.mean(x64)) if x64.size else 0.0,
    }


def k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """ITU-R BS.1770-4 K-weighting at any sample rate: high-shelf ("head") then high-pass (RLB).

    Bilinear designs from libebur128; at 48 kHz they reproduce the coefficients in the standard.
    """
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = (np.array([vh + vb * k / q + k * k, 2 * (k * k - vh), vh - vb * k / q + k * k]) / a0,
             np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]))
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    highpass = (np.array([1.0, -2.0, 1.0]),
                np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]))
    return [shelf, highpass]


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """Integrated loudness of a mono signal in LUFS (ITU-R BS.1770-4 / EBU R128 gating).

    400 ms blocks with 75% overlap, absolute gate at −70 LUFS, relative gate at −10 LU.
    Signals shorter than one block are measured as a single block. Returns −inf for silence.
    """
    y = x.astype(np.float64)
    for b, a in k_weighting(sr):
        y = signal.lfilter(b, a, y)
    block, step = int(round(0.4 * sr)), int(round(0.1 * sr))
    if len(y) <= block:
        z = np.array([np.mean(y**2)]) if len(y) else np.array([0.0])
    else:
        frames = np.lib.stride_tricks.sliding_window_view(y**2, block)[::step]
        z = frames.mean(axis=1)
    lufs = lambda p: -0.691 + 10 * np.log10(p + EPS)  # noqa: E731
    z = z[lufs(z) > -70.0]
    if not z.size:
        return float("-inf")
    z = z[lufs(z) > lufs(z.mean()) - 10.0]
    return float(lufs(z.mean()))


def loudness_match_gains(loudness: list[float], peaks: list[float], target_lufs: float = -20.0,
                         ceiling_dbfs: float = -1.0) -> tuple[list[float], float]:
    """Gains (dB) that bring every track to the same loudness, ideally `target_lufs`.

    If reaching the target would push any track's sample peak above `ceiling_dbfs`, all tracks are
    lowered by the same amount, so they stay equally loud relative to each other. Silent tracks
    get 0 dB. Returns (gains_db, achieved_lufs).
    """
    gains = [0.0 if not np.isfinite(lv) else target_lufs - lv for lv in loudness]
    over = [p + g - ceiling_dbfs for p, g, lv in zip(peaks, gains, loudness) if np.isfinite(lv)]
    headroom = max([0.0] + over)
    return [g - headroom if np.isfinite(lv) else 0.0 for g, lv in zip(gains, loudness)], target_lufs - headroom


def waveform_envelope(x: np.ndarray, sr: int, max_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max-per-bucket decimation that preserves the visual waveform shape."""
    n = len(x)
    if n <= max_points:
        return np.arange(n) / sr, x
    buckets = max_points // 2
    size = int(np.ceil(n / buckets))
    pad = (-n) % size
    xp = np.pad(x, (0, pad), mode="edge").reshape(-1, size)
    mins, maxs = xp.min(axis=1), xp.max(axis=1)
    y = np.empty(mins.size * 2, dtype=x.dtype)
    y[0::2], y[1::2] = mins, maxs
    t0 = np.arange(mins.size) * size
    t = np.empty_like(y, dtype=float)
    t[0::2] = t0
    t[1::2] = t0 + size / 2
    return t / sr, y
