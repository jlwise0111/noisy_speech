"""Audio loading, resampling and the project STFT.

Spectrograms are shaped (freq, frames) = (257, T): frequency on the rows, like an image,
which is the layout a 2D CNN expects. (linear_baseline uses the transpose, (T, 257).)
"""
from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import istft as _istft
from scipy.signal import resample_poly
from scipy.signal import stft as _stft

SR = 16000          # PESQ-WB is defined at 16 kHz; also 3x less data than 48 kHz
N_FFT = 512         # 32 ms window -> 257 frequency bins (31.25 Hz apart)
HOP = 128           # 8 ms hop, 75% overlap (same as the Ridge baseline)
N_BINS = N_FFT // 2 + 1
EPS = 1e-6          # log floor, same as the Ridge baseline


def load_audio(path, sr: int = SR) -> np.ndarray:
    """Read a wav (read-only), average channels to mono, resample to `sr`. float32."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Complex STFT, shape (n_fft // 2 + 1, frames)."""
    _, _, Z = _stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return Z


def istft(Z: np.ndarray, length: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Inverse of `stft`, trimmed or zero-padded to exactly `length` samples. float32."""
    _, x = _istft(Z, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    x = x[:length]
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    return x.astype(np.float32)


def log_mag(Z: np.ndarray) -> np.ndarray:
    """log(|Z| + EPS): compresses the ~100 dB range of speech spectra into a range a network can use."""
    return np.log(np.abs(Z) + EPS).astype(np.float32)
