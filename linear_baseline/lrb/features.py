"""STFT, log-magnitude and 3x3 local spectral patches."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import istft as _istft
from scipy.signal import stft as _stft

EPS = 1e-6
N_FEATURES = 9  # 3 frames x 3 frequency bins


def stft(x, n_fft, hop):
    """Complex STFT as (frames, bins)."""
    _, _, Z = _stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return Z.T


def istft(Z, n_fft, hop, length):
    """Inverse of `stft`, trimmed or zero-padded to exactly `length` samples."""
    _, x = _istft(Z.T, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    x = x[:length]
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    return x


def log_mag(Z):
    return np.log(np.abs(Z) + EPS)


def from_log_mag(L):
    return np.maximum(np.exp(L) - EPS, 0.0)


def patches(L):
    """3x3 neighbourhood of every (t, f), edge-padded: (T, F) -> (T, F, 9).

    Feature k = 3*dt + df for dt, df in {0,1,2}, i.e. (t-1+dt, f-1+df); k=4 is the centre.
    """
    P = np.pad(L, 1, mode="edge")
    W = sliding_window_view(P, (3, 3))  # (T, F, 3, 3)
    return W.reshape(L.shape[0], L.shape[1], N_FEATURES)
