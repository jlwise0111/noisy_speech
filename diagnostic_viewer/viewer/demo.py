"""Synthetic speech-like test material (no real recordings needed)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def speech_like(sr: int = 16000, seconds: float = 4.0, seed: int = 0) -> np.ndarray:
    """Voiced 'syllables': gliding harmonic source through formant resonators, ~4 Hz AM."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * rng.standard_normal() * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = sum(np.sin(k * phase) / k for k in range(1, 40) if k * f0.max() < sr / 2)
    out = np.zeros(n)
    for fc, bw, g in ((700, 130, 1.0), (1220, 70, 0.6), (2600, 160, 0.3)):
        b, a = signal.iirpeak(fc, fc / bw, fs=sr)
        out += g * signal.lfilter(b, a, src)
    syll = np.clip(np.sin(2 * np.pi * 3.5 * t + rng.uniform(0, np.pi)), 0, None) ** 1.5
    gaps = (np.sin(2 * np.pi * 0.4 * t) > -0.6).astype(float)  # pauses between "words"
    env = signal.lfilter([1 - 0.995], [1, -0.995], syll * gaps)
    out *= env
    return (0.5 * out / (np.max(np.abs(out)) + 1e-12)).astype(np.float32)


def pink_noise(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spec.size)
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n)
    return (x / np.std(x)).astype(np.float32)


def scale_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_c, p_n = np.mean(clean.astype(float) ** 2), np.mean(noise.astype(float) ** 2)
    return (noise * np.sqrt(p_c / (p_n * 10 ** (snr_db / 10)))).astype(np.float32)


def make_triplet(sr: int = 16000, seconds: float = 4.0, snr_db: float = 0.0, seed: int = 0):
    """Return (clean, noisy, processed). 'processed' = 15 dB less noise + mild HF speech loss."""
    clean = speech_like(sr, seconds, seed)
    noise = scale_to_snr(clean, pink_noise(len(clean), seed + 1), snr_db)
    noisy = clean + noise
    b, a = signal.butter(2, 3500, fs=sr)  # simulate over-suppression of high-frequency speech
    damaged = 0.7 * clean + 0.3 * signal.lfilter(b, a, clean)
    processed = damaged + noise * 10 ** (-15 / 20)
    return clean, noisy.astype(np.float32), processed.astype(np.float32)
