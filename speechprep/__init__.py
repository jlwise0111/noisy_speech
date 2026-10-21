"""Shared 16 kHz preprocessing for every model in the project.

One place defines the audio front end so that every system sees the same signal:
    48 kHz Valentini wav -> 16 kHz mono (resample_poly) -> STFT (Hann, n_fft=512, hop=128) -> 257 x T

The settings and the speaker split are identical to linear_baseline (lrb), which the tests check.
Only numpy, scipy and soundfile are needed, so any model's environment can import it.
"""
from .audio import EPS, HOP, N_FFT, N_BINS, SR, istft, load_audio, log_mag, stft
from .cache import WaveCache
from .splits import TEST_SET, TRAIN_SETS, list_pairs, speaker, split_by_speaker, test_pairs, train_pairs

__all__ = [
    "EPS", "HOP", "N_FFT", "N_BINS", "SR", "istft", "load_audio", "log_mag", "stft",
    "WaveCache",
    "TEST_SET", "TRAIN_SETS", "list_pairs", "speaker", "split_by_speaker", "test_pairs", "train_pairs",
]
