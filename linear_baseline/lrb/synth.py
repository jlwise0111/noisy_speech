"""Per-file feature/statistics extraction and enhancement (resynthesis with the noisy phase)."""

import numpy as np
import soundfile as sf

from .data import load_audio
from .features import N_FEATURES, from_log_mag, istft, log_mag, patches, stft
from .ridge import Stats


def file_stats(clean_path, noisy_path, sr, n_fft, hop):
    """Ridge sufficient statistics for one clean/noisy pair."""
    clean, noisy = load_audio(clean_path, sr), load_audio(noisy_path, sr)
    n = min(len(clean), len(noisy))
    Lc = log_mag(stft(clean[:n], n_fft, hop))
    Ln = log_mag(stft(noisy[:n], n_fft, hop))
    return Stats.from_arrays(patches(Ln), Lc)


def chunk_stats(pairs, sr, n_fft, hop):
    """Summed statistics for a list of (stem, clean_path, noisy_path)."""
    total = Stats.zeros(n_fft // 2 + 1, N_FEATURES)
    for _, c, nz in pairs:
        total = total + file_stats(c, nz, sr, n_fft, hop)
    return total


def enhance(noisy, model, n_fft, hop):
    """Predict the clean log-magnitude per bin and resynthesize with the noisy phase."""
    Z = stft(noisy, n_fft, hop)
    L_hat = model.predict(patches(log_mag(Z)))
    Z_hat = from_log_mag(L_hat) * np.exp(1j * np.angle(Z))
    return istft(Z_hat, n_fft, hop, len(noisy)).astype(np.float32)


def write_triplet(out_dir, stem, clean, noisy, enhanced, sr):
    """Write clean/noisy/enhanced wavs with identical sr, channels and length (viewer requirement)."""
    n = len(clean)
    for role, x in (("clean", clean), ("noisy", noisy), ("enhanced", enhanced)):
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        p = out_dir / role / f"{stem}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")
