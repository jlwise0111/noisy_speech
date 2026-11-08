"""Apply a trained U-Net to noisy audio and write viewer / metric-suite triplets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speechprep import SR, WaveCache, istft, log_mag, stft


@torch.no_grad()
def predict_mask(model, noisy: np.ndarray, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Whole-utterance mask (F, T) and the noisy STFT it applies to."""
    Y = stft(noisy)
    x = torch.from_numpy(log_mag(Y))[None].to(device)
    return model(x)[0].cpu().numpy(), Y


def enhance(model, noisy: np.ndarray, device="cpu") -> np.ndarray:
    """Mask the noisy STFT (noisy phase kept) and resynthesize, same length as the input."""
    M, Y = predict_mask(model, noisy, device)
    return istft(M * Y, len(noisy))


def write_triplet(out_dir: Path, stem: str, clean, noisy, enhanced, sr: int = SR) -> None:
    """clean/noisy/enhanced wavs with identical sr, channels and length (the viewer's strict check)."""
    n = len(clean)
    for role, x in (("clean", clean), ("noisy", noisy), ("enhanced", enhanced)):
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        p = out_dir / role / f"{stem}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def enhance_pairs(model, pairs, cache: WaveCache, out_dir: Path, device="cpu", log=print) -> int:
    """Enhance every (stem, clean, noisy) pair and write the clean/noisy/enhanced wavs. Returns the count."""
    model.eval()
    for i, (stem, clean_path, noisy_path) in enumerate(pairs, 1):
        clean, noisy = cache.load(clean_path), cache.load(noisy_path)
        write_triplet(out_dir, stem, clean, noisy, enhance(model, noisy, device))
        if i % 100 == 0 or i == len(pairs):
            log(f"  enhanced {i}/{len(pairs)}")
    return len(pairs)
