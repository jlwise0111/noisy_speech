"""Tiny stand-in model runs and dataset folders, so the tests need no trained weights or Valentini."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

import mplayer  # noqa: F401  (puts the project's packages on sys.path)

WIENER_PARAMS = {"frame_ms": 32.0, "overlap": 0.75, "alpha": 0.98, "gain_floor_db": -20.0, "xi_min_db": -25.0,
                 "noise_init_s": 0.25, "ms_window_s": 1.5, "ms_smooth": 0.85, "ms_bias": 2.0}


def speechlike(sr: int, seconds: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy): a few harmonic bursts after 0.3 s of silence, plus white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    env = ((t > 0.3) & ((t * 3) % 1 < 0.6)).astype(np.float32)
    clean = (0.3 * env * sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 540, 900), 1)))
    noisy = clean + 0.05 * rng.standard_normal(len(t))
    return clean.astype(np.float32), noisy.astype(np.float32)


def make_models(root: Path, with_unet: bool = True) -> Path:
    out = root / "linear_baseline/outputs"
    rng = np.random.default_rng(0)
    for t in ("28spk", "56spk", "both"):
        d = out / f"ridge_{t}"
        d.mkdir(parents=True)
        W = np.zeros((257, 9)); W[:, 4] = 0.9  # mostly the centre bin: a mild attenuation
        np.savez(d / "model.npz", W=W + 0.01 * rng.standard_normal(W.shape), b=np.full(257, -0.1),
                 alpha=np.ones(257), sr=16000, n_fft=512, hop=128)
    for name, est in (("wiener_leading_test", "leading"), ("wiener_minstats_test", "minstats"),
                      ("oracle_test", None)):
        d = root / "wiener_baseline/outputs" / name
        d.mkdir(parents=True)
        meta = {"method": "oracle"} if est is None else {"method": "wiener", "noise_est": est}
        (d / "params.json").write_text(json.dumps({**meta, "sr": 16000, "params": WIENER_PARAMS}))
    if with_unet:
        import torch
        from unet import UNet
        for t in ("28spk", "56spk", "both"):
            d = root / "unet_model/outputs" / f"unet_{t}"
            d.mkdir(parents=True)
            torch.manual_seed(0)
            m = UNet(base_channels=4).eval()
            torch.save({"model": m.state_dict(), "config": {"base_channels": 4}}, d / "best.pt")
        from unet import BiLSTM
        d = root / "unet_model/outputs/bilstm_28spk"
        d.mkdir(parents=True)
        m = BiLSTM(hidden=8).eval()
        torch.save({"model": m.state_dict(), "config": {"arch": "bilstm", "lstm_hidden": 8, "lstm_layers": 2}},
                   d / "best.pt")
    return root


def make_dataset(root: Path, n: int = 2, sr: int = 48000) -> Path:
    for split, (cdir, ndir) in {"test": ("clean_testset_wav", "noisy_testset_wav"),
                                "28": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav")}.items():
        for i in range(n):
            c, x = speechlike(sr, 1.5, seed=i)
            for d, y in ((cdir, c), (ndir, x)):
                (root / d).mkdir(parents=True, exist_ok=True)
                sf.write(root / d / f"p{232 if split == 'test' else 226}_{i:03d}.wav", y, sr)
    return root
