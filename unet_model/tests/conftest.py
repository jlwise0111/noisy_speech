import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parent
UNET = HERE.parent
REPO = UNET.parent
for p in (REPO, UNET):
    sys.path.insert(0, str(p))


def _utterance(rng, n, sr):
    """Harmonic 'speech' bursts with silent gaps, like a Valentini clip (lead-in pause first)."""
    t = np.arange(n) / sr
    env = ((t > 0.3) & (t < 0.9)) | ((t > 1.2) & (t < 1.7))
    f0 = rng.uniform(100, 220)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    return (0.3 * env * x).astype(np.float32)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A miniature Valentini tree: 48 kHz pairs, 4 train speakers, 2 test speakers."""
    root = tmp_path_factory.mktemp("valentini")
    rng = np.random.default_rng(0)
    sr = 48000
    layout = {("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"): ["p226", "p227", "p228", "p229"],
              ("clean_testset_wav", "noisy_testset_wav"): ["p232", "p257"]}
    for (clean_dir, noisy_dir), speakers in layout.items():
        (root / clean_dir).mkdir()
        (root / noisy_dir).mkdir()
        for spk in speakers:
            for i in range(1, 4):
                n = int(sr * rng.uniform(1.8, 2.4))
                clean = _utterance(rng, n, sr)
                noisy = np.clip(clean + 0.05 * rng.standard_normal(n).astype(np.float32), -1, 1)
                sf.write(root / clean_dir / f"{spk}_{i:03d}.wav", clean, sr, subtype="PCM_16")
                sf.write(root / noisy_dir / f"{spk}_{i:03d}.wav", noisy, sr, subtype="PCM_16")
    return root
