import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_baseline  # noqa: E402
from wiener import WienerParams  # noqa: E402
from wiener.audio_io import SR, read_wav  # noqa: E402


def _write_48k_pair(root: Path, stem: str = "p232_001", seconds: float = 1.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * 48000)) / 48000
    clean = 0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.3)
    noisy = clean + 0.05 * rng.normal(size=t.size)
    paths = {}
    for role, x in (("clean", clean), ("noisy", noisy)):
        p = root / f"{role}_testset_wav" / f"{stem}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, x, 48000, subtype="PCM_16")
        paths[role] = p
    return paths


def test_read_wav_resamples_to_16k(tmp_path):
    paths = _write_48k_pair(tmp_path)
    x, sr = read_wav(paths["noisy"], SR)
    assert sr == 16000 and len(x) == 16000 and x.dtype == np.float32
    x48, sr48 = read_wav(paths["noisy"])
    assert sr48 == 48000 and len(x48) == 48000


def test_48k_input_gives_16k_triplet(tmp_path):
    paths = _write_48k_pair(tmp_path / "data")
    run_dir = tmp_path / "out" / "wiener_leading_test"
    row = run_baseline.process_one(("p232_001", paths["noisy"], paths["clean"], run_dir, "wiener", "leading",
                                    WienerParams(), SR, None))
    infos = [sf.info(run_dir / role / "p232_001.wav") for role in ("clean", "noisy", "processed")]
    assert {(i.samplerate, i.channels, i.frames) for i in infos} == {(16000, 1, 16000)}
    assert row["si_sdr_processed"] > row["si_sdr_noisy"]
