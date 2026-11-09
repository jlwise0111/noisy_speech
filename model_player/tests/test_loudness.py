"""Every track handed to the player is at one common integrated loudness."""
import os
from pathlib import Path

import numpy as np
import pytest

from mplayer import engine, loudness
from fakes import make_models, speechlike


def _check(norm: loudness.Normalized):
    lufs = [loudness.measure(x, 16000)[0] for x in norm.signals.values()]
    audible = [v for v in lufs if np.isfinite(v)]
    assert np.allclose(audible, norm.achieved_lufs, atol=0.05), lufs
    assert max(loudness.measure(x, 16000)[1] for x in norm.signals.values()) <= loudness.CEILING_DBFS + 1e-3
    for k, x in norm.signals.items():
        assert x.dtype == np.float32
        assert abs(loudness.measure(x, 16000)[1] - norm.peak_out(k)) < 1e-3


def test_model_outputs_all_normalized(tmp_path):
    specs, _ = engine.registry(make_models(tmp_path))
    clean, noisy = speechlike(16000, 3.0)
    out, _ = engine.run_all(specs, engine.load_all(specs), noisy, clean)
    norm = loudness.normalize({"clean": clean, "noisy": noisy, **out}, 16000)
    assert set(norm.signals) == {"clean", "noisy", *out}
    _check(norm)


def test_peak_ceiling_lowers_everything_together():
    rng = np.random.default_rng(0)
    quiet_spiky = 0.001 * rng.standard_normal(32000); quiet_spiky[1000] = 0.5  # needs lots of gain
    norm = loudness.normalize({"a": quiet_spiky, "b": 0.1 * rng.standard_normal(32000)}, 16000)
    assert norm.achieved_lufs < loudness.TARGET_LUFS
    _check(norm)


def test_silent_track_is_left_alone():
    norm = loudness.normalize({"silent": np.zeros(16000, np.float32),
                               "tone": 0.1 * np.sin(np.arange(16000) / 5).astype(np.float32)}, 16000)
    assert norm.gain_db["silent"] == 0.0
    _check(norm)


REAL = Path(os.environ.get("PLAYER_MODELS_DIR", Path(__file__).resolve().parents[2]))
REF = REAL / "unet_model/outputs/unet_28spk"


@pytest.mark.skipif(not (REF / "noisy/p232_001.wav").is_file(), reason="trained runs not available")
def test_real_model_outputs_all_normalized():
    import soundfile as sf
    specs, _ = engine.registry(REAL)
    noisy, _ = sf.read(REF / "noisy/p232_001.wav", dtype="float32")
    clean, _ = sf.read(REF / "clean/p232_001.wav", dtype="float32")
    out, _ = engine.run_all(specs, engine.load_all(specs), noisy, clean)
    _check(loudness.normalize({"clean": clean, "noisy": noisy, **out}, 16000))
