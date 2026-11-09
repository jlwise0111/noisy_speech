"""Model registry and the lock-step guarantee: every output is exactly as long as the input."""
import os
from pathlib import Path

import numpy as np
import pytest

from mplayer import engine
from fakes import make_models, speechlike

EXPECTED = ["ridge_28spk", "ridge_56spk", "ridge_both", "wiener_minstats", "wiener_leading",
            "unet_28spk", "unet_56spk", "unet_both", "oracle"]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = make_models(tmp_path_factory.mktemp("models"))
    specs, missing = engine.registry(root)
    return specs, missing, engine.load_all(specs)


def test_registry_finds_every_run_in_order(models):
    specs, missing, _ = models
    assert [s.key for s in specs] == EXPECTED
    assert missing == []


def test_registry_reports_missing_runs(tmp_path):
    make_models(tmp_path, with_unet=False)
    specs, missing = engine.registry(tmp_path)
    assert "unet_28spk" not in [s.key for s in specs]
    assert len(missing) == 3 and all(m.startswith("U-Net") for m in missing)


@pytest.mark.parametrize("seconds", [0.5, 2.0, 3.3337])
def test_every_output_matches_input_length(models, seconds):
    specs, _, enh = models
    clean, noisy = speechlike(16000, seconds)
    out, errors = engine.run_all(specs, enh, noisy, clean)
    assert errors == {}
    assert list(out) == EXPECTED
    for k, y in out.items():
        assert y.dtype == np.float32 and y.shape == noisy.shape, k
        assert np.all(np.abs(y) <= 1.0), k
        assert np.all(np.isfinite(y)), k


def test_oracle_needs_clean(models):
    specs, _, enh = models
    _, noisy = speechlike(16000, 1.0)
    out, errors = engine.run_all(specs, enh, noisy, None)
    assert "oracle" not in out and errors == {}
    assert len(out) == len(EXPECTED) - 1


def test_parallel_equals_serial(models):
    specs, _, enh = models
    clean, noisy = speechlike(16000, 1.2, seed=3)
    par, _ = engine.run_all(specs, enh, noisy, clean)
    ser, _ = engine.run_all(specs, enh, noisy, clean, workers=1)
    for k in par:
        np.testing.assert_allclose(par[k], ser[k], atol=1e-6, err_msg=k)


def test_broken_model_does_not_stop_the_others(models):
    specs, _, enh = models
    enh = dict(enh, ridge_28spk=lambda n, c: 1 / 0)
    _, noisy = speechlike(16000, 1.0)
    out, errors = engine.run_all(specs, enh, noisy, None)
    assert "ZeroDivisionError" in errors["ridge_28spk"]
    assert "ridge_28spk" not in out and "unet_both" in out


def test_fit_length():
    assert engine.fit_length(np.ones(10) * 3, 12).tolist() == [1.0] * 10 + [0.0, 0.0]
    assert len(engine.fit_length(np.zeros(20), 12)) == 12


# The real trained runs, when they are mounted (/models in Docker, the project folder locally).
REAL = Path(os.environ.get("PLAYER_MODELS_DIR", Path(__file__).resolve().parents[2]))
REF = REAL / "unet_model/outputs/unet_28spk"


@pytest.mark.skipif(not (REF / "enhanced/p232_001.wav").is_file(), reason="trained runs not available")
def test_matches_pipeline_outputs():
    """Same file, same model: the player's output equals what the pipeline wrote (PCM-16 rounding)."""
    import soundfile as sf
    specs, _ = engine.registry(REAL)
    enh = engine.load_all(specs)
    noisy, _ = sf.read(REF / "noisy/p232_001.wav", dtype="float32")
    clean, _ = sf.read(REF / "clean/p232_001.wav", dtype="float32")
    out, errors = engine.run_all(specs, enh, noisy, clean)
    assert errors == {}
    runs = {"unet_28spk": "unet_model/outputs/unet_28spk/enhanced",
            "wiener_leading": "wiener_baseline/outputs/wiener_leading_test/processed",
            "oracle": "wiener_baseline/outputs/oracle_test/processed",
            "ridge_both": "linear_baseline/outputs/ridge_both/enhanced"}
    for k, d in runs.items():
        p = REAL / d / "p232_001.wav"
        if p.is_file() and k in out:
            ref, _ = sf.read(p, dtype="float32")
            assert np.max(np.abs(out[k] - ref)) < 1e-3, k
