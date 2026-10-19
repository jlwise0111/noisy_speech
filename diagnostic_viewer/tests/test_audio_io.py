import numpy as np
import pytest
import soundfile as sf

from viewer.audio_io import TripletError, load_wav, validate_strict


def _audio(tmp_path, name, n, sr, ch=1):
    p = tmp_path / name
    sf.write(p, np.zeros((n, ch), dtype=np.float32), sr)
    return load_wav(p)


def test_matching_triplet_passes(tmp_path):
    t = {r: _audio(tmp_path, f"{r}.wav", 1600, 16000) for r in ("clean", "noisy", "processed")}
    validate_strict(t)
    assert t["clean"].data.shape == (1600, 1)


def test_sample_rate_mismatch(tmp_path):
    t = {"clean": _audio(tmp_path, "c.wav", 1600, 16000),
         "noisy": _audio(tmp_path, "n.wav", 1600, 16000),
         "processed": _audio(tmp_path, "p.wav", 1600, 48000)}
    with pytest.raises(TripletError, match="processed: 48000 Hz ≠ clean: 16000 Hz"):
        validate_strict(t)


def test_length_and_channel_mismatch(tmp_path):
    t = {"clean": _audio(tmp_path, "c.wav", 1600, 16000),
         "noisy": _audio(tmp_path, "n.wav", 1601, 16000, ch=2)}
    with pytest.raises(TripletError) as e:
        validate_strict(t)
    assert "length" in str(e.value) and "channels" in str(e.value)
