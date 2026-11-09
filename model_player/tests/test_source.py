"""Uploads become 16 kHz mono; dataset splits pair noisy with clean."""
import io

import numpy as np
import soundfile as sf

from mplayer import source
from fakes import make_dataset


def test_stereo_44k_upload_becomes_mono_16k():
    t = np.arange(44100 * 2) / 44100
    x = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.1 * np.sin(2 * np.pi * 440 * t)], axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, 44100, format="WAV", subtype="PCM_24")
    y, file_sr = source.decode_upload(buf.getvalue())
    assert file_sr == 44100
    assert y.ndim == 1 and y.dtype == np.float32
    assert abs(len(y) - 32000) <= 1
    assert abs(np.max(np.abs(y[1000:-1000])) - 0.3) < 0.01  # channel average


def test_cap_length():
    y, cut = source.cap_length(np.zeros(16000 * 70, np.float32))
    assert cut and len(y) == 16000 * 60
    y, cut = source.cap_length(np.zeros(100, np.float32))
    assert not cut and len(y) == 100


def test_dataset_splits_and_pairs(tmp_path):
    root = make_dataset(tmp_path)
    assert source.available_splits(root) == ["Test set", "Train 28spk"]
    pairs = source.list_split(root, "Test set")
    assert [p[0] for p in pairs] == ["p232_000", "p232_001"]
    clean, noisy = source.load_pair(pairs[0][1], pairs[0][2])
    assert len(clean) == len(noisy) == 24000  # 1.5 s at 48 kHz -> 16 kHz
    assert source.list_split(root, "Train 56spk") == []


def test_no_dataset():
    assert source.available_splits("") == []
    assert source.available_splits("/does/not/exist") == []
