import numpy as np
import soundfile as sf

from viewer.dataset import discover, list_wavs


def _wav(p, n=1600, sr=16000):
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, np.zeros(n, dtype=np.float32), sr)


def test_subfolder_layout(tmp_path):
    for role in ("clean", "noisy", "enhanced"):
        _wav(tmp_path / role / "a.wav")
    (t,) = discover(tmp_path)
    assert t.resolved and t.stem == "a" and t.layout == "subfolders"
    assert t.files["processed"].parent.name == "enhanced"


def test_suffix_layout(tmp_path):
    for role in ("clean", "noisy", "processed"):
        _wav(tmp_path / "sub" / f"utt1_{role}.wav")
    (t,) = discover(tmp_path, layout="suffix")
    assert t.resolved and t.stem == "sub/utt1"


def test_missing_and_ambiguous(tmp_path):
    _wav(tmp_path / "x_clean.wav")
    _wav(tmp_path / "x_noisy.wav")
    _wav(tmp_path / "y_clean.wav")
    _wav(tmp_path / "y_noisy.wav")
    _wav(tmp_path / "y_processed.wav")
    _wav(tmp_path / "y_enhanced.wav")
    ts = {t.stem: t for t in discover(tmp_path)}
    assert not ts["x"].resolved and ts["x"].issues == ["missing processed"]
    assert not ts["y"].resolved and ts["y"].issues == ["2 candidates for processed"]
    assert ts["y"].files["processed"] is None
    assert len(list_wavs(tmp_path)) == 6


def test_resolved_sorted_first(tmp_path):
    _wav(tmp_path / "a_clean.wav")
    for role in ("clean", "noisy", "processed"):
        _wav(tmp_path / f"b_{role}.wav")
    assert [t.stem for t in discover(tmp_path)] == ["b", "a"]


def test_missing_root(tmp_path):
    assert discover(tmp_path / "nope") == []


def test_nested_subfolder_layout(tmp_path):
    for role in ("clean", "noisy", "processed"):
        _wav(tmp_path / "exp1" / role / "a.wav")
    (t,) = discover(tmp_path)
    assert t.resolved and t.stem == "exp1/a"


def test_hidden_and_cache_folders_are_skipped(tmp_path):
    for role in ("clean", "noisy", "processed"):
        _wav(tmp_path / "run" / role / "a.wav")
        _wav(tmp_path / ".venv" / "run" / role / "b.wav")  # e.g. scipy's test wavs in a virtualenv
        _wav(tmp_path / "__pycache__" / role / "c.wav")
    (tmp_path / "run" / "clean" / "._a.wav").write_bytes(b"macOS resource fork")
    assert [t.stem for t in discover(tmp_path)] == ["run/a"]
    assert len(list_wavs(tmp_path)) == 3
