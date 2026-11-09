"""Headless runs of the Streamlit script against stand-in models and a stand-in dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from fakes import make_dataset, make_models

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _fresh(monkeypatch, models: Path, dataset: str):
    monkeypatch.setenv("PLAYER_MODELS_DIR", str(models))
    monkeypatch.setenv("PLAYER_DATASET_DIR", dataset)
    monkeypatch.delenv("VALENTINI_DIR", raising=False)
    for m in [m for m in sys.modules if m == "mplayer" or m.startswith("mplayer.")]:
        monkeypatch.delitem(sys.modules, m)  # re-read the environment on import


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return make_models(tmp_path_factory.mktemp("models"))


def test_dataset_file_runs_every_model(models, tmp_path, monkeypatch):
    _fresh(monkeypatch, models, str(make_dataset(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.radio(key="mode").value == "Dataset file"
    assert at.selectbox(key="split").options == ["Test set", "Train 28spk"]
    assert any("p232_000" in m.value and "9 model outputs" in m.value for m in at.markdown)


def test_loudness_switch_lives_in_the_player_and_next_file(models, tmp_path, monkeypatch):
    _fresh(monkeypatch, models, str(make_dataset(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    # Match loudness is the player's first playback stage (no Streamlit rerun when it is flipped)
    assert len(at.toggle) == 0
    at.button[1].click().run()  # ▶ next file
    assert not at.exception, at.exception
    assert at.selectbox(key="stem").value == "p232_001"


def test_player_gets_original_levels_and_gains(monkeypatch):
    """The player receives every track at its original level plus the gain that matches its loudness."""
    from mplayer import component, loudness
    from fakes import speechlike
    got = {}
    monkeypatch.setattr(component.components, "html", lambda html, **kw: got.update(html=html))
    clean, noisy = speechlike(16000, 2.0)
    quiet = noisy * 0.05
    norm = loudness.normalize({"clean": clean, "noisy": noisy, "quiet": quiet}, 16000)
    import plotly.graph_objects as go
    component.player(go.Figure(), [{"slot": i + 1, "label": k, "description": k, "color": "#fff", "x": x,
                                    "gain_db": norm.gain_db[k]} for i, (k, x) in
                                   enumerate({"clean": clean, "noisy": noisy, "quiet": quiet}.items())],
                     16000, len(noisy), {}, levels=[{"lufs": norm.lufs_in[k], "peak_dbfs": norm.peak_in[k]}
                                                    for k in ("clean", "noisy", "quiet")],
                     format_line="", norm_lufs=norm.achieved_lufs)
    import json, re
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", got["html"], re.S).group(1).replace("<\\/", "</"))
    gains = [t["gain_db"] for t in data["tracks"]]
    assert abs(gains[2] - gains[1] - 20 * np.log10(1 / 0.05)) < 0.05  # quiet copy gets +26 dB more
    lufs_heard = [l["lufs"] + g for l, g in zip(data["levels"], gains)]
    assert np.allclose(lufs_heard, norm.achieved_lufs, atol=0.01)


def test_training_file_gets_a_note(models, tmp_path, monkeypatch):
    _fresh(monkeypatch, models, str(make_dataset(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.selectbox(key="split").set_value("Train 28spk").run()
    assert not at.exception, at.exception
    assert any("may have seen it in training" in c.value and "U-Net 28spk" in c.value for c in at.caption)


def test_without_dataset_asks_for_upload(models, monkeypatch):
    _fresh(monkeypatch, models, "")
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.radio(key="mode").value == "Upload a .wav"
    assert any("Upload a .wav file" in i.value for i in at.info)


def test_loudness_gains_equalise_lufs():
    """The gains the app applies (the viewer's BS.1770 code) bring every track to the same LUFS."""
    from mplayer import CODE_ROOT  # noqa: F401
    from viewer import dsp
    rng = np.random.default_rng(1)
    xs = [0.05 * rng.standard_normal(32000), 0.3 * rng.standard_normal(32000), 0.01 * rng.standard_normal(32000)]
    loud = [dsp.integrated_loudness(x, 16000) for x in xs]
    peaks = [20 * np.log10(np.max(np.abs(x))) for x in xs]
    gains, achieved = dsp.loudness_match_gains(loud, peaks, -20.0)
    after = [dsp.integrated_loudness(x * 10 ** (g / 20), 16000) for x, g in zip(xs, gains)]
    assert np.allclose(after, achieved, atol=0.05)
    assert max(p + g for p, g in zip(peaks, gains)) <= -1.0 + 1e-9
