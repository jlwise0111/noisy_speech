"""Headless run of the Streamlit script against the demo data (no browser)."""

from pathlib import Path

import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest

from viewer.demo import make_triplet

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for i in range(2):
        for role, x in zip(("clean", "noisy", "processed"), make_triplet(16000, 2.0, seed=i)):
            p = tmp_path / role / f"s{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, x, 16000)
    sf.write(tmp_path / "odd_clean.wav", make_triplet(16000, 2.0)[0], 16000)
    monkeypatch.setenv("VIEWER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIEWER_EXPORT_DIR", str(tmp_path / "exports"))
    return tmp_path


def test_app_renders_metrics(data_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert not at.error
    df = at.dataframe[0].value
    assert list(df["metric"]) == ["SNR (dB)", "SI-SDR (dB)", "PESQ", "STOI"]


def test_role_swap_and_reorder(data_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    first = at.dataframe[0].value.set_index("metric")
    # Reorder: move slot 1 down. Metrics must not change (they follow roles).
    at.button(key="dn_s0_clean").click().run()
    assert not at.exception
    after = at.dataframe[0].value.set_index("metric")
    assert after.loc["SI-SDR (dB)", "processed"] == pytest.approx(first.loc["SI-SDR (dB)", "processed"])
    # Duplicate a role -> warning, metrics disabled.
    at.selectbox(key="role_s0_noisy").set_value("clean").run()
    assert any("exactly once" in w.value for w in at.warning)


def test_unresolved_triplet_asks_user(data_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    labels = at.selectbox(key="tri_idx").options
    odd = next(i for i, l in enumerate(labels) if "odd" in l)
    at.selectbox(key="tri_idx").set_value(odd).run()  # value = triplet index
    assert any("missing noisy" in w.value for w in at.warning)


@pytest.mark.parametrize("scale", ["log", "mel"])
def test_frequency_scales(data_dir, scale):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="scale").set_value(scale).run()
    assert not at.exception, at.exception


def test_figure_export_png_svg():
    from viewer import dsp, export, figures

    c, n, p = make_triplet(16000, 1.0)
    params = dsp.StftParams(scale="mel")
    tracks = [figures.TrackView(i + 1, r, f"{r}.wav", x, 16000, dsp.spectrogram(x, 16000, params))
              for i, (r, x) in enumerate(zip(("clean", "noisy", "processed"), (c, n, p)))]
    fig = figures.stacked_figure(tracks, (-110, -20), monitored_slot=0)
    assert export.figure_image(fig, "png", 800, 1)[:4] == b"\x89PNG"
    assert b"<svg" in export.figure_image(fig, "svg", 800)[:500]


def test_strict_mismatch_shows_error(tmp_path, monkeypatch):
    c, n, p = make_triplet(16000, 1.0)
    for role, x, sr in (("clean", c, 16000), ("noisy", n, 16000), ("processed", p, 48000)):
        (tmp_path / role).mkdir()
        sf.write(tmp_path / role / "m.wav", x, sr)
    monkeypatch.setenv("VIEWER_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("processed: 48000 Hz ≠ clean: 16000 Hz" in e.value for e in at.error)
    assert len(at.dataframe) == 0  # nothing rendered


def test_stacked_figure_overlays_waveform_per_track():
    from viewer import dsp, figures

    sig = make_triplet(16000, 1.0)
    tracks = [figures.TrackView(i + 1, r, f"{r}.wav", x, 16000, dsp.spectrogram(x, 16000, dsp.StftParams()))
              for i, (r, x) in enumerate(zip(("clean", "noisy", "processed"), sig))]
    fig = figures.stacked_figure(tracks, (-110, -20), wave_opacity=0.5, monitored_slot=2)
    kinds = [(tr.type, tr.meta["kind"], tr.meta["slot"]) for tr in fig.data]
    assert kinds == [(t, k, s) for s in (1, 2, 3) for t, k in (("heatmap", "spec"), ("scatter", "wave"))]
    # Each waveform sits on its own (hidden) axis overlaying its spectrogram's axis.
    for heat, wave in zip(fig.data[0::2], fig.data[1::2]):
        assert heat.xaxis == wave.xaxis and heat.yaxis != wave.yaxis
        assert fig.layout[wave.yaxis.replace("y", "yaxis")].overlaying == heat.yaxis
    labels = [a for a in fig.layout.annotations if a.name and a.name.startswith("label")]
    assert len(labels) == 3 and "🎧" in labels[1].text and "🎧" not in labels[0].text
    # Opacity 0 drops the overlay entirely.
    assert all(tr.type == "heatmap" for tr in figures.stacked_figure(tracks, (-110, -20), wave_opacity=0).data)
