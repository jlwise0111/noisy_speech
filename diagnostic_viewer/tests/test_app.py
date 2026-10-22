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


def test_fixed_display_no_settings_menus(data_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert not at.get("popover")  # no Tracks / Display menus: the display settings are fixed
    assert [e.label for e in at.main.expander] == ["Analysis"]
    assert [t.label for t in at.tabs] == ["Long-term spectrum", "Loudness over time", "All metrics"]


def test_unresolved_triplet_asks_user(data_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    labels = at.selectbox(key="tri_idx").options
    odd = next(i for i, l in enumerate(labels) if "odd" in l)
    at.selectbox(key="tri_idx").set_value(odd).run()  # value = triplet index
    assert any("missing noisy" in w.value for w in at.warning)


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


@pytest.fixture
def systems_dir(tmp_path, monkeypatch):
    """Two systems with the same files; the Wiener run is much quieter."""
    layout = {"linear_baseline/outputs/ridge_a": 1.0, "wiener_baseline/outputs/wiener_a": 0.05}
    for run, gain in layout.items():
        for i in range(2):
            c, n, p = make_triplet(16000, 2.0, seed=i)
            for role, x in (("clean", c), ("noisy", n), ("enhanced", p * gain)):
                f = tmp_path / run / role / f"p232_00{i}.wav"
                f.parent.mkdir(parents=True, exist_ok=True)
                sf.write(f, x, 16000)
    monkeypatch.setenv("VIEWER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIEWER_EXPORT_DIR", str(tmp_path / "exports"))
    return tmp_path


def test_system_switch_keeps_file(systems_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="system").value == "Wiener baseline"  # first system with outputs
    next(b for b in at.button if b.label == "▶").click().run()  # p232_000 -> p232_001
    assert any("p232_001" in m.value and "Wiener baseline" in m.value for m in at.markdown)
    at.selectbox(key="system").set_value("Ridge baseline").run()
    assert not at.exception, at.exception
    assert at.selectbox(key="run").value == "linear_baseline/outputs/ridge_a"
    assert any("p232_001" in m.value and "Ridge baseline" in m.value for m in at.markdown)


def test_neural_network_without_outputs_explains_layout(systems_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="system").set_value("Neural network").run()
    assert not at.exception, at.exception
    assert any("no outputs found yet" in i.value for i in at.info)
    at.selectbox(key="system").set_value("Ridge baseline").run()  # comes back with a file selected
    assert not at.exception and len(at.dataframe) == 1


def test_loudness_match_toggle(systems_dir):
    at = AppTest.from_file(APP, default_timeout=120).run()
    before = at.dataframe[0].value.set_index("metric")
    at.toggle(key="match_lufs").set_value(True).run()
    assert not at.exception, at.exception
    assert any("Loudness matched" in c.value and "LUFS" in c.value for c in at.caption)
    assert "−20 LUFS" in at.toggle(key="match_lufs").help or "-20 LUFS" in at.toggle(key="match_lufs").help
    after = at.dataframe[0].value.set_index("metric")
    assert after.loc["PESQ", "processed"] == before.loc["PESQ", "processed"]  # scores use the files
