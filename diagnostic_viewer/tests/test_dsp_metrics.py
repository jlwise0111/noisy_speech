import numpy as np
import pytest

from viewer import dsp
from viewer.demo import make_triplet, scale_to_snr, speech_like
from viewer.metrics import compute_all, si_sdr, snr

SR = 16000


def test_si_sdr_identity_and_scale_invariance():
    x = speech_like(SR, 2.0)
    assert si_sdr(x, x) > 100
    assert si_sdr(0.3 * x, x) > 100  # scale invariant
    assert snr(0.3 * x, x) < 5  # plain SNR is not


def test_si_sdr_zero_db_noise():
    x = speech_like(SR, 3.0)
    white = np.random.default_rng(3).standard_normal(len(x)).astype(np.float32)
    n = scale_to_snr(x, white, 0.0)
    assert si_sdr(x + n, x) == pytest.approx(0.0, abs=0.2)
    assert snr(x + n, x) == pytest.approx(0.0, abs=0.01)


def test_processed_beats_noisy_on_demo():
    clean, noisy, processed = make_triplet(SR, 4.0, snr_db=0.0)
    df = compute_all(clean, noisy, processed, SR).set_index("metric")
    for m in ("SI-SDR (dB)", "PESQ", "STOI"):
        assert df.loc[m, "processed"] > df.loc[m, "noisy"], m
        assert df.loc[m, "Δ (processed − noisy)"] > 0


def test_pesq_resamples_non_native_rate():
    clean, noisy, processed = make_triplet(48000, 3.0)
    df = compute_all(clean, noisy, processed, 48000).set_index("metric")
    assert df.loc["PESQ", "processed"] is not None
    assert "48000→16000" in df.loc["PESQ", "notes"]


def test_pesq_silence_reports_na_instead_of_crashing():
    z = np.zeros(SR * 2, dtype=np.float32)
    df = compute_all(z, z, z, SR).set_index("metric")
    assert df.loc["PESQ", "processed"] is None or np.isnan(df.loc["PESQ", "processed"])
    assert "failed" in df.loc["PESQ", "notes"]


@pytest.mark.parametrize("scale", dsp.SCALES)
def test_spectrogram_shapes_and_decimation(scale):
    x = speech_like(SR, 30.0)
    p = dsp.StftParams(n_fft=1024, hop=128, scale=scale, n_mels=64, max_frames=500, max_bins=200)
    s = dsp.spectrogram(x, SR, p)
    assert s.db.shape == (s.freqs.size, s.times.size)
    assert s.times.size <= 500 and s.freqs.size <= 200
    assert np.all(np.diff(s.freqs) > 0) and np.all(np.diff(s.times) > 0)


def test_sine_lands_in_right_bin():
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    s = dsp.spectrogram(x, SR, dsp.StftParams(n_fft=1024, hop=256))
    peak_f = s.freqs[np.argmax(s.db.mean(axis=1))]
    assert abs(peak_f - 1000) < SR / 1024 * 2


def test_stats_and_envelope():
    x = np.full(SR, 1.0, dtype=np.float32)
    st = dsp.signal_stats(x, SR)
    assert st["clipped_samples"] == SR and st["dc_offset"] == pytest.approx(1.0)
    times, db = dsp.rms_envelope(x, SR)
    assert np.allclose(db, 0.0, atol=1e-6) and times.size == db.size


def test_waveform_envelope_keeps_extremes():
    x = speech_like(SR, 10.0)
    t, y = dsp.waveform_envelope(x, SR, max_points=2000)
    assert y.size <= 2000 and y.max() == x.max() and y.min() == x.min()


def test_noise_floor_tracks_pauses():
    from viewer.metrics import scorecard

    clean, noisy, processed = make_triplet(SR, 4.0, snr_db=0.0)
    st = {k: dsp.signal_stats(x, SR) for k, x in (("clean", clean), ("noisy", noisy), ("proc", processed))}
    for s in st.values():
        assert s["noise_floor_dbfs"] <= s["rms_dbfs"] <= s["max_rms_dbfs"]
    # Denoising should lower the level of the pauses by roughly the 15 dB applied to the noise.
    assert st["noisy"]["noise_floor_dbfs"] - st["proc"]["noise_floor_dbfs"] > 10
    card = scorecard(compute_all(clean, noisy, processed, SR))
    assert [c["label"] for c in card] == ["Separation", "Quality", "Intelligibility"]
    assert all(c["delta"] > 0 for c in card)
