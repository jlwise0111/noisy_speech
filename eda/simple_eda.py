#!/usr/bin/env python3
"""
simple_eda.py  (patched)
========================
Shared, lightweight EDA for our speech-denoising project (Valentini clean/noisy pairs).

Sections
  1. Overview         pairs, speakers, hours, native sample rates, split/speaker overlap
  2. Data checks      length mismatch, clipping, negative SNR, clean/noisy ALIGNMENT + GAIN
  3. Difficulty       global SNR, speech-frame SNR, log-spectral distance (LSD)
  4. Noise character  spectral centroid / flatness / stationarity of the added noise
  5. Frequency        average clean vs noise spectrum and per-frequency SNR
  6. Mask targets     magnitude-ratio mask: fraction CLIPPED, histograms for speech vs
                      silent STFT frames separately, phase-sensitive mask (PSM)
  7. Noisy phase      energy-weighted phase error, and oracle reconstructions
                      (clean mag + noisy phase, clipped-mask * noisy) = ceilings for
                      any magnitude-only mask model
  8. Silence          silent-frame fraction at 20/30/40 dB below peak, on the STFT grid
                      (the same grid a loss weight would use)
  9. Speakers         per-speaker table (for choosing validation speakers) and
                      train-subset speaker overlap (28spk vs 56spk leakage check)
 10. Examples         clean / noisy / noise / mask spectrograms at low, median, high SNR
 11. (optional)       PESQ-WB / STOI of noisy-vs-clean and of the oracles (--metrics)

Audio is resampled to 16 kHz with the same STFT and 30 dB silence threshold as the U-Net training
(n_fft=512, hop=128; unet_model/unet/targets.py), so these figures describe what the model sees.
The noise signal is noisy - clean, which is only valid if the pair is sample-aligned and
equally scaled, so section 2 checks that explicitly.

Requires: numpy scipy pandas soundfile librosa matplotlib joblib
Optional: pesq pystoi            (for --metrics)

  python simple_eda.py                       # 200 pairs per subset -> simple_eda_output/
  python simple_eda.py --per-subset 0        # every pair (use this for report numbers)
  python simple_eda.py --metrics             # also PESQ/STOI (slow)
  python simple_eda.py --data-root /path/to/valentini --show
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import correlate

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "valentini_eda"   # repo root, wherever the script is run from

SR = 16000
N_FFT, HOP = 512, 128                   # same STFT as speechprep / the U-Net
FRAME, FRAME_HOP = 320, 160             # 20 ms frames, 10 ms hop (used for SNR measures only)
EPS = 1e-10
SILENCE_THRESHOLDS_DB = (20, 30, 40)    # "silent" = frame energy this far below the utterance peak
MAIN_SILENCE_DB = 30                    # threshold for the speech/silent histograms (= SILENCE_DB in training)
MAX_LAG_S = 0.02                        # alignment search window (+/- 20 ms)
IRM_BINS = np.linspace(0.0, 1.0, 41)
FREQS = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
SUBSET_COLORS = {"testset": "#2a78d6", "trainset_28spk": "#eb6834", "trainset_56spk": "#1baf7a"}
# Valentini nominal SNRs: train 0/5/10/15 dB, test 2.5/7.5/12.5/17.5 dB
SNR_BIN_EDGES = [-np.inf, 2.5, 7.5, 12.5, np.inf]
SNR_BIN_LABELS = ["<=2.5", "2.5-7.5", "7.5-12.5", ">12.5"]


def db(power):
    """Power to dB, floored at EPS."""
    return 10.0 * np.log10(np.maximum(power, EPS))


def frame_power(x):
    """Mean power per 20 ms frame (10 ms hop), zero-padding clips shorter than one frame."""
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    return np.mean(np.lib.stride_tricks.sliding_window_view(x, FRAME)[::FRAME_HOP] ** 2, axis=1)


def snr_of(ref, est):
    """SNR (dB) of an estimate against a reference, over the common length."""
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return float(db(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + EPS)))


# --------------------------------------------------------------------------- data
def discover_pairs(root):
    """Pair clean_<subset>_wav/<name>.wav with noisy_<subset>_wav/<name>.wav."""
    pairs, unmatched = [], 0
    for clean_dir in sorted(root.glob("clean_*_wav")):
        subset = clean_dir.name[len("clean_"):-len("_wav")]
        noisy_dir = root / f"noisy_{subset}_wav"
        if not noisy_dir.is_dir():
            continue
        clean = {p.name for p in clean_dir.glob("*.wav")}
        noisy = {p.name for p in noisy_dir.glob("*.wav")}
        unmatched += len(clean ^ noisy)
        for name in sorted(clean & noisy):
            pairs.append(dict(pair_id=f"{subset}/{Path(name).stem}", subset=subset,
                              speaker=name.split("_")[0], clean=clean_dir / name, noisy=noisy_dir / name))
    return pairs, unmatched


def sample_pairs(pairs, per_subset, seed=0):
    """Up to `per_subset` random pairs from each subset (all pairs if per_subset <= 0)."""
    if per_subset <= 0:
        return pairs
    rng = random.Random(seed)
    out = []
    for s in sorted({p["subset"] for p in pairs}):
        group = [p for p in pairs if p["subset"] == s]
        out += rng.sample(group, min(per_subset, len(group)))
    return out


def alignment(c, y, sr):
    """Lag (samples, native rate) maximising clean/noisy correlation, plus LS gain of clean in noisy.

    Positive lag = noisy is delayed relative to clean. lag != 0 or gain far from 1 means
    noisy - clean is NOT purely the added noise and the SNR/noise EDA is inflated.
    """
    n = min(len(c), len(y))
    c, y = c[:n].astype(np.float64), y[:n].astype(np.float64)
    max_lag = int(MAX_LAG_S * sr)
    corr = correlate(y, c, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= max_lag
    lag = int(lags[keep][np.argmax(corr[keep])])
    gain = float(np.dot(y, c) / (np.dot(c, c) + EPS))
    return lag, gain


def load_pair(p):
    """Return clean, noisy and noise at 16 kHz, plus native-rate info and alignment diagnostics."""
    c, sr_c = sf.read(p["clean"], dtype="float32")
    y, sr_y = sf.read(p["noisy"], dtype="float32")
    if c.ndim > 1:
        c = c.mean(axis=1)
    if y.ndim > 1:
        y = y.mean(axis=1)
    lag, gain = alignment(c, y, sr_c)
    info = dict(native_sr=sr_c, sr_mismatch=sr_c != sr_y, len_diff_samples=len(y) - len(c),
                clean_peak=float(np.max(np.abs(c))), noisy_peak=float(np.max(np.abs(y))),
                align_lag_samples=lag, align_lag_ms=1000.0 * lag / sr_c, ls_gain=gain)
    c = librosa.resample(c, orig_sr=sr_c, target_sr=SR)
    y = librosa.resample(y, orig_sr=sr_y, target_sr=SR)
    n = min(len(c), len(y))
    c, y = c[:n], y[:n]
    return c, y, y - c, info


def objective(ref, deg):
    """PESQ-WB and STOI of deg against ref (NaN if the libraries are missing or a call fails)."""
    out = dict(pesq=np.nan, stoi=np.nan)
    ref, deg = ref.astype(np.float64), deg.astype(np.float64)
    try:
        from pesq import pesq
        out["pesq"] = float(pesq(SR, ref, deg, "wb"))
    except Exception:
        pass
    try:
        from pystoi import stoi
        out["stoi"] = float(stoi(ref, deg, SR, extended=False))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- features
def analyze_pair(p, metrics=False):
    """Every per-pair measure (SNR, noise character, masks, phase, silence, oracles) as one row, plus spectra for the plots."""
    c, y, r, info = load_pair(p)
    fc, fr = frame_power(c), frame_power(r)
    fc_db, fr_db = db(fc), db(fr)
    speech = (fc_db > fc_db.max() - 40) & (fc_db > -60)      # 10 ms-grid VAD, used for SNR only
    audible = fr_db > -90

    # complex STFTs: magnitudes for masks, phases for the noisy-phase analysis
    Cc = librosa.stft(c, n_fft=N_FFT, hop_length=HOP)
    Yc = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    Rc = librosa.stft(r, n_fft=N_FFT, hop_length=HOP)
    C, Y, R = np.abs(Cc), np.abs(Yc), np.abs(Rc)

    # ---- mask targets (same definition as mini_eda.py) and how much clip() hides
    ratio = C / (Y + 1e-8)
    irm = np.clip(ratio, 0.0, 1.0)
    clip_frac = float((ratio > 1.0).mean())

    # ---- noisy-phase analysis
    dphi = np.angle(Cc) - np.angle(Yc)
    w = C ** 2
    phase_cos = float((w * np.cos(dphi)).sum() / (w.sum() + EPS))   # 1.0 = noisy phase is fine
    psm = np.clip(ratio * np.cos(dphi), 0.0, 1.0)                    # phase-sensitive mask

    # ---- silence on the STFT frame grid (what a loss weight would use)
    frame_db = db((C ** 2).sum(axis=0))
    silent_fracs = {f"silent_frac_{t}": float((frame_db < frame_db.max() - t).mean())
                    for t in SILENCE_THRESHOLDS_DB}
    silent = frame_db < frame_db.max() - MAIN_SILENCE_DB

    active = (C ** 2).sum(axis=0) > (C ** 2).sum(axis=0).max() * 1e-5
    lsd = np.sqrt(np.mean((db(C ** 2) - db(Y ** 2)) ** 2, axis=0))[active].mean()

    # ---- oracle reconstructions: ceilings for any magnitude-only mask model
    s_cleanmag = librosa.istft(C * np.exp(1j * np.angle(Yc)), n_fft=N_FFT, hop_length=HOP, length=len(c))
    s_irm = librosa.istft(irm * Yc, n_fft=N_FFT, hop_length=HOP, length=len(c))

    noise_ltas = (R ** 2).mean(axis=1)
    row = dict(
        pair_id=p["pair_id"], subset=p["subset"], speaker=p["speaker"],
        duration_s=len(c) / SR, **info,
        clean_rms_dbfs=float(db(np.mean(c ** 2))),
        noisy_rms_dbfs=float(db(np.mean(y ** 2))),
        noise_rms_dbfs=float(db(np.mean(r ** 2))),
        speech_ratio_10ms=float(speech.mean()),
        snr_db=float(db(np.sum(c ** 2) / (np.sum(r ** 2) + EPS))),
        snr_speech_db=float(db(fc[speech].sum() / (fr[speech].sum() + EPS))),
        lsd_db=float(lsd),
        noise_centroid_hz=float((FREQS * noise_ltas).sum() / (noise_ltas.sum() + EPS)),
        noise_flatness=float(librosa.feature.spectral_flatness(S=R).mean()),
        noise_stationarity_db=float(fr_db[audible].std()) if audible.any() else np.nan,
        irm_mean=float(irm.mean()),
        irm_frac_low=float((irm < 0.2).mean()),               # mostly noise
        irm_frac_high=float((irm > 0.8).mean()),              # mostly speech
        irm_clip_frac=clip_frac,                              # bins where |S| > |Y| (clipped to 1)
        phase_cos=phase_cos,
        psm_mean=float(psm.mean()),
        oracle_cleanmag_snr_db=snr_of(c, s_cleanmag),         # ceiling: perfect magnitude, noisy phase
        oracle_irm_snr_db=snr_of(c, s_irm),                   # ceiling: perfect clipped mask, noisy phase
        **silent_fracs,
    )
    if metrics:
        for name, sig in (("noisy", y), ("oracle_cleanmag", s_cleanmag), ("oracle_irm", s_irm)):
            m = objective(c, sig)
            row[f"{name}_pesq"], row[f"{name}_stoi"] = m["pesq"], m["stoi"]

    extras = dict(
        irm_hist_speech=np.histogram(irm[:, ~silent], bins=IRM_BINS)[0],
        irm_hist_silent=np.histogram(irm[:, silent], bins=IRM_BINS)[0],
        clean_ltas=(C ** 2).mean(axis=1), noise_ltas=noise_ltas)
    return row, extras


# --------------------------------------------------------------------------- plots
def finish(fig, title, path, show):
    """Title, save and close (or show) a figure."""
    fig.suptitle(title, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  figure -> {path.name}")
    plt.show() if show else plt.close(fig)


def subset_hists(df, cols, title, path, show):
    """One histogram per column, one line per subset."""
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.6), squeeze=False)
    for ax, (col, label) in zip(axes[0], cols):
        bins = np.histogram_bin_edges(df[col].dropna(), bins=40)
        for s, g in df.groupby("subset"):
            ax.hist(g[col].dropna(), bins=bins, density=True, histtype="step", lw=2,
                    color=SUBSET_COLORS.get(s), label=s)
        ax.set(title=label, xlabel=col)
    axes[0, 0].legend(frameon=False, fontsize=8)
    finish(fig, title, path, show)


def plot_noise_map(df, path, show):
    """Noise character: spectral centroid vs flatness, coloured by stationarity."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sc = axes[0].scatter(df["noise_centroid_hz"], df["noise_flatness"], c=df["noise_stationarity_db"],
                         cmap="viridis", s=10, alpha=0.8, edgecolors="none")
    fig.colorbar(sc, ax=axes[0], label="noise stationarity (dB std, low = steady)")
    axes[0].set(xlabel="noise spectral centroid (Hz)", ylabel="noise spectral flatness",
                title="Noise-type map (clusters ~ DEMAND noise types)")
    for s, g in df.groupby("subset"):
        axes[1].scatter(g["snr_db"], g["lsd_db"], s=10, alpha=0.6, color=SUBSET_COLORS.get(s), label=s)
    axes[1].set(xlabel="global SNR (dB)", ylabel="log-spectral distance (dB)",
                title="SNR vs spectral damage")
    axes[1].legend(frameon=False, fontsize=8)
    finish(fig, "Noise character", path, show)


def plot_spectra(df, extras, path, show):
    """Average clean and noise spectra per subset, and the per-frequency SNR."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for s in sorted(df["subset"].unique()):
        idx = np.flatnonzero(df["subset"].to_numpy() == s)
        clean = np.mean([extras[i]["clean_ltas"] for i in idx], axis=0)
        noise = np.mean([extras[i]["noise_ltas"] for i in idx], axis=0)
        col = SUBSET_COLORS.get(s)
        axes[0].plot(FREQS, db(clean), color=col, lw=2, label=f"{s} clean")
        axes[0].plot(FREQS, db(noise), color=col, lw=1.5, ls="--", label=f"{s} noise")
        axes[1].plot(FREQS, db(clean) - db(noise), color=col, lw=2, label=s)
    axes[0].set(xlabel="frequency (Hz)", ylabel="mean power (dB)", title="Average spectrum: speech vs noise")
    axes[1].axhline(0, color="gray", lw=1, ls=":")
    axes[1].set(xlabel="frequency (Hz)", ylabel="SNR (dB)",
                title="Per-frequency SNR (low = where the model works hardest)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    finish(fig, "Where in frequency is the noise?", path, show)


def plot_irm(df, extras, path, show):
    """Mask histograms split into speech frames and silent frames, plus mean mask vs SNR."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    centers = (IRM_BINS[:-1] + IRM_BINS[1:]) / 2
    for ax, key, name in ((axes[0], "irm_hist_speech", "speech frames"),
                          (axes[1], "irm_hist_silent", f"silent frames (>{MAIN_SILENCE_DB} dB below peak)")):
        for s in sorted(df["subset"].unique()):
            idx = np.flatnonzero(df["subset"].to_numpy() == s)
            h = np.sum([extras[i][key] for i in idx], axis=0)
            if h.sum() > 0:
                ax.step(centers, h / h.sum(), where="mid", lw=2, color=SUBSET_COLORS.get(s), label=s)
        ax.set(xlabel="mask value (0 = noise, 1 = speech; last bin includes clipped bins)",
               ylabel="fraction of T-F bins", title=f"Mask distribution: {name}", yscale="log")
        ax.legend(frameon=False, fontsize=8)
    for s, g in df.groupby("subset"):
        axes[2].scatter(g["snr_db"], g["irm_mean"], s=10, alpha=0.6, color=SUBSET_COLORS.get(s), label=s)
    axes[2].set(xlabel="global SNR (dB)", ylabel="mean mask", title="Noisier pairs need more suppression")
    finish(fig, "Mask targets", path, show)


def plot_phase(df, path, show):
    """Noisy-phase quality and mask clipping vs SNR, and the oracle (clean magnitude + noisy phase) ceiling."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for s, g in df.groupby("subset"):
        col = SUBSET_COLORS.get(s)
        axes[0].scatter(g["snr_db"], g["phase_cos"], s=10, alpha=0.6, color=col, label=s)
        axes[1].scatter(g["snr_db"], 100 * g["irm_clip_frac"], s=10, alpha=0.6, color=col, label=s)
        axes[2].scatter(g["snr_db"], g["oracle_cleanmag_snr_db"], s=10, alpha=0.6, color=col, label=s)
    axes[0].set(xlabel="global SNR (dB)", ylabel="energy-weighted cos(phase error)",
                title="Noisy-phase quality (1.0 = noisy phase is fine)")
    axes[1].set(xlabel="global SNR (dB)", ylabel="% of T-F bins with |S| > |Y|",
                title="How much clip(., 0, 1) hides")
    lo, hi = df["snr_db"].min(), df["snr_db"].max()
    axes[2].plot([lo, hi], [lo, hi], color="gray", ls=":", label="input SNR")
    axes[2].set(xlabel="input SNR (dB)", ylabel="output SNR (dB)",
                title="Oracle: clean magnitude + noisy phase (ceiling)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    finish(fig, "Noisy phase and mask clipping", path, show)


def plot_silence(df, path, show):
    """Silent-frame fraction at each threshold, and its spread per utterance at the main threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = [f"silent_frac_{t}" for t in SILENCE_THRESHOLDS_DB]
    for s, g in df.groupby("subset"):
        axes[0].plot(SILENCE_THRESHOLDS_DB, [100 * g[c].mean() for c in cols], "o-", lw=2,
                     color=SUBSET_COLORS.get(s), label=s)
    axes[0].set(xlabel="threshold (dB below utterance peak)", ylabel="% silent STFT frames",
                title="Silence fraction depends on the threshold")
    axes[0].legend(frameon=False, fontsize=8)
    for s, g in df.groupby("subset"):
        axes[1].hist(100 * g[f"silent_frac_{MAIN_SILENCE_DB}"], bins=30, density=True, histtype="step",
                     lw=2, color=SUBSET_COLORS.get(s), label=s)
    axes[1].set(xlabel=f"% silent frames per utterance ({MAIN_SILENCE_DB} dB threshold)",
                title="Per-utterance spread")
    finish(fig, "Silence (STFT frame grid)", path, show)


def plot_examples(df, pairs_by_id, path, show):
    """Clean / noisy / noise / mask spectrograms for the lowest, median and highest SNR pairs."""
    s = df.sort_values("snr_db")
    picks = [("lowest SNR", s.iloc[0]), ("median SNR", s.iloc[len(s) // 2]), ("highest SNR", s.iloc[-1])]
    fig, axes = plt.subplots(len(picks), 4, figsize=(17, 3.0 * len(picks)), squeeze=False)
    for i, (label, row) in enumerate(picks):
        c, y, r, _ = load_pair(pairs_by_id[row["pair_id"]])
        C, Y, R = (np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) for x in (c, y, r))
        ref = Y.max()
        panels = [("clean", librosa.amplitude_to_db(C, ref=ref), "magma"),
                  ("noisy", librosa.amplitude_to_db(Y, ref=ref), "magma"),
                  ("noise (noisy - clean)", librosa.amplitude_to_db(R, ref=ref), "magma"),
                  ("mask target", np.clip(C / (Y + 1e-8), 0, 1), "gray")]
        for j, (name, img, cmap) in enumerate(panels):
            ax = axes[i, j]
            kw = dict(vmin=0, vmax=1) if name == "mask target" else dict(vmin=-80, vmax=0)
            librosa.display.specshow(img, sr=SR, hop_length=HOP, x_axis="time", y_axis="hz",
                                     cmap=cmap, ax=ax, **kw)
            ax.set_title(f"{name}" + (f"\n{row['pair_id']} | {label}, SNR {row['snr_db']:.1f} dB"
                                      if j == 0 else ""), fontsize=9)
            if j:
                ax.set_ylabel("")
    finish(fig, "Example pairs", path, show)


# --------------------------------------------------------------------------- report
def speaker_table(df):
    """Per-speaker summary, for choosing validation speakers."""
    return df.groupby(["subset", "speaker"]).agg(
        utterances=("pair_id", "size"),
        minutes=("duration_s", lambda d: d.sum() / 60),
        snr_median=("snr_db", "median"),
        snr_speech_median=("snr_speech_db", "median"),
        silent_frac=(f"silent_frac_{MAIN_SILENCE_DB}", "mean"),
        noise_centroid_hz=("noise_centroid_hz", "median"),
        noise_stationarity_db=("noise_stationarity_db", "median"),
        phase_cos=("phase_cos", "mean"),
    ).round(2)


def print_report(df, all_pairs, unmatched, metrics):
    """Print the numbered report sections (overview, data checks, silence, clipping and phase, speakers)."""
    print("\n=== 1. Overview (sampled pairs) ===")
    overview = df.groupby("subset").agg(
        pairs=("pair_id", "size"), speakers=("speaker", "nunique"),
        minutes=("duration_s", lambda d: d.sum() / 60), mean_dur_s=("duration_s", "mean"),
        snr_median=("snr_db", "median"), snr_speech_median=("snr_speech_db", "median"),
        lsd_median=("lsd_db", "median"), irm_mean=("irm_mean", "mean"),
    ).round(2)
    print(overview.to_string())
    print("native sample rates:", df["native_sr"].value_counts().to_dict())

    full = pd.DataFrame(all_pairs)
    print(f"total pairs on disk: {len(full)} {full['subset'].value_counts().sort_index().to_dict()}"
          f"  |  unmatched files: {unmatched}")

    # speaker overlap across ALL subsets (all pairs on disk, not just the sample)
    spk = {s: set(g["speaker"]) for s, g in full.groupby("subset")}
    subsets = sorted(spk)
    print("\nspeaker overlap between subsets (leakage check):")
    for i, a in enumerate(subsets):
        for b in subsets[i + 1:]:
            ov = sorted(spk[a] & spk[b])
            note = "none" if not ov else f"{len(ov)} shared: {ov[:8]}{' ...' if len(ov) > 8 else ''}"
            print(f"  {a} vs {b}: {note}")
    print("  (do not train on one train subset and validate on another if they share speakers)")

    print("\n=== 2. Data checks ===")
    checks = {
        "sample-rate mismatch": df["sr_mismatch"],
        "clean/noisy length differs": df["len_diff_samples"] != 0,
        "clean clipping (peak >= 0.999)": df["clean_peak"] >= 0.999,
        "noisy clipping (peak >= 0.999)": df["noisy_peak"] >= 0.999,
        "negative SNR": df["snr_db"] < 0,
        f"silent frames > 50% ({MAIN_SILENCE_DB} dB)": df[f"silent_frac_{MAIN_SILENCE_DB}"] > 0.5,
        "alignment lag != 0 (noise=noisy-clean suspect)": df["align_lag_samples"] != 0,
        "clean gain in noisy outside [0.95, 1.05]": (df["ls_gain"] < 0.95) | (df["ls_gain"] > 1.05),
        "mask clipped in > 5% of bins": df["irm_clip_frac"] > 0.05,
    }
    for name, mask in checks.items():
        print(f"  {name:<48} {int(mask.sum()):>6}  ({100 * mask.mean():.1f}%)")
    print(f"  alignment lag range: {df['align_lag_ms'].min():.2f} to {df['align_lag_ms'].max():.2f} ms"
          f" | LS gain median {df['ls_gain'].median():.3f}")

    q = df["snr_speech_db"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).tolist()
    print(f"\nspeech-frame SNR quantiles (5/25/50/75/95): {q}")
    print("  (Valentini labels: train 0/5/10/15 dB, test 2.5/7.5/12.5/17.5 dB)")
    snr_rng = df.groupby("subset")["snr_db"].agg(["min", "median", "max"]).round(1)
    print("global SNR range by subset (train vs test mismatch = generalisation evidence):")
    print(snr_rng.to_string())

    print("\n=== 3. Silence on the STFT frame grid (mean % silent frames) ===")
    sil = (100 * df.groupby("subset")[[f"silent_frac_{t}" for t in SILENCE_THRESHOLDS_DB]].mean()).round(1)
    sil.columns = [f"{t} dB below peak" for t in SILENCE_THRESHOLDS_DB]
    print(sil.to_string())
    print("  (state which threshold produced the silence figure quoted in the report)")

    print("\n=== 4. Mask clipping and noisy phase (by nominal SNR bin) ===")
    binned = df.assign(snr_bin=pd.cut(df["snr_speech_db"], SNR_BIN_EDGES, labels=SNR_BIN_LABELS))
    phase = binned.groupby(["subset", "snr_bin"], observed=True).agg(
        n=("pair_id", "size"),
        clip_pct=("irm_clip_frac", lambda x: 100 * x.mean()),
        phase_cos=("phase_cos", "mean"),
        mask_mean=("irm_mean", "mean"),
        psm_mean=("psm_mean", "mean"),
        oracle_cleanmag_snr=("oracle_cleanmag_snr_db", "mean"),
        oracle_irm_snr=("oracle_irm_snr_db", "mean"),
        input_snr=("snr_db", "mean"),
    ).round(2)
    print(phase.to_string())

    if metrics:
        print("\n=== 5. PESQ-WB / STOI vs clean (mean by subset and SNR bin) ===")
        cols = [c for c in df.columns if c.endswith("_pesq") or c.endswith("_stoi")]
        print(binned.groupby(["subset", "snr_bin"], observed=True)[cols].mean().round(3).to_string())
        print("  noisy_* = unprocessed baseline; oracle_cleanmag = ceiling for ANY magnitude-only model;"
              " oracle_irm = ceiling with this clipped mask")
        print("  NaN means the pesq/pystoi package is missing or a call failed")
    return overview


def main():
    """Analyse the sampled pairs in parallel, then write the tables and figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT,
                    help="folder with clean_*_wav / noisy_*_wav subfolders")
    ap.add_argument("--out-dir", type=Path, default=HERE / "simple_eda_output")
    ap.add_argument("--per-subset", type=int, default=200, help="pairs sampled per subset (0 = all)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--metrics", action="store_true",
                    help="also compute PESQ-WB/STOI for noisy and oracle signals (needs pesq, pystoi; slow)")
    ap.add_argument("--show", action="store_true", help="also display figures interactively")
    args = ap.parse_args()

    if not args.show:
        plt.switch_backend("Agg")

    all_pairs, unmatched = discover_pairs(args.data_root)
    if not all_pairs:
        raise SystemExit(f"no clean/noisy pairs found under {args.data_root}")
    pairs = sample_pairs(all_pairs, args.per_subset)
    print(f"data root: {args.data_root}\nanalyzing {len(pairs)} of {len(all_pairs)} pairs ...")

    results = Parallel(n_jobs=args.jobs)(delayed(analyze_pair)(p, args.metrics) for p in pairs)
    df = pd.DataFrame([r for r, _ in results])
    extras = [e for _, e in results]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overview = print_report(df, all_pairs, unmatched, args.metrics)
    spk_tbl = speaker_table(df)
    print("\n=== Per-speaker table (use to choose validation speakers) ===")
    print(spk_tbl.to_string())
    if args.per_subset > 0:
        print("  (computed from the sample; run with --per-subset 0 for full counts)")

    df.to_csv(args.out_dir / "pairs.csv", index=False)
    overview.to_csv(args.out_dir / "summary_by_subset.csv")
    spk_tbl.to_csv(args.out_dir / "speakers.csv")
    print(f"\ntables -> {args.out_dir}")

    out = args.out_dir
    subset_hists(df, [("duration_s", "Duration"), ("clean_rms_dbfs", "Clean level"),
                      (f"silent_frac_{MAIN_SILENCE_DB}", "Silent-frame fraction")],
                 "Clean speech", out / "01_clean_speech.png", args.show)
    subset_hists(df, [("snr_db", "Global SNR"), ("snr_speech_db", "Speech-frame SNR"),
                      ("lsd_db", "Log-spectral distance")], "Difficulty", out / "02_difficulty.png", args.show)
    plot_noise_map(df, out / "03_noise_character.png", args.show)
    plot_spectra(df, extras, out / "04_frequency_profile.png", args.show)
    plot_irm(df, extras, out / "05_mask_targets.png", args.show)
    plot_phase(df, out / "06_phase_and_clipping.png", args.show)
    plot_silence(df, out / "07_silence.png", args.show)
    plot_examples(df, {p["pair_id"]: p for p in pairs}, out / "08_examples.png", args.show)


if __name__ == "__main__":
    main()