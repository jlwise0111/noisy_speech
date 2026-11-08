"""Results figures for the final presentation, built from the scored runs.

Reads the per-file metric CSVs (metric_suite/results/runs), the U-Net training histories
(unet_model/outputs/<run>/history.csv) and the 16 kHz test outputs, and writes PNGs to figures/:

    improvement.png    each system's PESQ and STOI change over the noisy input, with 95% intervals
    experiments.png    the three U-Net experiments vs the main run (paired differences, 95% intervals)
    training.png       main run train/validation loss, and validation MSE of the three ratio-mask runs
    spectrograms.png   one test clip: noisy, Wiener, U-Net, U-Net (phase-sensitive mask), clean

Usage (from the project folder):  make figures   or   python figures/make_figures.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import stft  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "metric_suite" / "results" / "runs"
UNET = ROOT / "unet_model" / "outputs"
OUT = Path(__file__).resolve().parent

# Text and structure colors, plus one accent for the U-Nets (reference palette, slots 1-3).
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BASELINE, CEILING = "#8c959f", "#c6ccd2"

# (label, metric CSV, family) in display order, top to bottom.
SYSTEMS = [
    ("Ridge regression", "linear_baseline__outputs__ridge_28spk", "baseline"),
    ("Wiener, min. statistics", "wiener_baseline__outputs__wiener_minstats_test", "baseline"),
    ("Wiener, leading noise", "wiener_baseline__outputs__wiener_leading_test", "baseline"),
    ("U-Net, half width", "unet_model__outputs__unet_28spk_c8", "unet"),
    ("U-Net, plain MSE", "unet_model__outputs__unet_28spk_sw1", "unet"),
    ("U-Net (main)", "unet_model__outputs__unet_28spk", "unet"),
    ("U-Net, phase-sensitive mask", "unet_model__outputs__unet_28spk_psm", "unet"),
    ("Oracle mask (ceiling)", "wiener_baseline__outputs__oracle_test", "oracle"),
]
MAIN = "unet_model__outputs__unet_28spk"
EXPERIMENTS = [
    ("Phase-sensitive mask", "unet_model__outputs__unet_28spk_psm"),
    ("Plain MSE (silence weight 1.0)", "unet_model__outputs__unet_28spk_sw1"),
    ("Half width (¼ parameters)", "unet_model__outputs__unet_28spk_c8"),
]


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def load(name):
    """Per-file rows of one scored run, keyed by stem."""
    path = RUNS / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}: run make evaluate first")
    with open(path) as f:
        return {r["stem"]: r for r in csv.DictReader(f) if r.get("status", "ok") == "ok"}


def column(rows, stems, key):
    return np.array([float(rows[s][key]) for s in stems])


def bootstrap_ci(d, n=5000, seed=0):
    """95% interval of the mean of d, resampling files with replacement."""
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def bar_panel(ax, labels, means, cis, colors, title, fmt, zero=True):
    y = np.arange(len(labels))[::-1]
    err = np.abs(np.array(cis).T - means)
    ax.barh(y, means, height=0.62, color=colors, edgecolor="white", linewidth=2)
    ax.errorbar(means, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    if zero:
        ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title)
    span = max(abs(c) for ci in cis for c in ci)
    for yi, m, ci in zip(y, means, cis):
        edge = ci[1] if m >= 0 else ci[0]
        ax.text(edge + np.sign(m or 1) * span * 0.03, yi, fmt.format(m).replace("-", "\u2212"), va="center",
                ha="left" if m >= 0 else "right", color=INK, fontsize=10)
    lo = min(0, min(c[0] for c in cis))
    hi = max(0, max(c[1] for c in cis))
    pad = (hi - lo) * 0.3
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)


def improvement_figure():
    rows = {name: load(name) for _, name, _ in SYSTEMS}
    stems = sorted(set.intersection(*(set(r) for r in rows.values())))
    colors = [{"baseline": BASELINE, "unet": BLUE, "oracle": CEILING}[fam] for _, _, fam in SYSTEMS]
    labels = [label for label, _, _ in SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key, title, fmt in [
        (axes[0], "pesq_wb_delta", "PESQ-WB change vs noisy input", "{:+.2f}"),
        (axes[1], "stoi_delta", "STOI change vs noisy input", "{:+.3f}"),
    ]:
        deltas = [column(rows[name], stems, key) for _, name, _ in SYSTEMS]
        bar_panel(ax, labels, [d.mean() for d in deltas], [bootstrap_ci(d) for d in deltas], colors,
                  title, fmt)
    fig.suptitle(f"Improvement over the noisy input ({len(stems)} test files, 16 kHz; bars: 95% intervals)",
                 x=0.01, ha="left", color=MUTED, fontsize=10, y=0.995)
    fig.tight_layout()
    save(fig, "improvement.png")


def experiments_figure():
    main = load(MAIN)
    runs = [(label, load(name)) for label, name in EXPERIMENTS]
    stems = sorted(set(main).intersection(*(set(r) for _, r in runs)))
    labels = [label for label, _ in runs]
    colors = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, key, title, fmt in [
        (axes[0], "pesq_wb_processed", "PESQ-WB", "{:+.3f}"),
        (axes[1], "stoi_processed", "STOI", "{:+.4f}"),
        (axes[2], "si_sdr_db_processed", "SI-SDR (dB)", "{:+.2f}"),
    ]:
        base = column(main, stems, key)
        diffs = [column(r, stems, key) - base for _, r in runs]
        bar_panel(ax, labels, [d.mean() for d in diffs], [bootstrap_ci(d) for d in diffs], colors,
                  title, fmt)
    fig.suptitle("Each experiment minus the main U-Net, file by file (positive = better; bars: 95% intervals)",
                 x=0.01, ha="left", color=MUTED, fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, "experiments.png")


def history(run):
    with open(UNET / run / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def training_figure():
    main = history("unet_28spk")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(main["epoch"], main["train_loss"], color=MUTED, linewidth=2, label="training")
    ax.plot(main["epoch"], main["val_loss"], color=BLUE, linewidth=2, label="validation")
    best = int(np.argmin(main["val_loss"]))
    ax.plot(main["epoch"][best], main["val_loss"][best], "o", markersize=8, color=BLUE,
            markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.annotate(f"best checkpoint, epoch {int(main['epoch'][best])}", (main["epoch"][best], main["val_loss"][best]),
                xytext=(0, 18), textcoords="offset points", ha="center", color=INK, fontsize=10)
    for e in main["epoch"][1:][np.diff(main["lr"]) < 0]:
        ax.axvline(e, color=MUTED, linewidth=1, linestyle="--", alpha=0.6, zorder=0)
    ax.set_title("Main U-Net: silence-weighted loss")
    ax.set_xlabel("epoch  (dashed: learning rate halved)")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")

    ax = axes[1]
    for run, label, color in [("unet_28spk", "main (silence weight 0.25)", BLUE),
                              ("unet_28spk_sw1", "plain MSE (weight 1.0)", ORANGE),
                              ("unet_28spk_c8", "half width", AQUA)]:
        h = history(run)
        ax.plot(h["epoch"], h["val_mse"], color=color, linewidth=2, label=label)
        ax.text(h["epoch"][-1] + 0.4, h["val_mse"][-1], label.split(" (")[0], color=INK, fontsize=9,
                va="center")
    ax.set_title("Validation mask MSE (unweighted), ratio-mask runs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.legend(loc="upper right")
    for a in axes:
        a.grid(color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
    axes[1].set_xlim(right=axes[1].get_xlim()[1] + 3)
    fig.tight_layout()
    save(fig, "training.png")


def pick_clip(stem=None):
    """A typical clip: input SNR near the median, 2.5-4 s long, U-Net PESQ gain near its median."""
    rows = load(MAIN)
    if stem:
        return stem
    ok = [r for r in rows.values() if 2.5 <= float(r["duration_s"]) <= 4.0]
    snr_med = np.median([float(r["snr_db_noisy"]) for r in rows.values()])
    gain_med = np.median([float(r["pesq_wb_delta"]) for r in rows.values()])
    return min(ok, key=lambda r: abs(float(r["snr_db_noisy"]) - snr_med)
               + 2 * abs(float(r["pesq_wb_delta"]) - gain_med))["stem"]


def log_spec(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000, f"{path} is {sr} Hz, expected 16 kHz (re-run the model)"
    f, t, z = stft(x, fs=sr, window="hann", nperseg=512, noverlap=512 - 128)
    return f, t, 20 * np.log10(np.abs(z) + 1e-8)


def spectrogram_figure(stem):
    snr = float(load(MAIN)[stem]["snr_db_noisy"])
    panels = [
        ("Noisy input", UNET / "unet_28spk" / "noisy" / f"{stem}.wav"),
        ("Wiener, leading noise", ROOT / "wiener_baseline" / "outputs" / "wiener_leading_test" / "processed" / f"{stem}.wav"),
        ("U-Net (main)", UNET / "unet_28spk" / "enhanced" / f"{stem}.wav"),
        ("U-Net, phase-sensitive mask", UNET / "unet_28spk_psm" / "enhanced" / f"{stem}.wav"),
        ("Clean target", UNET / "unet_28spk" / "clean" / f"{stem}.wav"),
    ]
    specs = [log_spec(p) for _, p in panels]
    top = max(s[2].max() for s in specs)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 11), sharex=True, layout="constrained")
    for ax, (title, _), (f, t, db) in zip(axes, panels, specs):
        im = ax.pcolormesh(t, f / 1000, db, vmin=top - 80, vmax=top, cmap="magma", shading="auto",
                           rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("kHz")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("seconds")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("dB (80 dB range)")
    cb.outline.set_visible(False)
    fig.suptitle(f"Test clip {stem} (input SNR {snr:.1f} dB), STFT 512 / hop 128 at 16 kHz",
                 x=0.01, ha="left", color=MUTED, fontsize=10)
    save(fig, "spectrograms.png")


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", help="test file stem for the spectrogram figure (default: a typical clip)")
    args = ap.parse_args(argv)
    style()
    improvement_figure()
    experiments_figure()
    training_figure()
    stem = pick_clip(args.clip)
    spectrogram_figure(stem)
    print(f"  spectrogram clip: {stem}")


if __name__ == "__main__":
    main()
