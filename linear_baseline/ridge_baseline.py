"""Linear ML baseline: per-frequency-bin Ridge regression on 3x3 noisy log-magnitude patches.

Trains one Ridge model per STFT bin on the Valentini training set, sweeps the penalty α
(chosen on held-out speakers), and then enhances the Valentini test set. Output triplets go to
<out>/<run_name>/{clean,noisy,enhanced}/<stem>.wav, a layout diagnostic_viewer/ reads directly.

Usage (from linear_baseline/):
    python ridge_baseline.py --dataset /path/to/valentini_noisy_speech_database --train-set 28spk
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lrb import data, ridge, synth  # noqa: E402
from lrb.features import N_FEATURES  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ALPHAS = np.logspace(-6, 2, 17)  # per-frame penalty, see lrb/ridge.py


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="Valentini root folder (read only)")
    ap.add_argument("--train-set", choices=["28spk", "56spk", "both"], default="28spk")
    ap.add_argument("--alphas", type=lambda s: [float(a) for a in s.split(",")], default=None,
                    help="Comma-separated per-frame ridge penalties (default: logspace(-6, 2, 17))")
    ap.add_argument("--per-bin-alpha", action="store_true",
                    help="Pick the best α independently per frequency bin (default: one global α)")
    ap.add_argument("--val-speakers", type=lambda s: s.split(","), default=None,
                    help="Comma-separated held-out speakers, e.g. p226,p287 (default: ~10%% random)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit-files", type=int, default=None, help="Use only the first N training pairs")
    ap.add_argument("--test-limit", type=int, default=None, help="Enhance only the first N test pairs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--model", type=Path, default=None, help="Existing model.npz (use with --skip-train)")
    ap.add_argument("--skip-train", action="store_true")
    return ap.parse_args(argv)


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def accumulate(pairs, args, pool, label, t0):
    """Sum per-file statistics over `pairs` in parallel, printing progress."""
    total = ridge.Stats.zeros(args.n_fft // 2 + 1, N_FEATURES)
    work = chunks(pairs, 16)
    fn = partial(_chunk_stats, sr=args.sr, n_fft=args.n_fft, hop=args.hop)
    done = 0
    for i, (n_files, s) in enumerate(pool.imap_unordered(fn, work), 1):
        total = total + s
        done += n_files
        if i % 25 == 0 or i == len(work):
            log(f"{label}: {done}/{len(pairs)} files", t0)
    return total


def _chunk_stats(pairs, **kw):
    return len(pairs), synth.chunk_stats(pairs, **kw)


def train(args, run_dir, t0):
    pairs = data.train_pairs(args.dataset, args.train_set)
    if args.limit_files:
        pairs = pairs[:args.limit_files]
    tr_pairs, va_pairs, val_spk = data.split_by_speaker(pairs, args.val_speakers, args.val_frac, args.seed)
    if not tr_pairs or not va_pairs:
        sys.exit(f"Empty split: {len(tr_pairs)} train / {len(va_pairs)} val files. "
                 "Increase --limit-files or set --val-speakers.")
    log(f"Train set {args.train_set}: {len(tr_pairs)} train / {len(va_pairs)} val files "
        f"(val speakers: {', '.join(val_spk)})", t0)

    with Pool(args.workers) as pool:
        tr = accumulate(tr_pairs, args, pool, "train stats", t0)
        va = accumulate(va_pairs, args, pool, "val stats", t0)

    alphas = np.asarray(args.alphas if args.alphas else DEFAULT_ALPHAS, dtype=np.float64)
    tr_mse, va_mse = ridge.sweep(tr, va, alphas)  # (A, F)
    tr_tot = (tr_mse * tr.n).sum(1) / tr.n.sum()
    va_tot = (va_mse * va.n).sum(1) / va.n.sum()
    passthrough = float(ridge.sse(identity_model(len(tr.n)), va).sum() / va.n.sum())

    print(f"\n  Log-magnitude MSE vs clean (val, no processing): {passthrough:.6f}")
    print("  alpha        train MSE    val MSE")
    best_i = int(np.argmin(va_tot))
    for i, a in enumerate(alphas):
        mark = "  <- best" if i == best_i else ""
        print(f"  {a:<10.4g}  {tr_tot[i]:.6f}     {va_tot[i]:.6f}{mark}")
    print()

    if args.per_bin_alpha:
        chosen = alphas[np.argmin(va_mse, axis=0)]
        chosen_val = va_mse.min(axis=0) @ va.n / va.n.sum()
        log(f"Per-bin α chosen (median {np.median(chosen):.4g}); val MSE {chosen_val:.6f}", t0)
    else:
        chosen = np.full(len(tr.n), alphas[best_i])
        chosen_val = va_tot[best_i]
        log(f"Global α = {alphas[best_i]:.4g}; val MSE {chosen_val:.6f}", t0)

    # Final model: refit on train + val at the chosen penalty
    model = ridge.fit(tr + va, chosen)

    with open(run_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "train_mse", "val_mse"])
        w.writerows(zip(alphas, tr_tot, va_tot))
    with open(run_dir / "sweep_per_bin.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "freq_hz", "best_alpha", "best_val_mse", "chosen_alpha"]
                   + [f"val_mse@{a:.4g}" for a in alphas])
        freqs = np.arange(len(tr.n)) * args.sr / args.n_fft
        for fb in range(len(tr.n)):
            j = int(np.argmin(va_mse[:, fb]))
            w.writerow([fb, f"{freqs[fb]:.1f}", alphas[j], va_mse[j, fb], chosen[fb]] + list(va_mse[:, fb]))

    np.savez(run_dir / "model.npz", W=model.W, b=model.b, alpha=model.alpha,
             sr=args.sr, n_fft=args.n_fft, hop=args.hop)
    return model, {
        "train_files": len(tr_pairs), "val_files": len(va_pairs), "val_speakers": val_spk,
        "alphas": alphas.tolist(), "val_mse": va_tot.tolist(), "train_mse": tr_tot.tolist(),
        "selection": "per-bin" if args.per_bin_alpha else "global",
        "chosen_alpha": float(alphas[best_i]) if not args.per_bin_alpha else None,
        "chosen_val_mse": float(chosen_val), "val_mse_passthrough": passthrough,
    }


def identity_model(n_bins):
    """Output = noisy centre value; the 'no processing' reference."""
    W = np.zeros((n_bins, N_FEATURES))
    W[:, N_FEATURES // 2] = 1.0
    return ridge.Model(W=W, b=np.zeros(n_bins), alpha=np.zeros(n_bins))


def load_model(path, args):
    z = np.load(path)
    for k in ("sr", "n_fft", "hop"):
        setattr(args, k, int(z[k]))
    return ridge.Model(W=z["W"], b=z["b"], alpha=z["alpha"])


_MODEL = None


def _init_worker(W, b, alpha):
    global _MODEL
    _MODEL = ridge.Model(W=W, b=b, alpha=alpha)


def _enhance_one(pair, run_dir, sr, n_fft, hop):
    stem, clean_path, noisy_path = pair
    clean, noisy = data.load_audio(clean_path, sr), data.load_audio(noisy_path, sr)
    enhanced = synth.enhance(noisy, _MODEL, n_fft, hop)
    synth.write_triplet(run_dir, stem, clean, noisy, enhanced, sr)
    return stem


def enhance_test(model, args, run_dir, t0):
    pairs = data.list_pairs(args.dataset, *data.TEST_SET)
    if args.test_limit:
        pairs = pairs[:args.test_limit]
    fn = partial(_enhance_one, run_dir=run_dir, sr=args.sr, n_fft=args.n_fft, hop=args.hop)
    with Pool(args.workers, initializer=_init_worker, initargs=(model.W, model.b, model.alpha)) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, pairs, chunksize=8), 1):
            if i % 100 == 0 or i == len(pairs):
                log(f"test enhance: {i}/{len(pairs)} files", t0)
    return len(pairs)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    dataset = args.dataset.resolve()
    out = args.out.resolve()
    if out == dataset or dataset in out.parents:
        sys.exit("--out must not be inside the dataset folder (the dataset is read-only).")
    if args.skip_train and not args.model:
        sys.exit("--skip-train requires --model path/to/model.npz")

    run_name = args.run_name or f"ridge_{args.train_set}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run folder: {run_dir}", t0)

    if args.skip_train:
        model = load_model(args.model, args)
        info = {"model_from": str(args.model)}
    else:
        model, info = train(args, run_dir, t0)
    t_train = time.time() - t0

    n_test = enhance_test(model, args, run_dir, t0)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(info, test_files=n_test, train_seconds=round(t_train, 1),
                  total_seconds=round(time.time() - t0, 1))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"Done. {n_test} triplets in {run_dir}", t0)
    print(f"View: cd diagnostic_viewer && DATA_DIR={out} docker compose up")


if __name__ == "__main__":
    main()
