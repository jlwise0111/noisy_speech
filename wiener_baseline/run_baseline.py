#!/usr/bin/env python3
"""Run the Wiener baseline over a Valentini split and write viewer-ready triplets.

Output layout (what diagnostic_viewer's "subfolders" mode expects):

    outputs/<run-name>/
        clean/<stem>.wav       copied from the dataset
        noisy/<stem>.wav       copied from the dataset
        processed/<stem>.wav   Wiener output, same sr and length as noisy
        params.json            settings used for this run
        quick_metrics.csv      per-file SNR / SI-SDR sanity check

Example:
    python run_baseline.py --dataset "/path/to/valentini_noisy_speech_database" \
        --split test --limit 10

Full setup and usage guide: wiener_baseline/docs/wiener_baseline.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wiener import WienerParams, wiener_filter, wiener_oracle
from wiener.audio_io import SPLITS, find_pairs, read_wav, write_wav

HERE = Path(__file__).resolve().parent


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-12) + 1e-12))


def si_sdr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    target = (np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)) * ref
    noise = est - target
    return float(10 * np.log10(np.sum(target ** 2) / (np.sum(noise ** 2) + 1e-12) + 1e-12))


def process_one(job: tuple) -> dict:
    stem, noisy_path, clean_path, run_dir, method, noise_est, params = job
    noisy, sr = read_wav(noisy_path)
    clean, sr_c = read_wav(clean_path)
    if sr != sr_c or len(noisy) != len(clean):
        raise ValueError(f"{stem}: noisy/clean mismatch "
                         f"({sr} Hz, {len(noisy)}) vs ({sr_c} Hz, {len(clean)})")

    if method == "oracle":
        out = wiener_oracle(noisy, clean, sr, params)
    else:
        out = wiener_filter(noisy, sr, noise_est, params)

    write_wav(run_dir / "processed" / f"{stem}.wav", out, sr)
    for role, src in (("noisy", noisy_path), ("clean", clean_path)):
        dst = run_dir / role / f"{stem}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return {
        "stem": stem,
        "snr_noisy": snr_db(clean, noisy),
        "snr_processed": snr_db(clean, out),
        "si_sdr_noisy": si_sdr_db(clean, noisy),
        "si_sdr_processed": si_sdr_db(clean, out),
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("VALENTINI_DIR"),
                    help="Valentini dataset root (or set VALENTINI_DIR)")
    ap.add_argument("--split", choices=sorted(SPLITS), default="test")
    ap.add_argument("--method", choices=["wiener", "oracle"], default="wiener",
                    help="'oracle' uses the clean signal: upper bound only, not a real baseline")
    ap.add_argument("--noise-est", choices=["leading", "minstats"], default="leading")
    ap.add_argument("--limit", type=int, help="process only the first N files")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--run-name", help="default: <method>_<noise-est>_<split>")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    for name, default in WienerParams().to_dict().items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    args = ap.parse_args()

    if not args.dataset:
        ap.error("--dataset is required (or set VALENTINI_DIR)")
    dataset = Path(args.dataset).resolve()
    out_root = args.out.resolve()
    if out_root == dataset or dataset in out_root.parents:
        ap.error("refusing to write inside the dataset folder")

    params = WienerParams(**{k: getattr(args, k) for k in WienerParams().to_dict()})
    tag = "oracle" if args.method == "oracle" else f"wiener_{args.noise_est}"
    run_dir = out_root / (args.run_name or f"{tag}_{args.split}")

    pairs = find_pairs(dataset, args.split)[: args.limit]
    if not pairs:
        print("no matched noisy/clean pairs found", file=sys.stderr)
        return 1
    print(f"{len(pairs)} file(s) from '{args.split}' -> {run_dir}")

    jobs = [(s, n, c, run_dir, args.method, args.noise_est, params) for s, n, c in pairs]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for i, row in enumerate(ex.map(process_one, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    with open(run_dir / "quick_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (run_dir / "params.json").write_text(json.dumps({
        "method": args.method,
        "noise_est": args.noise_est if args.method == "wiener" else None,
        "split": args.split,
        "dataset": str(dataset),
        "n_files": len(rows),
        "git_commit": git_commit(),
        "params": params.to_dict(),
    }, indent=2))

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    print(f"\nmean over {len(rows)} files      noisy   processed   delta")
    for m in ("snr", "si_sdr"):
        a, b = mean(f"{m}_noisy"), mean(f"{m}_processed")
        print(f"  {m.upper():<8} (dB)          {a:6.2f}   {b:9.2f}   {b - a:+5.2f}")
    print(f"\nPESQ/STOI: point diagnostic_viewer at {run_dir} and use batch export.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
