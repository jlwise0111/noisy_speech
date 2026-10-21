#!/usr/bin/env python3
"""Train the U-Net mask model on Valentini and enhance the test set.

    python train_unet.py --dataset "$DS" --train-set 28spk
    python train_unet.py --dataset "$DS" --model outputs/<run>/best.pt --skip-train   # re-enhance only

This file is the *experiment script*. It holds no ML of its own: it calls `speechprep` and `unet`
in the right order, enforces the leakage rules, records what happened in config.json, and writes
outputs/<run>/{clean,noisy,enhanced}/<stem>.wav for the metric suite and the viewer.

Order of the run (each step is one function below; main() calls them in this order):
    parse_args -> check_paths -> new_run_info -> split_pairs -> limit_pairs -> warm_cache
    -> normalization_stats -> build_datasets -> train_model -> enhance_test -> write_json

Done when:  .venv/bin/python -m pytest tests/test_end_to_end.py -q   passes.
"""
from __future__ import annotations


import argparse
import dataclasses
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from speechprep import WaveCache, split_by_speaker, test_pairs, train_pairs  # noqa: E402
from unet import TrainConfig, fit, load_model, n_parameters  # noqa: E402
from unet.dataset import PairDataset, input_stats  # noqa: E402
from unet.enhance import enhance_pairs  # noqa: E402
from unet.train import pick_device  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.environ.get("VALENTINI_DIR"),
                   help="Valentini root (default: $VALENTINI_DIR)")
    p.add_argument("--train-set", default="28spk", choices=["28spk", "56spk", "both"])
    p.add_argument("--val-speakers", default=None, help="comma-separated, e.g. p226,p287 (default: ~10%% by --seed)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--limit-files", type=int, default=None,
                   help="use only N train + val pairs, sampled after the speaker split (smoke runs)")
    p.add_argument("--test-limit", type=int, default=None, help="enhance only the first N test files")
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--run-name", default=None, help="default: unet_<train-set>_<YYYYmmdd_HHMMSS>")
    p.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "speechprep")
    p.add_argument("--stats-files", type=int, default=1000, help="training files used for normalization stats")
    p.add_argument("--model", type=Path, default=None, help="checkpoint to use with --skip-train")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    for f in dataclasses.fields(TrainConfig):
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = p.parse_args(argv)
    if not args.dataset:
        p.error("--dataset is required (or set VALENTINI_DIR)")
    if args.skip_train and not args.model:
        p.error("--skip-train needs --model")
    if "train" in (args.run_name or ""):
        p.error("--run-name must not contain 'train': the metric suite skips such folders as training-split output")
    return args


def inside(path: Path, root: Path) -> bool:
    """True if `path` is `root` or anywhere below it."""

    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def check_paths(args) -> None:
    """Refuse to write into the dataset. It is shared and read-only by convention."""

    for d, name in ((args.out, "--out"), (args.cache_dir, "--cache-dir")):
        if inside(d, Path(args.dataset)):
            sys.exit(f"{name} must not be inside the dataset folder (the dataset is read-only)")


def git_commit() -> str | None:
    """The current commit hash, or None outside a git checkout."""

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def new_run_info(args, cfg, run_name: str) -> dict:
    """Start the record of this run. It grows as the run goes and is saved as config.json."""

    return {"run_name": run_name, "dataset": str(args.dataset), "train_set": args.train_set,
           "git_commit": git_commit(), "config": cfg.to_dict(), "torch": torch.__version__}


def split_pairs(args, cfg):
    """Discover the training pairs and hold out whole speakers for validation.

    The single most important leakage control. Speakers have hundreds of utterances each, so a
    random *file* split would put voices the model trained on into validation, and validation
    would overestimate test performance."""

    pairs = train_pairs(args.dataset, args.train_set)
    val_speakers = args.val_speakers.split(",") if args.val_speakers else None
    train, val, held = split_by_speaker(pairs, val_speakers, args.val_frac, cfg.seed)

    return pairs, train, val, held

def limit_pairs(pairs, train, val, limit, seed: int):
    """For smoke runs: keep about `limit` files in total, sampled AFTER the speaker split.

    Cutting the list BEFORE splitting keeps the first few hundred files by name, which all belong
    to one speaker, so validation comes out empty."""

    if limit is None or limit == 0:
        return train, val
    rng = random.Random(seed)
    n_val = max(1, round(limit * len(val) / len(pairs)))
    train = sorted(rng.sample(train, min(len(train), limit - n_val)))
    val = sorted(rng.sample(val, min(len(val), n_val)))

    return train, val


def warm_cache(cache, pair_lists) -> int:
    """Resample every needed file to 16 kHz once, in parallel, before training starts.

    Reading and resampling 48 kHz audio is the slow part of loading. Done once here, every epoch
    afterward reads small .npy files."""

    paths = [p for pairs in pair_lists for _, clean, noisy in pairs for p in (clean, noisy)]
    return cache.warm(paths)


def normalization_stats(train, cache, args, cfg) -> dict:
    """Per-bin mean/std of the noisy log-magnitude, from TRAINING speakers only.

    The input scaling is part of the model. If validation or test audio shaped it, information
    from those sets would leak into training."""

    norm = input_stats(train, cache, args.stats_files, cfg.seed, cfg.silence_db)
    print(f"normalization from {norm['n_files']} training files; "
          f"silent frames ({cfg.silence_db:g} dB below peak): {100 * norm['silent_frac']:.1f}%")

    return norm


def build_datasets(train, val, cache, cfg):
    """Training set = random fixed-length crops; validation set = whole utterances.

    Crops give fixed-size batches (no padding for BatchNorm to see) and a little augmentation.
    Validation should look like the test set, which is whole files."""

    train_ds = PairDataset(train, cache, cfg.crop_frames, cfg.target, cfg.silence_db)
    val_ds = PairDataset(val, cache, None, cfg.target, cfg.silence_db)

    return train_ds, val_ds


def train_model(train_ds, val_ds, norm, cfg, run_dir: Path, resume: bool, info: dict) -> Path:
    """Run the training loop and record the best epoch. Returns the path to best.pt.

    fit() already does AdamW, the LR schedule, early stopping on the validation loss,
    history.csv, best.pt and last.pt. This step only calls it and records the outcome."""

    checkpoint = fit(train_ds, val_ds, norm, cfg, run_dir, resume=resume)
    best = torch.load(checkpoint, map_location="cpu")
    info.update(best_epoch=best["epoch"], **{k: best[k] for k in best if k.startswith("val_")})
    print(f"best checkpoint: epoch {best['epoch']}, val_loss {best['val_loss']:.5f}")

    return checkpoint


def enhance_test(checkpoint: Path, test, cache, run_dir: Path, cfg, info: dict) -> None:
    """Score-ready output: run the chosen checkpoint over the test set, once.

    If you look at test scores, change something and retrain, the test set has become a
    validation set, and the final number is optimistic. Decide everything on validation first."""

    device = pick_device(cfg.device)
    model = load_model(checkpoint, device)
    info["checkpoint"] = str(checkpoint)
    info["n_parameters"] = n_parameters(model)
    enhance_pairs(model, test, cache, run_dir, device)
    info["n_test"] = len(test)


def write_json(path: Path, obj) -> None:
    """Save `obj` as readable JSON."""

    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def main(argv=None):
    """Run one experiment end to end.

    `argv=None` means "use the real command line"; the test passes its own list instead."""

    args = parse_args(argv)
    check_paths(args)
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)})
    run_name = args.run_name or f"unet_{args.train_set}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out / run_name
    cache = WaveCache(args.cache_dir)
    t_start = time.time()
    info = new_run_info(args, cfg, run_name)
    test = test_pairs(args.dataset)[:args.test_limit]
    if args.skip_train:
        checkpoint = args.model
    else:
        pairs, train, val, held = split_pairs(args, cfg)
        train, val = limit_pairs(pairs, train, val, args.limit_files, cfg.seed)
        if not train or not val:
            sys.exit(f"Empty split: {len(train)} train / {len(val)} val files")
        print(f"{len(train)} train / {len(val)} val files; validation speakers {', '.join(held)}")
        n_new = warm_cache(cache, [train, val] + ([] if args.skip_test else [test]))
        print(f"16 kHz cache: {n_new} file(s) added under {cache.root}")
        norm = normalization_stats(train, cache, args, cfg)
        info.update(val_speakers=held, n_train=len(train), n_val=len(val),
                    silent_frac_train=norm["silent_frac"], norm_files=norm["n_files"])
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", info)
        train_ds, val_ds = build_datasets(train, val, cache, cfg)
        checkpoint = train_model(train_ds, val_ds, norm, cfg, run_dir, args.resume, info)

    if not args.skip_test:
        print(f"enhancing {len(test)} test file(s) -> {run_dir}")
        enhance_test(checkpoint, test, cache, run_dir, cfg, info)

    info["seconds"] = round(time.time() - t_start, 1)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", info)
    print(f'done in {info["seconds"]:.0f}s. Score with: make evaluate METRIC_ARGS="--include {run_name}"')


if __name__ == "__main__":
    main()
