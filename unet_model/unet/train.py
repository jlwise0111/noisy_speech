"""Training loop: AdamW, ReduceLROnPlateau or cosine schedule, early stopping on validation loss.

Model selection only ever looks at the validation speakers. The test set is used once, after
training, by enhance.py.
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .loss import masked_mse_parts, weighted_mask_loss
from .model import UNet


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    crop_frames: int = 256         # ~2.05 s at hop 128 / 16 kHz; multiple of 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    scheduler: str = "plateau"     # "plateau" or "cosine"
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    min_lr: float = 1e-5
    early_stop: int = 6            # epochs without val improvement before stopping (0 = never)
    grad_clip: float = 5.0
    loss: str = "mse"              # "mse" or "huber"
    huber_delta: float = 0.1
    silence_weight: float = 0.25   # 1.0 = plain unweighted loss
    silence_db: float = 30.0       # silent = clean frame energy this far below the utterance peak
    target: str = "irm"            # "irm" or "psm"
    base_channels: int = 16
    workers: int = -1              # DataLoader processes for training crops (-1 = from the CPU count)
    val_workers: int = 1           # DataLoader processes for validation (whole utterances, batch 1)
    threads: int = 0               # torch intra-op threads (0 = the cores the loader workers leave free)
    prefetch_factor: int = 4       # batches each worker keeps ready, so the model never waits on loading
    seed: int = 0
    device: str = "auto"

    def to_dict(self):
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def available_cores() -> int:
    """Cores this process may use (respects CPU affinity and container limits where Python can see them)."""
    count = getattr(os, "process_cpu_count", None) or os.cpu_count
    return max(1, count() or 1)


def resolve_parallelism(cfg: TrainConfig, cores: int | None = None) -> TrainConfig:
    """Fill in the automatic worker and thread counts so loaders + torch never exceed `cores`.

    The loader workers and torch's own threads compete for the same cores. Left at their defaults
    (a fixed worker count plus one torch thread per core), they oversubscribe the CPU and thrash."""
    cores = cores or available_cores()
    workers = cfg.workers if cfg.workers >= 0 else min(4, (cores - 1) // 2)
    threads = cfg.threads if cfg.threads > 0 else max(1, cores - workers)
    return replace(cfg, workers=workers, threads=threads)


def apply_threads(cfg: TrainConfig) -> None:
    """Cap torch's CPU threads at cfg.threads (call after resolve_parallelism)."""
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def make_loader(dataset, cfg: TrainConfig, train: bool) -> DataLoader:
    """Workers load and transform the next batches in the background while the current one trains."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    workers = max(0, cfg.workers if train else cfg.val_workers)
    extra = {"persistent_workers": True, "prefetch_factor": cfg.prefetch_factor} if workers else {}
    return DataLoader(dataset, batch_size=cfg.batch_size if train else 1, shuffle=train,
                      num_workers=workers, generator=g, drop_last=train,
                      pin_memory=torch.cuda.is_available(), **extra)


def _to(batch, device):
    return [batch[k].to(device, non_blocking=True) for k in ("x", "y", "speech", "valid")]


def train_epoch(model, loader, optimizer, cfg: TrainConfig, device, scheduler=None) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x, y, speech, valid = _to(batch, device)
        loss = weighted_mask_loss(model(x), y, speech, valid, cfg.silence_weight, cfg.loss, cfg.huber_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if scheduler is not None:            # cosine steps per batch
            scheduler.step()
        total += float(loss) * len(x)
        n += len(x)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig, device) -> dict:
    """Validation loss (the training objective) plus unweighted MSE split into speech / silent frames."""
    model.eval()
    loss_sum, n = 0.0, 0
    parts = {"se_speech": 0.0, "n_speech": 0, "se_silent": 0.0, "n_silent": 0}
    for batch in loader:
        x, y, speech, valid = _to(batch, device)
        pred = model(x)
        loss_sum += float(weighted_mask_loss(pred, y, speech, valid, cfg.silence_weight, cfg.loss,
                                             cfg.huber_delta))
        n += 1
        for k, v in masked_mse_parts(pred, y, speech, valid).items():
            parts[k] += v
    n_all = parts["n_speech"] + parts["n_silent"]
    return {
        "val_loss": loss_sum / max(n, 1),
        "val_mse": (parts["se_speech"] + parts["se_silent"]) / max(n_all, 1),
        "val_mse_speech": parts["se_speech"] / max(parts["n_speech"], 1),
        "val_mse_silent": parts["se_silent"] / max(parts["n_silent"], 1),
    }


def fit(train_ds, val_ds, norm: dict, cfg: TrainConfig, run_dir: Path, resume: bool = False, log=print) -> Path:
    """Train, keep the best checkpoint by validation loss, return its path."""
    cfg = resolve_parallelism(cfg)
    apply_threads(cfg)
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = UNet(base_channels=cfg.base_channels)
    model.set_normalization(norm["mean"], norm["std"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader, val_loader = make_loader(train_ds, cfg, True), make_loader(val_ds, cfg, False)
    if cfg.scheduler == "cosine":
        batch_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs * len(train_loader), eta_min=cfg.min_lr)
        plateau = None
    elif cfg.scheduler == "plateau":
        batch_sched = None
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience, min_lr=cfg.min_lr)
    else:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r} (plateau or cosine)")

    best_path, last_path, hist_path = run_dir / "best.pt", run_dir / "last.pt", run_dir / "history.csv"
    start, best, stale = 1, math.inf, 0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        (batch_sched or plateau).load_state_dict(ck["scheduler"])
        start, best, stale = ck["epoch"] + 1, ck["best"], ck["stale"]
        log(f"resumed after epoch {ck['epoch']} (best val_loss {best:.5f})")
    elif hist_path.exists():
        hist_path.unlink()

    log(f"device {device}, {cfg.threads} torch thread(s), {cfg.workers}+{cfg.val_workers} loader workers")
    log(f"{sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{len(train_ds)} train / {len(val_ds)} val files, {len(train_loader)} batches per epoch")
    for epoch in range(start, cfg.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, cfg, device, batch_sched)
        val = evaluate(model, val_loader, cfg, device)
        if plateau is not None:
            plateau.step(val["val_loss"])

        improved = val["val_loss"] < best
        if improved:
            best, stale = val["val_loss"], 0
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch, **val}, best_path)
        else:
            stale += 1
        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, **val, "seconds": round(time.time() - t0, 1)}
        _append_csv(hist_path, row)
        log(f"epoch {epoch:3d}  lr {lr:.2e}  train {train_loss:.5f}  val {val['val_loss']:.5f}  "
            f"(mse speech {val['val_mse_speech']:.5f} / silent {val['val_mse_silent']:.5f})  "
            f"{row['seconds']:.0f}s{'  *best' if improved else ''}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": (batch_sched or plateau).state_dict(),
                    "epoch": epoch, "best": best, "stale": stale}, last_path)
        if cfg.early_stop and stale >= cfg.early_stop:
            log(f"early stop: no val improvement for {stale} epochs")
            break
    return best_path


def load_model(checkpoint, device="cpu") -> UNet:
    ck = torch.load(checkpoint, map_location=device)
    model = UNet(base_channels=ck["config"]["base_channels"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
