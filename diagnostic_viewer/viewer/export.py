"""Figure export (kaleido) and metrics CSV, including batch over a whole dataset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import plotly.graph_objects as go

from . import metrics
from .audio_io import TripletError, load_wav, validate_strict
from .dataset import ROLES, Triplet


def figure_image(fig: go.Figure, fmt: str = "png", width: int = 1600, scale: float = 2.0) -> bytes:
    height = int(fig.layout.height or 900)
    return fig.to_image(format=fmt, width=width, height=height, scale=scale if fmt == "png" else 1)


def save(data: bytes | str, export_dir: Path, name: str) -> Path | None:
    """Best-effort write to the exports volume; returns None if it isn't writable."""
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path
    except OSError:
        return None


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def batch_metrics(
    triplets: Iterable[Triplet], root: Path, channel: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Metrics for every triplet; unresolved or mismatched triplets get a status row instead."""
    triplets = list(triplets)
    rows = []
    for i, t in enumerate(triplets):
        if progress:
            progress(i, len(triplets), t.stem)
        row = {"stem": t.stem, "layout": t.layout}
        if not t.resolved:
            rows.append({**row, "status": "skipped: " + ", ".join(t.issues)})
            continue
        files = t.files
        row.update({f"{r}_file": str(files[r].relative_to(root)) for r in ROLES})
        try:
            audios = {r: load_wav(files[r]) for r in ROLES}
            validate_strict(audios)
            ch = min(channel, audios["clean"].n_channels - 1)
            sig = {r: a.channel(ch) for r, a in audios.items()}
            sr = audios["clean"].sr
            df = metrics.compute_all(sig["clean"], sig["noisy"], sig["processed"], sr)
            rows.append({**row, "status": "ok", "sample_rate_hz": sr,
                         "duration_s": audios["clean"].duration, **metrics.flatten(df)})
        except TripletError as e:
            rows.append({**row, "status": "rejected: " + str(e).replace("\n", " ")})
        except Exception as e:  # unreadable file etc. — never abort the whole batch
            rows.append({**row, "status": f"error: {e}"})
    if progress:
        progress(len(triplets), len(triplets), "done")
    return pd.DataFrame(rows)
