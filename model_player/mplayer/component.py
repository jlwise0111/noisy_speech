"""The lock-step multi-track player: one iframe with the stacked figure, transport and levels.

Python ships every track as raw 16-bit PCM plus the figure. In the browser one AudioWorklet holds
all tracks and reads them with a single shared position, so they cannot drift apart; exactly one
track is audible (XOR solo) and switching crossfades over 2 ms with equal-power gains.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

from . import CODE_ROOT  # noqa: F401  (sets sys.path for viewer)
from viewer.player import PLOTLY_SRC, _encode_heatmaps, _NumpyEncoder

_TEMPLATE = (Path(__file__).parent / "player.html").read_text(encoding="utf-8")
TRANSPORT_PX = 132  # transport (track buttons wrap onto a second line) + grid gap
PANEL_PX = 420


def pcm16_b64(x: np.ndarray) -> str:
    """float in [-1, 1] -> base64 little-endian int16."""
    q = np.round(np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def player(
    fig: go.Figure,
    tracks: list[dict],
    sr: int,
    n_samples: int,
    freqs_hz: dict[int, np.ndarray],
    levels: list[dict],
    format_line: str,
    norm_lufs: float,
    solo: int = 0,
    debug: bool = False,
) -> None:
    """tracks: [{slot, label, description, color, x (original level), gain_db (loudness normalization)}]
    in display order; levels: [{lufs, peak_dbfs}] at the original level. The player applies gain_db as
    the first stage of playback when "Match loudness" is on (it starts on)."""
    fig_dict = _encode_heatmaps(fig)
    color = {t["slot"]: t["color"] for t in tracks}
    label = {t["slot"]: t["label"] for t in tracks}
    for tr in fig_dict["data"]:
        meta = tr.get("meta") or {}
        if meta.get("kind") == "spec":
            meta["freqs_hz"] = np.asarray(freqs_hz[meta["slot"]]).round(2).tolist()
            meta["color"] = color[meta["slot"]]
            meta["role"] = label[meta["slot"]]
    data = {
        "figure": fig_dict,
        "sr": sr,
        "n": n_samples,
        "duration": n_samples / sr,
        "solo": solo,
        "debug": debug,
        "format_line": format_line,
        "levels": levels,
        "norm_lufs": norm_lufs,
        "tracks": [{"slot": t["slot"], "label": t["label"], "description": t["description"],
                    "color": t["color"], "gain_db": float(t["gain_db"]), "pcm": pcm16_b64(t["x"])}
                   for t in tracks],
    }
    payload = json.dumps(data, cls=_NumpyEncoder)  # embedded as a JS literal: NaN / Infinity are fine
    html = (_TEMPLATE.replace("__PLOTLY_SRC__", PLOTLY_SRC)
            .replace("__DATA__", payload.replace("</", "<\\/")))
    height = max(int(fig.layout.height or 700) + TRANSPORT_PX, PANEL_PX)
    components.html(html, height=height, scrolling=False)
