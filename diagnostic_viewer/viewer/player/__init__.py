"""Synced multi-track player: spectrogram panels, numbers panel and transport in one iframe.

Streamlit reruns can't animate a playhead smoothly, so playback, the playhead, the cursor
readout and the spectrum-at-playhead card all run client-side. Python only ships the data.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

from ..figures import ROLE_COLORS

_TEMPLATE = (Path(__file__).parent / "player.html").read_text(encoding="utf-8")
PLOTLY_SRC = "/app/static/plotly.min.js"  # written into ./static by the Dockerfile
TRANSPORT_PX = 96  # transport bar (two lines) + grid gap
SLICE_CARD_PX = 270
PANEL_PX = 610  # cursor + quality + levels cards


def _encode_heatmaps(fig: go.Figure) -> dict:
    """Figure dict with heatmap z replaced by base64 float32 (≈5× smaller than JSON text)."""
    d = fig.to_plotly_json()
    for tr in d["data"]:
        if tr.get("type") == "heatmap":
            z = np.ascontiguousarray(np.asarray(tr["z"], dtype="<f4"))
            tr["z"] = {"__f32__": base64.b64encode(z.tobytes()).decode("ascii"), "shape": list(z.shape)}
    return d


def player(
    fig: go.Figure,
    tracks: list[dict],
    duration: float,
    scale: str,
    db_range: tuple[float, float],
    freqs_hz: dict[int, np.ndarray],
    stats: list[dict],
    scorecard: list[dict] | None,
    format_line: str,
    show_slice: bool = False,
) -> None:
    """tracks: [{slot, role, name, wav: bytes}] in display order; stats: per track, same order."""
    fig_dict = _encode_heatmaps(fig)
    for tr in fig_dict["data"]:
        meta = tr.get("meta") or {}
        if meta.get("kind") == "spec":
            meta["freqs_hz"] = np.asarray(freqs_hz[meta["slot"]]).round(2).tolist()
            meta["color"] = ROLE_COLORS.get(meta["role"], "#888")
    data = {
        "figure": fig_dict,
        "duration": duration,
        "scale": scale,
        "db_range": list(db_range),
        "show_slice": show_slice,
        "format_line": format_line,
        "scorecard": scorecard,
        "stats": [{**s, "role": t["role"], "name": t["name"], "color": ROLE_COLORS.get(t["role"], "#888")}
                  for s, t in zip(stats, tracks)],
        "tracks": [{"slot": t["slot"], "role": t["role"], "name": t["name"],
                    "b64": base64.b64encode(t["wav"]).decode("ascii")} for t in tracks],
    }
    # Embedded as a JS literal (not JSON.parse), so NaN / Infinity are valid.
    payload = json.dumps(data, cls=_NumpyEncoder)
    html = (_TEMPLATE.replace("__PLOTLY_SRC__", PLOTLY_SRC)
            .replace("__DATA__", payload.replace("</", "<\\/")))
    height = max(int(fig.layout.height or 700) + TRANSPORT_PX,
                 PANEL_PX + (SLICE_CARD_PX if show_slice else 0))
    components.html(html, height=height, scrolling=False)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)
