"""Plotly figure builders. The stacked figure is used both live (player) and for export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import dsp

# Dark, RX-like palette shared with .streamlit/config.toml and the player.
BG = "#14161b"
PLOT_BG = "#05060a"
FG = "#d6d9df"
MUTED = "#8a919e"
GRID = "rgba(255,255,255,0.06)"
ACCENT = "#f39c12"
WAVE_COLOR = "77,163,255"  # RGB of #4da3ff; alpha comes from the opacity setting

# Distinguishable on a dark background; used for the line charts and the stats panel.
ROLE_COLORS = {"clean": "#4fd1a5", "noisy": "#f5a142", "processed": "#a495ff"}

EMBER = [  # black → deep blue → rust → orange → amber → near-white, approximating RX
    [0.00, "#030409"], [0.18, "#0d1838"], [0.38, "#4a1f3a"], [0.55, "#a8401c"],
    [0.72, "#e8811f"], [0.88, "#f9c74f"], [1.00, "#fff6d8"],
]
SPEC_COLORMAPS = {"Ember": EMBER, "Magma": "Magma", "Viridis": "Viridis", "Greys": "Greys"}

TRACK_PX, DIFF_PX, GAP_PX = 230, 200, 8
MARGIN = dict(l=52, r=58, t=6, b=34)
MEL_TICKS_HZ = (100, 200, 500, 1000, 2000, 4000, 8000, 16000)


@dataclass
class TrackView:
    slot: int
    role: str
    name: str
    x: np.ndarray
    sr: int
    spec: dsp.Spectrogram


@dataclass
class DiffView:
    label: str
    db: np.ndarray
    spec: dsp.Spectrogram  # for the time/frequency axes


def _dark(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PLOT_BG, height=height,
        margin=MARGIN, font=dict(color=MUTED, size=11), hovermode="closest", dragmode="zoom",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="")


def _freq_axis(fig, row, spec: dsp.Spectrogram, sr: int):
    # secondary_y=False: never touch the hidden waveform axis that overlays this panel.
    kw = dict(row=row, col=1, secondary_y=False, tickfont=dict(size=10), tickformat="~s", nticks=6)
    if spec.scale == "log":
        fig.update_yaxes(type="log", **kw)
    elif spec.scale == "mel":
        ticks = [f for f in MEL_TICKS_HZ if f < sr / 2]
        kw.pop("tickformat"), kw.pop("nticks")
        fig.update_yaxes(tickvals=list(dsp.hz_to_mel(ticks)),
                         ticktext=[f"{f/1000:g}k" if f >= 1000 else str(f) for f in ticks], **kw)
    else:
        fig.update_yaxes(**kw)


def _spec_y(spec: dsp.Spectrogram) -> np.ndarray:
    return dsp.hz_to_mel(spec.freqs) if spec.scale == "mel" else spec.freqs


def _time_axis(fig, row: int):
    fig.update_xaxes(showticklabels=True, ticks="outside", ticklen=4, tickcolor=MUTED, ticksuffix=" s",
                     tickfont=dict(size=10), row=row, col=1)


def _label(fig, row: int, text: str, name: str, active: bool = False):
    yref = fig.get_subplot(row, 1).yaxis.plotly_name.replace("axis", "")
    fig.add_annotation(
        text=text, name=name, xref="paper", yref=f"{yref} domain", x=0.004, y=0.97,
        xanchor="left", yanchor="top", showarrow=False, captureevents=True,
        font=dict(size=11, color="#111" if active else FG),
        bgcolor=ACCENT if active else "rgba(10,12,16,0.72)", borderpad=4,
    )


def track_label(t: TrackView, monitored: bool) -> str:
    return f"{'🎧 ' if monitored else ''}<b>{t.role.upper()}</b> · {t.name}"


def stacked_figure(
    tracks: list[TrackView],
    db_range: tuple[float, float],
    colormap: str = "Ember",
    wave_opacity: float = 0.55,
    shared_amplitude: bool = True,
    monitored_slot: int = 1,
) -> go.Figure:
    """One panel per track: spectrogram with the waveform overlaid on a hidden secondary y-axis."""
    n = len(tracks)
    height = n * TRACK_PX + (n - 1) * GAP_PX + MARGIN["t"] + MARGIN["b"]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=GAP_PX / height,
                        specs=[[{"secondary_y": True}]] * n)
    amp = max(max(float(np.max(np.abs(t.x))) for t in tracks), 1e-3) * 1.05

    for i, t in enumerate(tracks, start=1):
        s = t.spec
        fig.add_trace(go.Heatmap(
            x=s.times, y=_spec_y(s), z=s.db, coloraxis="coloraxis",
            hovertemplate="%{x:.3f} s · %{y:.0f} " + ("mel" if s.scale == "mel" else "Hz")
            + " · %{z:.1f} dB<extra>" + t.role + "</extra>",
            meta={"kind": "spec", "slot": t.slot, "role": t.role},
        ), row=i, col=1, secondary_y=False)
        _freq_axis(fig, i, s, t.sr)

        if wave_opacity > 0:
            tx, y = dsp.waveform_envelope(t.x, t.sr)
            a = amp if shared_amplitude else max(float(np.max(np.abs(t.x))), 1e-3) * 1.05
            fig.add_trace(go.Scatter(
                x=tx, y=y, mode="lines", hoverinfo="skip",
                line=dict(width=0.8, color=f"rgba({WAVE_COLOR},{wave_opacity})"),
                meta={"kind": "wave", "slot": t.slot, "role": t.role},
            ), row=i, col=1, secondary_y=True)
            fig.update_yaxes(range=[-a, a], visible=False, row=i, col=1, secondary_y=True)
        _label(fig, i, track_label(t, t.slot == monitored_slot), f"label{t.slot}", t.slot == monitored_slot)

    _dark(fig, height)
    _time_axis(fig, n)
    fig.update_layout(coloraxis=dict(
        colorscale=SPEC_COLORMAPS.get(colormap, colormap), cmin=db_range[0], cmax=db_range[1],
        colorbar=dict(thickness=8, len=1, y=0.5, x=1.004, xpad=6, outlinewidth=0, ticksuffix=" dB",
                      tickfont=dict(size=10, color=MUTED), nticks=6)))
    return fig


def difference_figure(diffs: list[DiffView], sr: int, diff_range: float = 20.0) -> go.Figure:
    n = len(diffs)
    height = n * DIFF_PX + (n - 1) * GAP_PX + MARGIN["t"] + MARGIN["b"]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=GAP_PX / height)
    for i, d in enumerate(diffs, start=1):
        s = d.spec
        fig.add_trace(go.Heatmap(
            x=s.times, y=_spec_y(s), z=d.db, coloraxis="coloraxis",
            hovertemplate="%{x:.3f} s · %{y:.0f} " + ("mel" if s.scale == "mel" else "Hz")
            + " · %{z:+.1f} dB<extra>" + d.label + "</extra>",
        ), row=i, col=1)
        _freq_axis(fig, i, s, sr)
        _label(fig, i, f"<b>{d.label}</b>", f"diff{i}")
    _dark(fig, height)
    _time_axis(fig, n)
    fig.update_layout(coloraxis=dict(
        colorscale="RdBu_r", cmid=0, cmin=-diff_range, cmax=diff_range,
        colorbar=dict(thickness=8, len=1, x=1.004, xpad=6, outlinewidth=0, ticksuffix=" dB",
                      tickfont=dict(size=10, color=MUTED))))
    return fig


def _line_layout(fig: go.Figure, height: int, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
                      margin=dict(l=60, r=20, t=30, b=50), font=dict(color=MUTED, size=11),
                      xaxis_title=xtitle, yaxis_title=ytitle, hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def ltas_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], log_freq: bool = True) -> go.Figure:
    fig = go.Figure()
    for role, (f, db) in curves.items():
        keep = f > 0
        fig.add_trace(go.Scatter(x=f[keep], y=db[keep], name=role, mode="lines",
                                 line=dict(color=ROLE_COLORS.get(role), width=1.5),
                                 hovertemplate="%{x:.0f} Hz<br>%{y:.1f} dB/Hz<extra>" + role + "</extra>"))
    _line_layout(fig, 360, "Frequency (Hz)", "Level (dB/Hz)")
    if log_freq:
        fig.update_xaxes(type="log")
    return fig


def envelope_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], floor_db: float = -80) -> go.Figure:
    fig = go.Figure()
    for role, (t, db) in curves.items():
        fig.add_trace(go.Scatter(x=t, y=np.maximum(db, floor_db), name=role, mode="lines",
                                 line=dict(color=ROLE_COLORS.get(role), width=1.2),
                                 hovertemplate="%{x:.3f} s<br>%{y:.1f} dBFS<extra>" + role + "</extra>"))
    return _line_layout(fig, 320, "Time (s)", "Loudness (dBFS)")
