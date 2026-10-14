"""Plotly figure builders. The stacked figure is used both live (player) and for export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import dsp

# Colour-blind-safe (ColorBrewer Dark2) colours, fixed per role everywhere in the app.
ROLE_COLORS = {"clean": "#1b9e77", "noisy": "#d95f02", "processed": "#7570b3"}
SPEC_COLORMAPS = ("Magma", "Viridis", "Inferno", "Cividis", "Greys")
TEMPLATE = "plotly_white"

WAVE_PX, SPEC_PX, DIFF_PX = 90, 220, 190
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


def _freq_axis(fig, row, spec: dsp.Spectrogram, sr: int):
    title = "Hz"
    if spec.scale == "log":
        fig.update_yaxes(type="log", row=row, col=1)
    elif spec.scale == "mel":
        ticks = [f for f in MEL_TICKS_HZ if f < sr / 2]
        fig.update_yaxes(tickvals=list(dsp.hz_to_mel(ticks)),
                         ticktext=[f"{f/1000:g}k" if f >= 1000 else str(f) for f in ticks],
                         row=row, col=1)
        title = "Hz (mel)"
    fig.update_yaxes(title_text=title, title_standoff=4, row=row, col=1)


def _spec_y(spec: dsp.Spectrogram) -> np.ndarray:
    return dsp.hz_to_mel(spec.freqs) if spec.scale == "mel" else spec.freqs


def stacked_figure(
    tracks: list[TrackView],
    diffs: list[DiffView],
    db_range: tuple[float, float],
    colormap: str = "Magma",
    diff_range: float = 20.0,
    shared_amplitude: bool = True,
) -> go.Figure:
    rows, heights, titles = [], [], []
    for t in tracks:
        rows += [("wave", t), ("spec", t)]
        heights += [WAVE_PX, SPEC_PX]
        titles += [f"<b>{t.slot}. {t.role.upper()}</b> — {t.name}", ""]
    for d in diffs:
        rows.append(("diff", d))
        heights.append(DIFF_PX)
        titles.append(f"<b>Δ {d.label}</b> (dB)")

    total = sum(heights)
    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=28 / total,
        row_heights=[h / total for h in heights], subplot_titles=titles,
    )
    amp = max(float(np.max(np.abs(t.x))) for t in tracks) if tracks else 1.0
    amp = max(amp, 1e-3) * 1.05

    for i, (kind, obj) in enumerate(rows, start=1):
        if kind == "wave":
            tx, y = dsp.waveform_envelope(obj.x, obj.sr)
            fig.add_trace(go.Scatter(
                x=tx, y=y, mode="lines", line=dict(width=0.7, color=ROLE_COLORS.get(obj.role, "#444")),
                name=obj.role, showlegend=False,
                hovertemplate="t=%{x:.3f} s<br>amp=%{y:.4f}<extra>" + obj.role + "</extra>",
                meta={"kind": "wave", "slot": obj.slot, "role": obj.role},
            ), row=i, col=1)
            a = amp if shared_amplitude else max(float(np.max(np.abs(obj.x))), 1e-3) * 1.05
            fig.update_yaxes(range=[-a, a], title_text="amp", title_standoff=4, zeroline=True, row=i, col=1)
        elif kind == "spec":
            s = obj.spec
            unit = "mel" if s.scale == "mel" else "Hz"
            fig.add_trace(go.Heatmap(
                x=s.times, y=_spec_y(s), z=s.db, coloraxis="coloraxis",
                hovertemplate="t=%{x:.3f} s<br>f=%{y:.0f} " + unit + "<br>%{z:.1f} dB<extra>"
                + obj.role + "</extra>",
                meta={"kind": "spec", "slot": obj.slot, "role": obj.role},
            ), row=i, col=1)
            _freq_axis(fig, i, s, obj.sr)
        else:
            s = obj.spec
            unit = "mel" if s.scale == "mel" else "Hz"
            fig.add_trace(go.Heatmap(
                x=s.times, y=_spec_y(s), z=obj.db, coloraxis="coloraxis2",
                hovertemplate="t=%{x:.3f} s<br>f=%{y:.0f} " + unit + "<br>Δ %{z:+.1f} dB<extra>"
                + obj.label + "</extra>",
                meta={"kind": "diff", "label": obj.label},
            ), row=i, col=1)
            _freq_axis(fig, i, s, tracks[0].sr if tracks else 16000)

    spec_frac = (len(tracks) * (WAVE_PX + SPEC_PX)) / total
    fig.update_layout(
        template=TEMPLATE,
        height=total + 90,
        margin=dict(l=70, r=90, t=40, b=50),
        hovermode="closest",
        coloraxis=dict(colorscale=colormap, cmin=db_range[0], cmax=db_range[1],
                       colorbar=dict(title="dB", len=min(0.5, spec_frac), y=1, yanchor="top", thickness=12)),
        coloraxis2=dict(colorscale="RdBu_r", cmid=0, cmin=-diff_range, cmax=diff_range,
                        colorbar=dict(title="Δ dB", len=max(0.15, 1 - spec_frac), y=0, yanchor="bottom",
                                      thickness=12)),
        dragmode="zoom",
    )
    fig.update_annotations(font_size=12, xanchor="left", x=0)
    fig.update_xaxes(showticklabels=False, ticks="")
    fig.update_xaxes(title_text="Time (s)", showticklabels=True, ticks="outside", row=len(rows), col=1)
    return fig


def ltas_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], log_freq: bool = True) -> go.Figure:
    fig = go.Figure()
    for role, (f, db) in curves.items():
        keep = f > 0
        fig.add_trace(go.Scatter(x=f[keep], y=db[keep], name=role, mode="lines",
                                 line=dict(color=ROLE_COLORS.get(role), width=1.5),
                                 hovertemplate="%{x:.0f} Hz<br>%{y:.1f} dB/Hz<extra>" + role + "</extra>"))
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(l=60, r=20, t=30, b=50),
                      xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB/Hz)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    if log_freq:
        fig.update_xaxes(type="log")
    return fig


def envelope_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], floor_db: float = -80) -> go.Figure:
    fig = go.Figure()
    for role, (t, db) in curves.items():
        fig.add_trace(go.Scatter(x=t, y=np.maximum(db, floor_db), name=role, mode="lines",
                                 line=dict(color=ROLE_COLORS.get(role), width=1.2),
                                 hovertemplate="%{x:.3f} s<br>%{y:.1f} dBFS<extra>" + role + "</extra>"))
    fig.update_layout(template=TEMPLATE, height=320, margin=dict(l=60, r=20, t=30, b=50),
                      xaxis_title="Time (s)", yaxis_title="RMS level (dBFS)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    return fig
