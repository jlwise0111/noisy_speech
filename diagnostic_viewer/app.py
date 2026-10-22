"""Speech-isolation diagnostic viewer (Streamlit entry point).

Run with `docker compose up` and open http://localhost:8501.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from viewer import dsp, export, figures, metrics
from viewer.audio_io import Audio, TripletError, load_wav, to_wav_bytes, validate_strict
from viewer.dataset import DEFAULT_FOLDER_ALIASES, DEFAULT_SUFFIX_ALIASES, ROLES, discover, list_wavs
from viewer.player import player
from viewer.systems import SYSTEMS, group_runs, split_key

DATA_DIR = os.environ.get("VIEWER_DATA_DIR", "data")
EXPORT_DIR = Path(os.environ.get("VIEWER_EXPORT_DIR", "exports"))
MANUAL = "__manual__"
TARGET_LUFS = -16.0
SYSTEM_COLORS = {"Neural network": "#a495ff", "Wiener baseline": "#4fd1a5", "Ridge baseline": "#f39c12",
                 "Other": "#8a919e"}

st.set_page_config(page_title="Speech Isolation Viewer", page_icon="🎧", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:2.6rem;padding-bottom:1rem;padding-left:2rem;padding-right:2rem;max-width:100%}
h3{margin:0!important;padding:0!important}
/* header: labels on buttons, popovers and toggles never break mid-word */
.stButton button p,[data-testid="stPopover"] button p,[data-testid="stPopoverButton"] p,
.stCheckbox label p{white-space:nowrap}
.stButton button,[data-testid="stPopover"] button{padding-left:.6rem;padding-right:.6rem}
/* header rows: selection on the left, tools on the right; fixed-size controls keep their natural width
   and the rows wrap whole controls (never their labels) when the window is narrow */
@media (min-width:641px){
[data-testid="stHorizontalBlock"]:has(.stPopover){flex-wrap:wrap;row-gap:.5rem}
[data-testid="stHorizontalBlock"]:has(.stPopover)>[data-testid="stColumn"]{flex:0 0 auto!important;width:auto!important}
[data-testid="stHorizontalBlock"]:has(.stPopover)>[data-testid="stColumn"]:nth-child(1){flex:1 1 0!important;min-width:180px;max-width:280px}
[data-testid="stHorizontalBlock"]:has(.stPopover)>[data-testid="stColumn"]:nth-child(2){flex:1.3 1 0!important;min-width:200px;max-width:360px}
[data-testid="stHorizontalBlock"]:has(.stPopover)>[data-testid="stColumn"]:nth-child(3){padding-left:.5rem;min-height:2.5rem;display:flex;justify-content:center}
[data-testid="stHorizontalBlock"]:has(.stPopover)>[data-testid="stColumn"]:nth-child(4){margin-left:auto}
[data-testid="stHorizontalBlock"]:has(.np){flex-wrap:wrap;row-gap:.5rem}
[data-testid="stHorizontalBlock"]:has(.np)>[data-testid="stColumn"]{flex:0 0 3rem!important;width:3rem!important;min-width:0;max-width:3rem}
[data-testid="stHorizontalBlock"]:has(.np)>[data-testid="stColumn"]:nth-child(1){flex:2 1 0!important;width:auto!important;min-width:260px;max-width:none}
[data-testid="stHorizontalBlock"]:has(.np)>[data-testid="stColumn"]:nth-child(4){flex:1 1 240px!important;width:auto!important;max-width:340px}
}
.stMarkdown:has(.np)>div{margin-bottom:0!important}
/* the "now playing" block: which system + run + file is on screen */
.np{display:flex;flex-direction:column;gap:5px;min-width:0;padding:2px 0 2px 12px;border-left:3px solid}
.np-top{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;min-width:0}
.np-badge{color:#14161b;font-weight:700;font-size:.78rem;line-height:1;padding:4px 8px;
  border-radius:3px;white-space:nowrap;align-self:center}
.np h3{font-size:1.45rem!important;line-height:1.15!important;white-space:nowrap;font-weight:600}
.np-path{color:#8a919e;font-size:.8rem;line-height:1.3;overflow-wrap:anywhere}
.np-path b{color:#c3c8d1;font-weight:500}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------- cached work
# cache_resource, not cache_data: one shared copy of the (large, read-only) scan results instead of
# a pickled copy rebuilt on every rerun. The nonce is per browser session (see below), so a few
# scans are kept: enough for the open tabs, without growing for as long as the container runs.
@st.cache_resource(show_spinner="Scanning dataset…", max_entries=4)
def cached_discover(root: str, layout: str, folder_aliases: tuple, suffix_aliases: tuple, nonce: int):
    return discover(root, layout, {r: list(a) for r, a in folder_aliases}, {r: list(a) for r, a in suffix_aliases})


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_list_wavs(root: str, nonce: int) -> list[Path]:
    return list_wavs(Path(root))


@st.cache_data(show_spinner=False, max_entries=48)
def cached_load(path: str, mtime: float) -> Audio:
    return load_wav(path)


@st.cache_data(show_spinner=False, max_entries=96)
def cached_spec(path: str, mtime: float, ch: int, params: tuple) -> dsp.Spectrogram:
    a = cached_load(path, mtime)
    return dsp.spectrogram(a.channel(ch), a.sr, dsp.StftParams(*params))


@st.cache_data(show_spinner="Scoring quality…", max_entries=48)
def cached_metrics(paths: tuple[str, str, str], mtimes: tuple, ch: int) -> pd.DataFrame:
    c, n, p = (cached_load(pth, m) for pth, m in zip(paths, mtimes))
    return metrics.compute_all(c.channel(ch), n.channel(ch), p.channel(ch), c.sr)


@st.cache_data(show_spinner=False, max_entries=96)
def cached_stats(path: str, mtime: float, ch: int) -> dict:
    a = cached_load(path, mtime)
    return dsp.signal_stats(a.channel(ch), a.sr)


@st.cache_data(show_spinner=False, max_entries=96)
def cached_wav_bytes(path: str, mtime: float, ch: int, gain_db: float = 0.0) -> bytes:
    a = cached_load(path, mtime)
    return to_wav_bytes(a.channel(ch) * 10 ** (gain_db / 20), a.sr)


@st.cache_data(show_spinner=False, max_entries=96)
def cached_loudness(path: str, mtime: float, ch: int) -> tuple[float, float]:
    """(integrated loudness in LUFS, sample peak in dBFS)."""
    a = cached_load(path, mtime)
    x = a.channel(ch)
    return dsp.integrated_loudness(x, a.sr), float(20 * np.log10(np.max(np.abs(x)) + dsp.EPS))


def mtime(p: Path) -> float:
    return p.stat().st_mtime


def aliases_input(label: str, defaults: dict[str, list[str]], key: str) -> tuple:
    out = []
    for r in ROLES:
        txt = st.text_input(f"{label} — {r}", ", ".join(defaults[r]), key=f"{key}_{r}")
        out.append((r, tuple(a.strip() for a in txt.split(",") if a.strip())))
    return tuple(out)


# ---------------------------------------------------------------------------- sidebar: dataset only
ss = st.session_state
# A new browser session (first visit, page reload, or the tab the pipeline opens) scans the folder
# afresh, so outputs written while the container was running show up without ↻ Rescan.
ss.setdefault("nonce", time.time_ns())
ss.setdefault("tri_idx", 0)

with st.sidebar:
    st.header("Dataset")
    root = st.text_input("Data folder (container path)", DATA_DIR,
                         help="The host folder is set with DATA_DIR in .env and mounted read-only at /data.")
    layout = st.radio("Layout", ["auto", "subfolders", "suffix"], horizontal=True,
                      help="subfolders: root/clean/x.wav, root/noisy/x.wav …  ·  suffix: x_clean.wav, x_noisy.wav …")
    with st.expander("Role naming (comma-separated aliases)"):
        folder_aliases = aliases_input("Folder", DEFAULT_FOLDER_ALIASES, "fa")
        suffix_aliases = aliases_input("Suffix", DEFAULT_SUFFIX_ALIASES, "sa")
    if st.button("↻ Rescan folder", use_container_width=True):
        ss.nonce += 1

root_path = Path(root)
triplets = cached_discover(root, layout, folder_aliases, suffix_aliases, ss.nonce)
all_wavs = cached_list_wavs(root, ss.nonce)
n_ok = sum(t.resolved for t in triplets)
with st.sidebar:
    st.caption(f"{len(all_wavs)} .wav files · {len(triplets)} triplets ({n_ok} complete)")

if not root_path.is_dir():
    st.error(f"Data folder `{root}` does not exist inside the container. Check DATA_DIR in `.env` "
             "(dataset settings are in the sidebar, top-left `>`).")
    st.stop()
if not all_wavs:
    st.info(f"No .wav files found under `{root}`. Put your audio in the folder set by DATA_DIR in `.env` "
            "(default: the project folder), or generate demo data — see README.")
    st.stop()

# ---------------------------------------------------------------------------- systems → runs → files
if ss.get("scan_id") != id(triplets):  # a new scan can renumber the files: keep the same one selected
    ss.scan_id = id(triplets)
    keys = {t.stem: i for i, t in enumerate(triplets)}
    ss.sel_tri = keys.get(ss.get("sel_key"), ss.get("sel_tri"))
systems = group_runs(triplets)
runs_by_path = {r.path: r for rs in systems.values() for r in rs}


# The selection lives in sel_* keys: Streamlit forgets a widget's value when the widget is not drawn
# (e.g. while a system without outputs is selected), and the file should survive that.
def file_stem(i) -> str | None:
    return split_key(triplets[i].stem)[1] if isinstance(i, int) and i < len(triplets) else None


def select_run(path: str):
    """Switch to a run, keeping the same file (e.g. p232_001) selected if that run has it."""
    run, stem = runs_by_path[path], file_stem(ss.get("sel_tri"))
    ss.sel_run = path
    same = [i for i in run.triplets if file_stem(i) == stem]
    ss.sel_tri = same[0] if same else run.triplets[0]


def on_system_change():
    ss.sel_system = ss.system
    if systems[ss.system]:
        select_run(systems[ss.system][0].path)


if runs_by_path:
    if ss.get("sel_run") not in runs_by_path:  # first load, or the dataset changed under us
        first = next(rs[0] for rs in systems.values() if rs)
        ss.sel_system = first.system
        select_run(first.path)
    if ss.get("sel_system") not in systems:
        ss.sel_system = runs_by_path[ss.sel_run].system
    run = runs_by_path[ss.sel_run] if systems[ss.sel_system] else None
    options = (run.triplets if run else []) + [MANUAL]
else:  # wavs, but nothing that forms a triplet: manual selection only
    run, options = None, [MANUAL]
if ss.get("sel_tri") not in options:
    ss.sel_tri = options[0]
ss.system, ss.run, ss.tri_idx = ss.get("sel_system"), ss.get("sel_run"), ss.sel_tri  # mirror into widgets
ss.sel_key = triplets[ss.sel_tri].stem if ss.sel_tri != MANUAL else None


def step(delta: int):
    i = options.index(ss.sel_tri) if ss.sel_tri in options else 0
    ss.sel_tri = options[(i + delta) % len(options)]


def file_label(i) -> str:
    if i == MANUAL:
        return "✎ Manual selection…"
    t = triplets[i]
    return f"✓ {file_stem(i)}" if t.resolved else f"⚠ {file_stem(i)} ({', '.join(t.issues)})"


# ---------------------------------------------------------------------------- header rows
def run_name(path: str) -> str:
    """Run menu entry: its own folder name, or the full path if two runs of the system share that name."""
    r = runs_by_path[path]
    same = [o for o in systems[r.system] if o.name == r.name]
    return r.name if len(same) == 1 else r.label


s_sys, s_run, s_lufs, s_tracks, s_disp = st.columns([1.8, 2.4, 1.9, 1, 1.15], vertical_alignment="bottom")
if runs_by_path:
    s_sys.selectbox("System", list(systems), key="system", on_change=on_system_change,
                    help="Which processing system's output you are looking at and hearing.")
    if run:
        sys_runs = systems[ss.sel_system]
        s_run.selectbox(f"Run ({len(sys_runs)} available)" if len(sys_runs) > 1 else "Run",
                        [r.path for r in sys_runs], key="run",
                        on_change=lambda: select_run(ss.run), format_func=run_name,
                        help="Output folders of this system:  \n"
                             + "  \n".join(f"`{r.label}` ({len(r.triplets)} files)" for r in sys_runs))
match_lufs = s_lufs.toggle("Match loudness", key="match_lufs",
                           help=f"Play and draw every track at the same integrated loudness ({TARGET_LUFS:g} LUFS, "
                                "ITU-R BS.1770), so quieter processing can be judged on sound quality alone. "
                                "Quality scores are always computed from the original files.")
tracks_pop = s_tracks.popover("Tracks", use_container_width=True)
display_pop = s_disp.popover("⚙ Display", use_container_width=True)

if runs_by_path and not run:
    st.info(f"**{ss.sel_system}: no outputs found yet** under `{root}`.  \n"
            "Write them as `<folder>/<run>/{clean,noisy,enhanced}/<stem>.wav` (for example "
            "`unet_model/outputs/unet_28spk/enhanced/p232_001.wav`). Any folder name containing one of "
            f"{', '.join(dict(SYSTEMS).get(ss.sel_system, ()))} is listed under this system after **↻ Rescan folder**.")
    st.stop()

h_title, h_prev, h_next, h_pick = st.columns([3.6, 0.5, 0.5, 2.6], vertical_alignment="center")
h_prev.button("◀", on_click=step, args=(-1,), help="Previous file", use_container_width=True)
h_next.button("▶", on_click=step, args=(1,), help="Next file", use_container_width=True)
h_pick.selectbox("File", options, key="tri_idx", label_visibility="collapsed", format_func=file_label,
                 on_change=lambda: ss.update(sel_tri=ss.tri_idx))

tri = None if ss.sel_tri == MANUAL else triplets[ss.sel_tri]
tri_key = "manual" if tri is None else tri.stem
system_name = run.system if (run and tri) else "Manual selection"
if tri:
    color = SYSTEM_COLORS.get(system_name, "#8a919e")
    h_title.markdown(
        f"<div class='np' style='border-left-color:{color}'><div class='np-top'>"
        f"<span class='np-badge' style='background:{color}'>{system_name}</span>"
        f"<h3>{file_stem(ss.sel_tri)}</h3></div>"
        f"<div class='np-path' title='{run.label}'><b>{run.name}</b>"
        f"{' in ' + run.label.rpartition('/')[0] if '/' in run.label else ''}"
        f" <span style='white-space:nowrap'>· {len(run.triplets)} files</span></div></div>", unsafe_allow_html=True)
else:
    h_title.markdown("<div class='np' style='border-left-color:#8a919e'><h3>Manual selection</h3></div>",
                     unsafe_allow_html=True)

# ---------------------------------------------------------------------------- resolve the triplet
rel = lambda p: str(p.relative_to(root_path)) if p else "—"  # noqa: E731

files: dict[str, Path | None] = {r: None for r in ROLES} if tri is None else dict(tri.files)
unresolved = [r for r in ROLES if files[r] is None]
if unresolved:
    if tri:
        st.warning(f"**{tri.stem}**: {', '.join(tri.issues)}. Pick the file for each unresolved role.")
    else:
        st.info("Pick a file for each role.")
    cols = st.columns(len(unresolved))
    for col, r in zip(cols, unresolved):
        cands = tri.candidates[r] if tri else []
        choices = [None] + cands + [w for w in all_wavs if w not in cands]
        files[r] = col.selectbox(r, choices, key=f"pick_{tri_key}_{r}",
                                 format_func=lambda p: "— choose —" if p is None else rel(p))
    if any(files[r] is None for r in ROLES):
        st.stop()

try:
    audios = {r: cached_load(str(files[r]), mtime(files[r])) for r in ROLES}
    validate_strict(audios)
except TripletError as e:
    st.error(str(e).replace("\n", "  \n"))
    st.stop()
except Exception as e:
    st.error(f"Could not read audio: {e}")
    st.stop()

sr = audios["clean"].sr
n_ch = audios["clean"].n_channels
duration = audios["clean"].duration

# ---------------------------------------------------------------------------- display popover
with display_pop:
    channel = st.selectbox("Channel", range(n_ch), format_func=lambda c: f"ch {c + 1}") if n_ch > 1 else 0
    scale = st.radio("Frequency scale", dsp.SCALES, index=1, horizontal=True, key="scale",
                     help="log and mel give low frequencies (where speech lives) more room.")
    nfft_opts = [256, 512, 1024, 2048, 4096]
    default_nfft = min(nfft_opts, key=lambda n: abs(n - sr * 0.064))  # ≈64 ms
    n_fft = st.select_slider("Detail: sharper in time ↔ sharper in pitch", nfft_opts, value=default_nfft,
                             format_func=lambda n: f"{1000 * n / sr:.0f} ms")
    db_range = st.slider("Contrast (dB shown dark → bright)", -160, 20, (-110, -20), 5)
    colormap = st.selectbox("Colours", list(figures.SPEC_COLORMAPS))
    wave_opacity = st.slider("Waveform overlay", 0.0, 1.0, 0.55, 0.05)
    shared_amp = st.checkbox("Same waveform scale on all tracks", True)
    show_slice = st.toggle("Show spectrum at playhead", False)
    st.divider()
    st.caption("Advanced")
    hop_div = st.select_slider("Hop (fraction of window)", [2, 4, 8, 16], value=4, format_func=lambda d: f"1/{d}")
    window = st.selectbox("Window", dsp.WINDOWS)
    n_mels = st.slider("Mel bands", 32, 256, 128, 16) if scale == "mel" else 128

params = dsp.StftParams(n_fft=n_fft, hop=n_fft // hop_div, window=window, scale=scale, n_mels=n_mels)
ptuple = dataclasses.astuple(params)

# ---------------------------------------------------------------------------- tracks popover
order_key, entry_roles = f"order_{tri_key}", {}
ss.setdefault(order_key, list(ROLES))


def move(pos: int, delta: int):
    o = ss[order_key]
    o[pos], o[pos + delta] = o[pos + delta], o[pos]


with tracks_pop:
    st.caption("Display order, top to bottom. Roles decide what is scored against what.")
    for pos, entry in enumerate(ss[order_key]):
        c = st.columns([0.9, 0.9, 2.2, 4], vertical_alignment="center")
        c[0].button("↑", key=f"up_{tri_key}_{entry}", on_click=move, args=(pos, -1), disabled=pos == 0)
        c[1].button("↓", key=f"dn_{tri_key}_{entry}", on_click=move, args=(pos, 1),
                    disabled=pos == len(ROLES) - 1)
        entry_roles[entry] = c[2].selectbox("role", ROLES, index=ROLES.index(entry),
                                            key=f"role_{tri_key}_{entry}", label_visibility="collapsed")
        c[3].caption(rel(audios[entry].path))

roles_valid = sorted(entry_roles.values()) == sorted(ROLES)
if not roles_valid:
    st.warning("Each role must be used exactly once (Tracks menu). Quality scores are hidden until then.")
by_role = {role: entry for entry, role in entry_roles.items()} if roles_valid else {}

# ---------------------------------------------------------------------------- main view
order = ss[order_key]

# Loudness matching: one gain per track, applied to what is drawn and played (never to the scores).
loud = {e: cached_loudness(str(audios[e].path), mtime(audios[e].path), channel) for e in ROLES}
gain_db = {e: 0.0 for e in ROLES}
if match_lufs:
    g, achieved = dsp.loudness_match_gains([loud[e][0] for e in ROLES], [loud[e][1] for e in ROLES], TARGET_LUFS)
    gain_db = {e: round(v, 2) for e, v in zip(ROLES, g)}
    note = (f"all tracks at **{achieved:.1f} LUFS**" if achieved == TARGET_LUFS else
            f"all tracks at **{achieved:.1f} LUFS** (lowered {TARGET_LUFS - achieved:.1f} dB from {TARGET_LUFS:g} "
            "so no track's peak exceeds −1 dBFS)")
    st.caption(f"🔊 Loudness matched: {note}. Gain: "
               + " · ".join(f"{entry_roles[e]} {gain_db[e]:+.1f} dB" for e in order)
               + ". Quality scores use the original files.")
sig = {e: audios[e].channel(channel) * 10 ** (gain_db[e] / 20) for e in ROLES}


def display_name(entry: str) -> str:
    name = audios[entry].path.name
    return f"{name} · {system_name}: {run.label}" if entry == "processed" and run and tri else name


with st.spinner("Drawing spectrograms…"):
    tracks, specs = [], {}
    for pos, entry in enumerate(order, start=1):
        a = audios[entry]
        spec = cached_spec(str(a.path), mtime(a.path), channel, ptuple)
        specs[entry] = dataclasses.replace(spec, db=spec.db + np.float32(gain_db[entry]))
        tracks.append(figures.TrackView(pos, entry_roles[entry], display_name(entry), sig[entry], sr, specs[entry]))
    fig = figures.stacked_figure(tracks, db_range, colormap, wave_opacity, shared_amp)


def level_stats(entry: str) -> dict:
    """Level readouts as heard: dB values shifted by the matching gain, plus loudness."""
    s = dict(cached_stats(str(audios[entry].path), mtime(audios[entry].path), channel))
    for k in ("peak_dbfs", "rms_dbfs", "max_rms_dbfs", "noise_floor_dbfs"):
        s[k] += gain_db[entry]
    s["lufs"] = loud[entry][0] + gain_db[entry]
    s["gain_db"] = gain_db[entry] if match_lufs else None
    return s


metrics_df = None
if roles_valid:
    paths = tuple(str(audios[by_role[r]].path) for r in ROLES)
    metrics_df = cached_metrics(paths, tuple(mtime(Path(p)) for p in paths), channel)

bits = {"PCM_16": "16-bit", "PCM_24": "24-bit", "PCM_32": "32-bit", "FLOAT": "32-bit float",
        "DOUBLE": "64-bit float"}.get(audios["clean"].subtype, audios["clean"].subtype)
chans = "mono" if n_ch == 1 else f"{n_ch} ch (showing ch {channel + 1})"
player(
    fig,
    [{"slot": t.slot, "role": t.role, "name": t.name,
      "wav": cached_wav_bytes(str(audios[e].path), mtime(audios[e].path), channel, gain_db[e])}
     for t, e in zip(tracks, order)],
    duration, scale, db_range, {t.slot: t.spec.freqs for t in tracks},
    stats=[level_stats(e) for e in order],
    scorecard=metrics.scorecard(metrics_df) if metrics_df is not None else None,
    format_line=f"{system_name} · {run.label if run and tri else 'manual'} · {sr / 1000:g} kHz · {bits} · "
                f"{chans} · {duration:.3f} s" + (f" · matched to {TARGET_LUFS:g} LUFS" if match_lufs else ""),
    show_slice=show_slice,
)

# ---------------------------------------------------------------------------- analysis drawer
with st.expander("Analysis & export", expanded=False):
    tab_d, tab_l, tab_e, tab_m, tab_x = st.tabs(
        ["Differences", "Long-term spectrum", "Loudness over time", "All metrics", "Export"])

    with tab_d:
        if roles_valid:
            diff_range = st.slider("Colour range (± dB)", 6, 60, 20, 2)
            sp, sc, sn = (specs[by_role[r]] for r in ("processed", "clean", "noisy"))
            diffs = [figures.DiffView("processed − clean", dsp.difference_db(sp, sc, db_range[0]), sp),
                     figures.DiffView("processed − noisy", dsp.difference_db(sp, sn, db_range[0]), sp)]
            st.plotly_chart(figures.difference_figure(diffs, sr, diff_range), use_container_width=True,
                            theme=None)
            st.caption("Red = more energy than the reference (residual noise, artefacts). "
                       "Blue = less (in *processed − clean*: speech that was removed).")
        else:
            st.info("Assign each role once to compare tracks.")

    with tab_l:
        lt = {entry_roles[e]: dsp.ltas(sig[e], sr, n_fft=max(2048, n_fft), window=window)
              for e in order}
        st.plotly_chart(figures.ltas_figure(lt, log_freq=scale != "linear"), use_container_width=True,
                        theme=None)
        st.caption("Average spectrum over the whole file.")

    with tab_e:
        env = {entry_roles[e]: dsp.rms_envelope(sig[e], sr) for e in order}
        st.plotly_chart(figures.envelope_figure(env), use_container_width=True, theme=None)
        st.caption("Loudness in 20 ms steps. Look for pumping or gating in the pauses.")

    with tab_m:
        if metrics_df is not None:
            st.dataframe(
                metrics_df.style.format({c: "{:.3f}" for c in ("noisy", "processed", "Δ (processed − noisy)")},
                                        na_rep="N/A"),
                hide_index=True, use_container_width=True)
            st.caption("Scored against the clean file. "
                       + " · ".join(f"**{k}**: {v}" for k, v in metrics.METRIC_INFO.items()))
        else:
            st.info("Assign each role once to compute metrics.")

    with tab_x:
        st.markdown("**Figure**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
        fmt = c1.radio("Format", ["png", "svg"], horizontal=True)
        width = c2.number_input("Width (px)", 800, 6000, 1800, 100)
        img_scale = c3.number_input("PNG scale", 1.0, 4.0, 2.0, 0.5, disabled=fmt == "svg")
        base = f"{tri_key.replace('/', '_')}_{export.timestamp()}"
        if c4.button("Render figure", use_container_width=True):
            with st.spinner("Rendering…"):
                # No 🎧 chip in exports (emoji fonts aren't available to the renderer).
                exp_fig = figures.stacked_figure(tracks, db_range, colormap, wave_opacity, shared_amp,
                                                 monitored_slot=0)
                ss["fig_export"] = (f"{base}.{fmt}", export.figure_image(exp_fig, fmt, int(width), img_scale))
                saved = export.save(ss["fig_export"][1], EXPORT_DIR, ss["fig_export"][0])
                if saved:
                    st.success(f"Saved to exports/{saved.name}")
        if "fig_export" in ss:
            name, data = ss["fig_export"]
            st.download_button(f"Download {name}", data, file_name=name,
                               mime="image/png" if name.endswith("png") else "image/svg+xml")

        st.divider()
        st.markdown("**Metrics CSV: this triplet**")
        if metrics_df is not None:
            out = metrics_df.copy()
            out.insert(0, "stem", tri_key)
            for r in ROLES:
                out[f"{r}_file"] = rel(audios[by_role[r]].path)
            out["sample_rate_hz"], out["channel"] = sr, channel + 1
            st.download_button("Download metrics CSV", out.to_csv(index=False), file_name=f"metrics_{base}.csv",
                               mime="text/csv")
        else:
            st.caption("Unavailable until each role is assigned exactly once.")

        st.divider()
        batch = [triplets[i] for i in run.triplets] if run else triplets
        n_batch_ok = sum(t.resolved for t in batch)
        st.markdown(f"**Batch metrics: all {len(batch)} files in "
                    f"{system_name + ' · ' + run.label if run else '`' + root + '`'}**")
        st.caption("Uses the discovered roles (not the Tracks menu). Incomplete or mismatched triplets are "
                   "listed with a status instead of scores.")
        if st.button(f"Run batch ({n_batch_ok} complete triplets)", disabled=n_batch_ok == 0):
            bar = st.progress(0.0, text="Starting…")
            df = export.batch_metrics(batch, root_path, channel,
                                      progress=lambda i, n, s: bar.progress(i / max(n, 1), text=f"{i}/{n} {s}"))
            name = f"metrics_batch_{export.timestamp()}.csv"
            ss["batch"] = (name, df)
            saved = export.save(df.to_csv(index=False), EXPORT_DIR, name)
            if saved:
                st.success(f"Saved to exports/{saved.name}")
        if "batch" in ss:
            name, df = ss["batch"]
            st.dataframe(df, hide_index=True, use_container_width=True)
            ok = df[df["status"] == "ok"] if "status" in df else df
            num = ok.select_dtypes("number")
            if not num.empty:
                st.markdown("Mean over OK triplets")
                st.dataframe(num.mean().to_frame("mean").T, use_container_width=True, hide_index=True)
            st.download_button(f"Download {name}", df.to_csv(index=False), file_name=name, mime="text/csv")
