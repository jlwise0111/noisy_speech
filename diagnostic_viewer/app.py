"""Speech-isolation diagnostic viewer (Streamlit entry point).

Run with `docker compose up` and open http://localhost:8501.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from viewer import dsp, export, figures, metrics
from viewer.audio_io import Audio, TripletError, load_wav, to_wav_bytes, validate_strict
from viewer.dataset import DEFAULT_FOLDER_ALIASES, DEFAULT_SUFFIX_ALIASES, ROLES, discover, list_wavs
from viewer.player import player

DATA_DIR = os.environ.get("VIEWER_DATA_DIR", "data")
EXPORT_DIR = Path(os.environ.get("VIEWER_EXPORT_DIR", "exports"))
MANUAL = "__manual__"

st.set_page_config(page_title="Speech Isolation Viewer", page_icon="🎧", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:2.6rem;padding-bottom:1rem;max-width:100%}
h3{margin:0!important;padding:0!important}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------- cached work
@st.cache_data(show_spinner="Scanning dataset…")
def cached_discover(root: str, layout: str, folder_aliases: tuple, suffix_aliases: tuple, nonce: int):
    return discover(root, layout, {r: list(a) for r, a in folder_aliases}, {r: list(a) for r, a in suffix_aliases})


@st.cache_data(show_spinner=False)
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
def cached_wav_bytes(path: str, mtime: float, ch: int) -> bytes:
    a = cached_load(path, mtime)
    return to_wav_bytes(a.channel(ch), a.sr)


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
ss.setdefault("nonce", 0)
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
options = list(range(len(triplets))) + [MANUAL]
if ss.tri_idx not in options:  # dataset changed under us
    ss.tri_idx = options[0]
n_ok = sum(t.resolved for t in triplets)
with st.sidebar:
    st.caption(f"{len(all_wavs)} .wav files · {len(triplets)} triplets ({n_ok} complete)")


def step(delta: int):
    i = options.index(ss.tri_idx) if ss.tri_idx in options else 0
    ss.tri_idx = options[(i + delta) % len(options)]


# ---------------------------------------------------------------------------- header row
h_title, h_prev, h_next, h_pick, h_tracks, h_disp = st.columns([3.2, 0.45, 0.45, 3, 1, 1],
                                                               vertical_alignment="center")
h_prev.button("◀", on_click=step, args=(-1,), help="Previous triplet", use_container_width=True)
h_next.button("▶", on_click=step, args=(1,), help="Next triplet", use_container_width=True)
h_pick.selectbox("Triplet", options, key="tri_idx", label_visibility="collapsed",
                 format_func=lambda i: "✎ Manual selection…" if i == MANUAL else triplets[i].label)
tracks_pop = h_tracks.popover("Tracks", use_container_width=True)
display_pop = h_disp.popover("⚙ Display", use_container_width=True)

tri = None if ss.tri_idx == MANUAL else triplets[ss.tri_idx] if triplets else None
tri_key = "manual" if tri is None else tri.stem
h_title.markdown(f"### {tri.stem if tri else 'Manual selection'}")

if not root_path.is_dir():
    st.error(f"Data folder `{root}` does not exist inside the container. Check DATA_DIR in `.env` "
             "(dataset settings are in the sidebar, top-left `>`).")
    st.stop()
if not all_wavs:
    st.info(f"No .wav files found under `{root}`. Put your audio in the folder set by DATA_DIR in `.env` "
            "(default `./data`), or generate demo data — see README.")
    st.stop()

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
with st.spinner("Drawing spectrograms…"):
    tracks, specs = [], {}
    for pos, entry in enumerate(order, start=1):
        a = audios[entry]
        specs[entry] = cached_spec(str(a.path), mtime(a.path), channel, ptuple)
        tracks.append(figures.TrackView(pos, entry_roles[entry], a.path.name, a.channel(channel), sr,
                                        specs[entry]))
    fig = figures.stacked_figure(tracks, db_range, colormap, wave_opacity, shared_amp)

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
      "wav": cached_wav_bytes(str(audios[e].path), mtime(audios[e].path), channel)}
     for t, e in zip(tracks, order)],
    duration, scale, db_range, {t.slot: t.spec.freqs for t in tracks},
    stats=[cached_stats(str(audios[e].path), mtime(audios[e].path), channel) for e in order],
    scorecard=metrics.scorecard(metrics_df) if metrics_df is not None else None,
    format_line=f"{sr / 1000:g} kHz · {bits} · {chans} · {duration:.3f} s",
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
        lt = {entry_roles[e]: dsp.ltas(audios[e].channel(channel), sr, n_fft=max(2048, n_fft), window=window)
              for e in order}
        st.plotly_chart(figures.ltas_figure(lt, log_freq=scale != "linear"), use_container_width=True,
                        theme=None)
        st.caption("Average spectrum over the whole file.")

    with tab_e:
        env = {entry_roles[e]: dsp.rms_envelope(audios[e].channel(channel), sr) for e in order}
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
        st.markdown(f"**Batch metrics: all {len(triplets)} triplets in `{root}`**")
        st.caption("Uses the discovered roles (not the Tracks menu). Incomplete or mismatched triplets are "
                   "listed with a status instead of scores.")
        if st.button(f"Run batch ({n_ok} complete triplets)", disabled=n_ok == 0):
            bar = st.progress(0.0, text="Starting…")
            df = export.batch_metrics(triplets, root_path, channel,
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
