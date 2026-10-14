"""Speech-isolation diagnostic viewer (Streamlit entry point).

Run with `docker compose up` and open http://localhost:8501.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from viewer import dsp, export, figures, metrics
from viewer.audio_io import Audio, TripletError, load_wav, to_wav_bytes, validate_strict
from viewer.dataset import DEFAULT_FOLDER_ALIASES, DEFAULT_SUFFIX_ALIASES, ROLES, discover, list_wavs
from viewer.player import player

DATA_DIR = os.environ.get("VIEWER_DATA_DIR", "data")
EXPORT_DIR = Path(os.environ.get("VIEWER_EXPORT_DIR", "exports"))
MANUAL = "__manual__"

st.set_page_config(page_title="Speech Isolation Diagnostic Viewer", page_icon="🔬", layout="wide")
st.markdown("<style>.block-container{padding-top:1.2rem;padding-bottom:1rem}</style>", unsafe_allow_html=True)


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


@st.cache_data(show_spinner="Computing metrics (PESQ, STOI)…", max_entries=48)
def cached_metrics(paths: tuple[str, str, str], mtimes: tuple, ch: int) -> pd.DataFrame:
    c, n, p = (cached_load(pth, m) for pth, m in zip(paths, mtimes))
    return metrics.compute_all(c.channel(ch), n.channel(ch), p.channel(ch), c.sr)


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


# ---------------------------------------------------------------------------- sidebar: dataset
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
ss.tri_idx = min(ss.tri_idx, len(options) - 1) if isinstance(ss.tri_idx, int) else ss.tri_idx


def step(delta: int):
    i = options.index(ss.tri_idx) if ss.tri_idx in options else 0
    ss.tri_idx = options[(i + delta) % len(options)]


with st.sidebar:
    n_ok = sum(t.resolved for t in triplets)
    st.caption(f"{len(all_wavs)} .wav files · {len(triplets)} triplets ({n_ok} complete)")
    c1, c2 = st.columns(2)
    c1.button("◀ Prev", on_click=step, args=(-1,), use_container_width=True)
    c2.button("Next ▶", on_click=step, args=(1,), use_container_width=True)
    st.selectbox("Triplet", options, key="tri_idx",
                 format_func=lambda i: "✎ Manual selection…" if i == MANUAL else triplets[i].label)

if not root_path.is_dir():
    st.error(f"Data folder `{root}` does not exist inside the container. Check DATA_DIR in `.env`.")
    st.stop()
if not all_wavs:
    st.info(f"No .wav files found under `{root}`. Put your audio in the folder set by DATA_DIR in `.env` "
            "(default `./data`), or generate demo data — see README.")
    st.stop()

# ---------------------------------------------------------------------------- resolve the triplet
tri = None if ss.tri_idx == MANUAL else triplets[ss.tri_idx]
tri_key = "manual" if tri is None else tri.stem
rel = lambda p: str(p.relative_to(root_path)) if p else "—"  # noqa: E731

files: dict[str, Path | None] = {r: None for r in ROLES} if tri is None else dict(tri.files)
unresolved = [r for r in ROLES if files[r] is None]
if unresolved:
    st.subheader("Resolve triplet" if tri else "Manual selection")
    if tri:
        st.warning(f"**{tri.stem}**: {', '.join(tri.issues)}. Pick the file for each unresolved role.")
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

# ---------------------------------------------------------------------------- sidebar: display
with st.sidebar:
    channel = st.selectbox("Channel", range(n_ch), format_func=lambda c: f"ch {c + 1}") if n_ch > 1 else 0
    st.header("Spectrogram")
    nfft_opts = [256, 512, 1024, 2048, 4096]
    default_nfft = min(nfft_opts, key=lambda n: abs(n - sr * 0.064))  # ≈64 ms
    n_fft = st.select_slider("FFT size", nfft_opts, value=default_nfft,
                             help=f"{default_nfft} ≈ 64 ms at {sr} Hz")
    hop_div = st.select_slider("Hop (fraction of FFT)", [2, 4, 8, 16], value=4, format_func=lambda d: f"1/{d}")
    window = st.selectbox("Window", dsp.WINDOWS)
    scale = st.radio("Frequency axis", dsp.SCALES, horizontal=True, key="scale")
    n_mels = st.slider("Mel bands", 32, 256, 128, 16) if scale == "mel" else 128
    db_range = st.slider("Colour range (dB)", -160, 20, (-110, -20), 5)
    colormap = st.selectbox("Colormap", figures.SPEC_COLORMAPS)
    st.caption(f"Δf = {sr / n_fft:.1f} Hz · Δt = {1000 * n_fft / hop_div / sr:.1f} ms hop · "
               f"window {1000 * n_fft / sr:.0f} ms")
    st.header("Diagnostics")
    show_diffs = st.checkbox("Difference spectrograms", True)
    diff_range = st.slider("Difference range (± dB)", 6, 60, 20, 2, disabled=not show_diffs)
    shared_amp = st.checkbox("Shared amplitude scale", True,
                             help="Same y-range on all waveforms so levels are visually comparable.")

params = dsp.StftParams(n_fft=n_fft, hop=n_fft // hop_div, window=window, scale=scale, n_mels=n_mels)
ptuple = dataclasses.astuple(params)

# ---------------------------------------------------------------------------- slots / roles
order_key, entry_roles = f"order_{tri_key}", {}
ss.setdefault(order_key, list(ROLES))


def move(pos: int, delta: int):
    o = ss[order_key]
    o[pos], o[pos + delta] = o[pos + delta], o[pos]


title = tri.stem if tri else "Manual selection"
st.markdown(f"### {title}  \n<span style='color:#656d76'>{sr} Hz · {n_ch} ch · {duration:.3f} s · "
            f"{audios['clean'].subtype}</span>", unsafe_allow_html=True)

for pos, entry in enumerate(ss[order_key]):
    a = audios[entry]
    stt = dsp.signal_stats(a.channel(channel), sr)
    c = st.columns([0.35, 0.45, 0.45, 1.6, 6.5], vertical_alignment="center")
    c[0].markdown(f"**{pos + 1}**")
    c[1].button("↑", key=f"up_{tri_key}_{entry}", on_click=move, args=(pos, -1), disabled=pos == 0)
    c[2].button("↓", key=f"dn_{tri_key}_{entry}", on_click=move, args=(pos, 1), disabled=pos == len(ROLES) - 1)
    entry_roles[entry] = c[3].selectbox("role", ROLES, index=ROLES.index(entry), key=f"role_{tri_key}_{entry}",
                                        label_visibility="collapsed")
    clip = f" · :red[**{stt['clipped_samples']} clipped**]" if stt["clipped_samples"] else ""
    c[4].markdown(f"`{rel(a.path)}` · peak {stt['peak_dbfs']:.1f} dBFS · RMS {stt['rms_dbfs']:.1f} dBFS{clip}")

roles_valid = sorted(entry_roles.values()) == sorted(ROLES)
if not roles_valid:
    st.warning("Each role must be assigned to exactly one slot. Metrics and difference views are disabled.")
by_role = {role: entry for entry, role in entry_roles.items()} if roles_valid else {}

# ---------------------------------------------------------------------------- main figure + player
with st.spinner("Computing spectrograms…"):
    tracks, specs = [], {}
    for pos, entry in enumerate(ss[order_key], start=1):
        a = audios[entry]
        spec = cached_spec(str(a.path), mtime(a.path), channel, ptuple)
        specs[entry] = spec
        tracks.append(figures.TrackView(pos, entry_roles[entry], a.path.name, a.channel(channel), sr, spec))

    diffs = []
    if show_diffs and roles_valid:
        sp, sc, sn = (specs[by_role[r]] for r in ("processed", "clean", "noisy"))
        diffs = [figures.DiffView("processed − clean", dsp.difference_db(sp, sc, db_range[0]), sp),
                 figures.DiffView("processed − noisy", dsp.difference_db(sp, sn, db_range[0]), sp)]
    fig = figures.stacked_figure(tracks, diffs, db_range, colormap, diff_range, shared_amp)

player(
    fig,
    [{"slot": t.slot, "role": t.role, "name": t.name,
      "wav": cached_wav_bytes(str(audios[e].path), mtime(audios[e].path), channel)}
     for t, e in zip(tracks, ss[order_key])],
    duration, scale, db_range, {t.slot: t.spec.freqs for t in tracks},
)
if diffs:
    st.caption("Difference rows (floored at the colour-range minimum): **processed − clean** red = energy the "
               "processor left or added (residual noise, artefacts), blue = speech energy removed "
               "(over-suppression). **processed − noisy** blue = what the processor removed.")

# ---------------------------------------------------------------------------- analysis tabs
tab_m, tab_ltas, tab_env, tab_stats, tab_exp = st.tabs(
    ["Metrics", "Long-term spectrum", "Level envelope", "Signal stats", "Export"])

metrics_df = None
with tab_m:
    if roles_valid:
        paths = tuple(str(audios[by_role[r]].path) for r in ROLES)
        metrics_df = cached_metrics(paths, tuple(mtime(Path(p)) for p in paths), channel)
        st.dataframe(
            metrics_df.style.format({c: "{:.3f}" for c in ("noisy", "processed", "Δ (processed − noisy)")},
                                    na_rep="N/A"),
            hide_index=True, use_container_width=True)
        st.caption("All metrics compare against the **clean** reference. "
                   + " · ".join(f"**{k}**: {v}" for k, v in metrics.METRIC_INFO.items()))
    else:
        st.info("Assign each role exactly once to compute metrics.")

with tab_ltas:
    lt = {entry_roles[e]: dsp.ltas(audios[e].channel(channel), sr, n_fft=max(2048, n_fft), window=window)
          for e in ss[order_key]}
    st.plotly_chart(figures.ltas_figure(lt, log_freq=scale != "linear"), use_container_width=True)
    st.caption("Welch PSD over the whole file. Gaps between processed and clean show spectral colouration; "
               "the processed−noisy gap shows where noise was removed.")

with tab_env:
    env = {entry_roles[e]: dsp.rms_envelope(audios[e].channel(channel), sr) for e in ss[order_key]}
    st.plotly_chart(figures.envelope_figure(env), use_container_width=True)
    st.caption("20 ms RMS windows, 10 ms hop. Look for pumping/gating in pauses and level loss during speech.")

with tab_stats:
    st.dataframe(pd.DataFrame([{"slot": i + 1, "role": entry_roles[e], "file": rel(audios[e].path),
                                **dsp.signal_stats(audios[e].channel(channel), sr)}
                               for i, e in enumerate(ss[order_key])]).round(4),
                 hide_index=True, use_container_width=True)

with tab_exp:
    st.markdown("**Stacked figure**")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
    fmt = c1.radio("Format", ["png", "svg"], horizontal=True)
    width = c2.number_input("Width (px)", 800, 6000, 1800, 100)
    img_scale = c3.number_input("PNG scale", 1.0, 4.0, 2.0, 0.5, disabled=fmt == "svg")
    base = f"{tri_key.replace('/', '_')}_{export.timestamp()}"
    if c4.button("Render figure", use_container_width=True):
        with st.spinner("Rendering with kaleido…"):
            ss["fig_export"] = (f"{base}.{fmt}", export.figure_image(fig, fmt, int(width), img_scale))
            saved = export.save(ss["fig_export"][1], EXPORT_DIR, ss["fig_export"][0])
            if saved:
                st.success(f"Saved to exports/{saved.name}")
    if "fig_export" in ss:
        name, data = ss["fig_export"]
        st.download_button(f"Download {name}", data, file_name=name,
                           mime="image/png" if name.endswith("png") else "image/svg+xml")

    st.divider()
    st.markdown("**Metrics CSV — this triplet**")
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
    st.markdown(f"**Batch metrics — all {len(triplets)} triplets in `{root}`**")
    st.caption("Uses the discovered role assignments (not the slot dropdowns). Incomplete or mismatched "
               "triplets are listed with a status instead of metrics.")
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
