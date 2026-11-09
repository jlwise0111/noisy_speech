"""Model player (Streamlit entry point): one file through every model, outputs switchable live.

Run with `make player` (Docker) and open http://localhost:8502.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import streamlit as st

from mplayer import DATASET_ROOT, MODELS_ROOT, engine, loudness, source
from mplayer.component import player
from viewer import dsp, figures

TARGET_LUFS = loudness.TARGET_LUFS
DB_RANGE = (-110, -20)
SR = engine.SR
FAMILY_COLORS = {"clean": "#4fd1a5", "noisy": "#f5a142", "ridge": "#f39c12", "wiener": "#5fb3d9",
                 "unet": "#a495ff", "oracle": "#9aa3b1"}

st.set_page_config(page_title="Model Player", page_icon="🎛️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2.6rem;padding-bottom:1rem;padding-left:2rem;padding-right:2rem;max-width:100%}
[data-testid="stStatusWidget"]{display:none!important}
[data-testid="stToolbar"]{display:none!important}
.stButton button p,.stCheckbox label p{white-space:nowrap}
.np{display:flex;flex-direction:column;gap:4px;padding:2px 0 2px 12px;border-left:3px solid #f39c12}
.np h3{font-size:1.4rem!important;line-height:1.15!important;margin:0!important;padding:0!important;font-weight:600}
.np-sub{color:#8a919e;font-size:.8rem;overflow-wrap:anywhere}
</style>""", unsafe_allow_html=True)
ss = st.session_state


# ---------------------------------------------------------------------------- cached work
@st.cache_resource(show_spinner="Loading models…")
def cached_models(models_root: str):
    specs, missing = engine.registry(models_root)
    return specs, missing, engine.load_all(specs)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_split(root: str, split: str):
    return source.list_split(root, split)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_pair(clean_path: str, noisy_path: str):
    return source.load_pair(Path(clean_path), Path(noisy_path))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_upload(data: bytes, name: str):
    return source.decode_upload(data, Path(name).suffix or ".wav")


def digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()


@st.cache_resource
def result_store() -> OrderedDict:
    """Model outputs of the last few inputs, keyed by SHA-256 of the 16 kHz audio (shared by all sessions)."""
    return OrderedDict()


def process(key: str, noisy: np.ndarray, clean: np.ndarray | None):
    store = result_store()
    if key in store:
        store.move_to_end(key)
        return store[key]
    specs_, _, enhancers = cached_models(str(MODELS_ROOT))
    bar = st.progress(0.0, text="Running every model in parallel…")
    result = engine.run_all(specs_, enhancers, noisy, clean,
                            on_done=lambda s, i, n: bar.progress(i / n, text=f"{s.label} done ({i}/{n})"))
    bar.empty()
    store[key] = result
    while len(store) > 8:
        store.popitem(last=False)
    return result


@st.cache_data(show_spinner=False, max_entries=256)
def cached_loudness(key: str, _x: np.ndarray) -> tuple[float, float]:
    return loudness.measure(_x, SR)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_spec(key: str, _x: np.ndarray, params: tuple) -> dsp.Spectrogram:
    return dsp.spectrogram(_x, SR, dsp.StftParams(*params))


# ---------------------------------------------------------------------------- source picker
specs, missing, _ = cached_models(str(MODELS_ROOT))
splits = source.available_splits(DATASET_ROOT)

mode = st.radio("Audio", ["Upload a .wav", "Dataset file"], horizontal=True, key="mode",
                index=1 if splits else 0,
                help="Upload any .wav (any sample rate, mono or stereo), or pick a file from the Valentini "
                     "train/test sets. Everything is converted to 16 kHz mono, the rate the models use.")

noisy = clean = None
title, subtitle, notes = "", "", []
if mode == "Upload a .wav":
    up = st.file_uploader("Upload a .wav", type=["wav"], label_visibility="collapsed")
    if up is None:
        st.info("Upload a .wav file to run it through every model.")
        st.stop()
    try:
        noisy, file_sr = cached_upload(up.getvalue(), up.name)
    except Exception as e:
        st.error(f"Could not read **{up.name}**: {e}")
        st.stop()
    noisy, cut = source.cap_length(noisy)
    if cut:
        notes.append(f"Only the first {source.MAX_SECONDS:g} s are used.")
    title, subtitle = up.name, f"uploaded · {file_sr / 1000:g} kHz → 16 kHz mono"
else:
    if not splits:
        st.warning("No Valentini dataset found. Set `DATASET` (Makefile / `.env`) to the "
                   "`valentini_noisy_speech_database` folder and restart, or upload a file instead.")
        st.stop()
    c_split, c_prev, c_next, c_file = st.columns([1.4, 0.35, 0.35, 3], vertical_alignment="bottom")
    split = c_split.selectbox("Set", splits, key="split")
    pairs = cached_split(DATASET_ROOT, split)
    if not pairs:
        st.warning(f"No files in {split}.")
        st.stop()
    stems = [p[0] for p in pairs]
    if ss.get("stem") not in stems:
        ss.stem = stems[0]

    def step(d: int):
        ss.stem = stems[(stems.index(ss.stem) + d) % len(stems)]

    c_prev.button("◀", on_click=step, args=(-1,), help="Previous file", use_container_width=True)
    c_next.button("▶", on_click=step, args=(1,), help="Next file", use_container_width=True)
    c_file.selectbox(f"File ({len(stems)})", stems, key="stem")
    stem, clean_path, noisy_path = pairs[stems.index(ss.stem)]
    clean, noisy = cached_pair(str(clean_path), str(noisy_path))
    title, subtitle = stem, f"{split} · {noisy_path.parent.name} (48 kHz → 16 kHz)"
    if split in source.SEEN_BY:
        seen = ", ".join(s.label for s in specs if s.trained_on in source.SEEN_BY[split])
        notes.append(f"Training file: {seen or 'some models'} may have seen it in training.")

if len(noisy) < SR // 4:
    st.error("The file is shorter than 0.25 s; too short to process.")
    st.stop()

# ---------------------------------------------------------------------------- process
key = digest(noisy, clean)
outputs, errors = process(key, noisy, clean)

st.markdown(f"<div class='np'><h3>{title}</h3><div class='np-sub'>{subtitle} · {len(noisy) / SR:.2f} s · "
            f"{len(outputs)} model outputs</div></div>", unsafe_allow_html=True)
for m in missing:
    notes.append(f"Skipped {m}.")
if clean is None and any(s.needs_clean for s in specs):
    notes.append("Oracle needs the clean recording, so it is only shown for dataset files.")
for k, e in errors.items():
    st.error(f"{k} failed: {e}")
if notes:
    st.caption("  \n".join(notes))

# ---------------------------------------------------------------------------- tracks
rows = []  # (key, label, description, family, signal)
if clean is not None:
    rows.append(("clean", "Clean", "clean reference", "clean", clean))
rows.append(("noisy", "Noisy input", "unprocessed input", "noisy", noisy))
for s in specs:
    if s.key in outputs:
        rows.append((s.key, s.label, s.description, s.family, outputs[s.key]))

# Loudness normalization gains for every track. They are applied in the browser as the FIRST stage of
# the playback chain (per-track gain → solo/diff mixer), switched by the player's "Match loudness"
# control, which starts on. The tracks themselves are sent at their original level.
norm = loudness.normalize({k: x for k, *_, x in rows}, SR, TARGET_LUFS,
                          measured={k: cached_loudness(f"{key}:{k}", x) for k, *_, x in rows})
gain_db = norm.gain_db
sig = {k: x for k, *_, x in rows}

n_fft = 1024  # ≈64 ms at 16 kHz, as in the diagnostic viewer
params = dsp.StftParams(n_fft=n_fft, hop=n_fft // 4, window="hann", scale="log")
figures.TRACK_PX = 140  # stacked_figure reads this module constant: smaller panels so ~11 tracks fit
with st.spinner("Drawing spectrograms…"):
    views = []
    for slot, (k, label, desc, fam, x) in enumerate(rows, start=1):
        spec = cached_spec(f"{key}:{k}", x, dataclasses.astuple(params))  # original level; the player shifts it
        views.append(figures.TrackView(slot, label, desc, sig[k], SR, spec))
    solo_slot = next(v.slot for v, r in zip(views, rows) if r[0] == "noisy")
    fig = figures.stacked_figure(views, DB_RANGE, monitored_slot=solo_slot)
    for a in fig.layout.annotations:  # label text: "<b>U-Net 28spk</b> · description" (not upper-cased)
        if a.name and a.name.startswith("label"):
            v = views[int(a.name[5:]) - 1]
            a.text = f"{'🎧 ' if v.slot == solo_slot else ''}<b>{v.role}</b> · {v.name}"

player(
    fig,
    [{"slot": v.slot, "label": label, "description": desc, "color": FAMILY_COLORS[fam], "x": sig[k],
      "gain_db": gain_db[k]}
     for v, (k, label, desc, fam, _) in zip(views, rows)],
    SR, len(noisy), {v.slot: v.spec.freqs for v in views},
    levels=[{"lufs": norm.lufs_in[k], "peak_dbfs": norm.peak_in[k]} for k, *_ in rows],
    format_line=f"{title} · 16 kHz mono · {len(noisy) / SR:.3f} s · {len(rows)} tracks in lock-step",
    norm_lufs=norm.achieved_lufs,
    solo=solo_slot - 1,
    debug="debug" in st.query_params,
)
