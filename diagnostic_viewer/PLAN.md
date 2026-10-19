# Speech-Isolation Diagnostic Viewer — Plan

## Context
The group project evaluates neural-network speech isolation. We need a local, diagnostic (not editing) tool that loads a **clean / noisy / processed** `.wav` triplet, shows each as a waveform + spectrogram stacked vertically on a shared time axis, plays them back in real time with a synced playhead, and reports objective metrics. The repo (`noisy_speech`) is empty apart from a README. All work goes on a new branch, `feature/diagnostic-viewer`, not `main`.

### Decisions you made
- **Streamlit**, no Gradio.
- **Docker Compose only**: `docker compose up`, then open `http://localhost:8501`.
- **Folder browse only.** Triplets are discovered automatically using either layout: subfolders `root/{clean,noisy,processed}/<stem>.wav` or suffixes `root/<stem>_{clean,noisy,processed}.wav` (names configurable). If a triplet is incomplete or ambiguous, you pick the file manually.
- **Strict mismatch**: a triplet is refused, with a clear reason, if sample rate, channel count or sample length differ.
- A **role dropdown per slot**, plus up/down reordering. Metrics follow the role, not the position.
- **Shared transport + solo**: one play/pause/seek control and one playhead; solo switches which track you hear without interrupting playback.
- **Metrics**: SNR, SI-SDR (plus SI-SDR improvement over noisy), PESQ and STOI.
- **Diagnostics**: difference spectrograms, long-term average spectrum (LTAS), RMS envelope, signal stats, hover readouts, a spectrum slice at the playhead, and linear/log/mel axes with adjustable STFT parameters.
- **Plotly** rendering with zoom linked across tracks.
- **Export**: figure PNG/SVG and a metrics CSV, including a batch CSV over the whole folder.
- The data folder defaults to `./data`; `DATA_DIR` in `.env` can point at any host path. It is mounted read-only.

## Architecture
All work lives in a single subfolder of the repo, `diagnostic_viewer/`. Compose, `.env` and the `./data` and `./exports` paths all resolve relative to that folder.

```
noisy_speech/diagnostic_viewer/
├── PLAN.md                 # copy of this accepted plan, saved before any other work
├── docker-compose.yml      # service "viewer", port 8501, ${DATA_DIR:-./data}:/data:ro, ./exports:/exports
├── Dockerfile              # python:3.12-slim + build-essential (pesq), libsndfile1
├── .env.example            # DATA_DIR=/path/to/dataset
├── requirements.txt        # streamlit, numpy, scipy, soundfile, librosa, plotly, kaleido==0.2.1, pesq, pystoi, pandas, pytest
├── app.py                  # Streamlit entry: sidebar + main layout
├── viewer/
│   ├── dataset.py          # triplet discovery (both layouts), ambiguity reporting
│   ├── audio_io.py         # soundfile load, strict validation → TripletError(reason)
│   ├── dsp.py              # STFT/mel/log, dB, difference specs, LTAS, RMS envelope, stats
│   ├── metrics.py          # snr, si_sdr, pesq (resampled to 16k for scoring only), stoi
│   ├── figures.py          # Plotly figure builders (stacked tracks, LTAS, envelope)
│   ├── export.py           # PNG/SVG via kaleido, CSV writers, batch runner
│   └── player/
│       ├── __init__.py     # declare component (streamlit.components.v1.html)
│       └── player.html.j2  # Web Audio + Plotly.js template
├── scripts/make_demo_data.py  # synthesizes a demo triplet in both layouts
└── tests/                  # pytest: dataset, audio_io, dsp, metrics
```

### Key design points
1. **Real-time playback + playhead (the one custom piece).** Streamlit reruns can't move a playhead smoothly, so playback and the stacked plots live in one HTML component rendered with `components.html`:
   - The three decoded tracks are passed as base64 WAV and decoded into **Web Audio** `AudioBuffer`s. One `AudioBufferSourceNode` per track runs into its own `GainNode`, all started at the same `currentTime`. Solo just sets the gains to 0 or 1, so A/B/C switching is sample-synchronous and click-free (a short gain ramp avoids clicks).
   - The Plotly figure JSON is built in Python (`figures.py`) and rendered with Plotly.js. To work offline, the library is inlined from `plotly.offline.get_plotlyjs()` rather than loaded from a CDN.
   - A `requestAnimationFrame` loop moves a vertical playhead shape on every subplot, using `Plotly.relayout` throttled to about 30 fps. Clicking any subplot seeks to that time. The spacebar toggles play.
   - The **spectrum slice at the playhead** is drawn in a side panel: JS reads the column at the current frame from each spectrogram's `z` matrix and overlays the three traces.
2. **Layout of the main area:** three slots, each with a role dropdown, ↑/↓ buttons and a waveform row on top of a spectrogram row. All rows share x-axes (`matches='x'`), so zooming one zooms all. The difference spectrograms (processed − clean, processed − noisy, in dB, on a diverging colormap centred at 0) are optional rows underneath. Below the stack are a metrics table and an LTAS overlay plus RMS envelope (each with its own tab).
3. **Sidebar:**
   - Dataset: the root folder (default `/data`), a layout override, the role folder names and suffixes, and a triplet list with previous/next.
   - When a triplet is unresolved, a per-role file selectbox appears.
   - STFT settings: n_fft, hop, window, frequency scale (linear/log/mel with n_mels), dB floor and ceiling, and colormap.
   - Export buttons.
4. **Performance:** expensive work goes through `st.cache_data`, keyed on file path + mtime + STFT parameters. Spectrograms are decimated to at most about 2000 time frames before being sent to Plotly. Speech clips of 2–15 s render comfortably.
5. **Metrics (`metrics.py`):**
   - SNR: clean taken as the reference, the residual as noise.
   - SI-SDR: the standard projection formula, written in numpy.
   - PESQ: mode `wb` at 16 kHz or `nb` at 8 kHz. Audio is resampled to 16 kHz only for scoring, and the resampling is shown in the UI.
   - STOI: via `pystoi`.

   Each metric is computed for noisy→clean and for processed→clean, plus the processed−noisy delta. Failures (e.g. PESQ "no utterance detected") appear as N/A with the reason; they never crash the app.
6. **Strict validation (`audio_io.py`):** mismatches show `st.error` with the exact values (e.g. "processed: 48000 Hz ≠ clean: 16000 Hz"), and nothing is rendered for that triplet. Multichannel files are refused unless all three have the same channel count; channel 0 is displayed and there is a channel selector.
7. **Stats per track:** duration, sample rate, peak dBFS, RMS dBFS, crest factor, clipped-sample count, and DC offset.
8. **Export:**
   - The static stacked figure goes to PNG/SVG with kaleido (pinned to 0.2.1 because it bundles Chromium, so the image needs no browser) and is offered with `st.download_button`.
   - The metrics CSV covers the current triplet.
   - A batch run computes metrics for every resolvable triplet, with a progress bar, and writes `exports/metrics_<timestamp>.csv` as well as offering it for download.

## Implementation steps
1. `git checkout -b feature/diagnostic-viewer`.
2. Create `diagnostic_viewer/` and save this accepted plan as `diagnostic_viewer/PLAN.md` **before any other work**. Every later step happens inside that folder.
3. Add the Docker scaffolding (Dockerfile, compose, `.env.example`, `.dockerignore`, requirements) and a `.gitignore` covering `data/`, `exports/` and `.env`.
4. Write `dataset.py` and `audio_io.py`, with tests.
5. Write `dsp.py` and `metrics.py`, with tests on synthetic signals.
6. Write `figures.py`, then the `player` component.
7. Write `app.py` to wire up the sidebar, slots and role mapping, the metrics and the tabs.
8. Write `export.py` (figure, CSV, batch).
9. Write `scripts/make_demo_data.py` and the README (setup, `.env`, folder layouts, metric caveats).
10. Commit on the branch.

All commands in the Verification section are run from `diagnostic_viewer/`.

## Verification
- Run `docker compose build && docker compose run --rm viewer pytest`. The tests check:
  - SI-SDR(clean, clean) is above 100 dB, and SI-SDR of clean plus 0 dB noise is about 0 dB.
  - PESQ and STOI are higher for processed than for noisy on the demo data.
  - Triplet discovery handles both layouts, plus a missing file and an ambiguous file.
  - Strict validation rejects a sample-rate mismatch.
- Run `docker compose run --rm viewer python scripts/make_demo_data.py`, then `docker compose up` and open `localhost:8501` in a browser. Check that:
  - The triplet loads.
  - Playback runs and the playhead moves across all panes.
  - Solo switching works mid-playback.
  - Zoom is linked across tracks, and hover shows time, frequency and dB.
  - Role reordering keeps metrics attached to the correct roles.
  - PNG, SVG and CSV export, including batch, work.
- Copy in a deliberately mismatched file and confirm the strict error message appears.
