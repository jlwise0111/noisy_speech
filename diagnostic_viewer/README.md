# Speech-Isolation Diagnostic Viewer

A local web app for **diagnosing** (not editing) neural speech-isolation output. It loads a
**clean / noisy / processed** `.wav` triplet and stacks the files vertically, each as a
waveform plus a spectrogram on a shared, linked time axis. The three tracks play in
sync, and only one is audible at a time. The app also shows objective metrics and
diagnostic views.

Built with Streamlit and Plotly, and run with Docker Compose so it behaves the same on
macOS, Windows and Linux.

## Quick start

```bash
cd diagnostic_viewer
docker compose up --build     # first build takes a few minutes
```

By default the app scans the **whole project folder** (`DATA_DIR=..`), so the outputs of every
system (the neural network, the Wiener baseline and the Ridge baseline) appear side by side. To
scan somewhere else, `cp .env.example .env` and set `DATA_DIR` to a host path.

Open **http://localhost:8501**. Stop the app with `Ctrl+C`, or with `docker compose down`.

To try the app without real data, generate synthetic demo triplets into `./data` (they are listed
under the system **Other**):

```bash
docker compose run --rm -v ./data:/out viewer python scripts/make_demo_data.py --out /out
```

## Data layout

The folder `DATA_DIR` is mounted **read-only** at `/data`. Triplets are found automatically
in either layout, at any depth:

| Layout     | Example                                                                        |
|------------|--------------------------------------------------------------------------------|
| subfolders | `exp1/clean/p232_001.wav`, `exp1/noisy/p232_001.wav`, `exp1/enhanced/p232_001.wav` |
| suffix     | `p232_001_clean.wav`, `p232_001_noisy.wav`, `p232_001_processed.wav`           |

Role aliases can be edited in the sidebar under **Role naming**. The defaults are:
- folders: `clean`, `noisy`, and `processed`/`enhanced`/`denoised`/`output`
- suffixes: `clean`, `noisy`, and `processed`/`enhanced`/`denoised`

A triplet with a missing file or more than one candidate per role is marked ⚠. When you
select it, the app asks you to pick the files. **✎ Manual selection…** lets you build a
triplet from any three files.

### Systems and runs

Every folder that contains a triplet is a **run**, for example `linear_baseline/outputs/ridge_28spk`.
Folders whose name starts with `smoke_` (the few-file runs from `make pipeline-smoke`) are not
shown, so every run listed is a real result.
Each run is assigned to a **system** by the words in its folder path:

| System | Folder words | Example run folder |
|---|---|---|
| Neural network | `neural`, `nn`, `dnn`, `network`, `unet` | `unet_model/outputs/unet_28spk/` |
| Wiener baseline | `wiener`, `oracle` | `wiener_baseline/outputs/wiener_leading_test/` |
| Ridge baseline | `ridge`, `linear` | `linear_baseline/outputs/ridge_28spk/` |
| Other | anything else | `diagnostic_viewer/data/demo/subfolders/` |

The rules are in `viewer/systems.py`. The header always lists all three project systems; one with
no outputs yet shows where to write them. All three systems write their outputs at 16 kHz. Each
run also carries its own `clean/` and `noisy/` copies, so a run at any other sample rate would
still be compared strictly against its own references.

The folder is scanned when a browser tab opens (or is reloaded). Outputs written while the viewer
is open, for example a model that finishes training later, appear after a page reload or
**↻ Rescan folder** in the sidebar; the selected file stays selected.

**Strict comparison:** files must match exactly in sample rate, channel count and length
(in samples). If they don't, the app shows the exact mismatch and refuses to render.
Resample or trim the files before loading them.

## Using it

The screen has three areas: the stacked spectrograms on the left, a numbers panel on the right, and a transport bar above the spectrograms.

- **Header, top row:**
  - **System** (Neural network / Wiener baseline / Ridge baseline) and **Run** (shown by run name; hover ⓘ for every run's full path and file count). Switching either keeps the same file (e.g. `p232_001`) selected when the new run has it, so you can A/B systems on identical audio.
  - **Match loudness** toggle (−20 LUFS; details in its ⓘ tooltip): plays *and draws* every track at the same integrated loudness (ITU-R BS.1770 / EBU R128 gating), so a processor that makes things quieter can be judged on sound quality alone. If reaching −20 LUFS would push any track's peak above −1 dBFS, all three are lowered by the same amount and the achieved level is shown. Waveforms, spectrograms, levels, long-term spectrum and loudness over time all follow the gain. **Quality scores always use the original files.**
- **Header, second row:** a coloured badge with the **system** and **run** you are looking at and listening to, the file name, ◀/▶ and the file picker. The processed track's label and the line under the transport also name the system and run.

- **Spectrograms:** one panel per track (clean, noisy, processed, top to bottom), with the waveform overlaid as in iZotope RX.
  - Hover over a panel for a tooltip beside the cursor with the time, frequency, level and track under it.
  - Drag on any panel to zoom; the panels zoom together.
  - Click a panel to seek.
  - Click a panel's label to listen to that track.
- **Transport bar:**
  - Play/pause (`space`) and rewind (`Home`).
  - **Listen to**: an exclusive switch (keys `1`–`3`). All tracks keep playing in sync; only the selected one is audible.
  - Volume.
  - Zoom −/fit/+ (keys `-` `0` `+`) and *follow*.
- **Numbers panel:**
  - **Quality vs clean:** Separation (SI-SDR), Quality (PESQ) and Intelligibility (STOI) for noisy and processed, with a ▲/▼ change badge.
  - **Levels per track:** loudness (LUFS), match gain, peak, RMS, max RMS, noise floor (how quiet the pauses are), clipped samples and DC offset.
- **Display settings are fixed:** log frequency axis, ≈64 ms Hann window with 1/4 hop, −110 to −20 dB
  contrast, Ember colours, waveform overlay at 55 % on a shared scale. Change them in `app.py`
  (display settings section) if needed.
- Dataset settings are in the collapsed sidebar (`>` at the top left).
- **Analysis** (collapsed drawer below the main view):

| Tab | Contents |
|---|---|
| Long-term spectrum | Average spectrum of each file |
| Loudness over time | 20 ms RMS; reveals gating and pumping |
| All metrics | Full table, including SNR, with notes |

### Metric notes

- **SI-SDR** removes the mean and is scale-invariant. **SNR** is not, so a gain change in
  the processor lowers SNR but leaves SI-SDR unchanged.
- **PESQ** only runs at 8 kHz (narrowband) or 16 kHz (wideband). Other sample rates are
  resampled to 16 kHz *for scoring only*, and the Metrics table notes this. Silent or
  speech-free input yields N/A with the reason.
- **STOI** resamples to 10 kHz internally.
- Batch mode uses the discovered roles, not the slot dropdowns.

## Development

```bash
docker compose run --rm -v ./:/app viewer pytest -q
```

| Path | Contents |
|---|---|
| `app.py` | Streamlit UI |
| `viewer/dataset.py` | Triplet discovery |
| `viewer/systems.py` | Grouping runs into systems (neural network, baselines) |
| `viewer/audio_io.py` | Loading and strict validation |
| `viewer/dsp.py` | STFT, mel, LTAS, envelope, stats, BS.1770 loudness and matching |
| `viewer/metrics.py` | Metrics |
| `viewer/figures.py` | Plotly figures |
| `viewer/export.py` | Figure export and CSV |
| `viewer/player/` | Web Audio + Plotly.js player that runs in the browser |

`PLAN.md` is the accepted design plan.
