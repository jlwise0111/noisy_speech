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
cp .env.example .env          # then set DATA_DIR to your dataset folder (host path)
docker compose up --build     # first build takes a few minutes
```

Open **http://localhost:8501**. Stop the app with `Ctrl+C`, or with `docker compose down`.

To try the app without real data, generate synthetic demo triplets into `./data`:

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

**Strict comparison:** files must match exactly in sample rate, channel count and length
(in samples). If they don't, the app shows the exact mismatch and refuses to render.
Resample or trim the files before loading them.

## Using it

The screen has three areas: the stacked spectrograms on the left, a numbers panel on the right, and a transport bar below the spectrograms.

- **Spectrograms:** one panel per track, with the waveform overlaid as in iZotope RX.
  - Drag on any panel to zoom; the panels zoom together.
  - Click a panel to seek.
  - Click a panel's label to listen to that track.
- **Transport bar:**
  - Play/pause (`space`) and rewind (`Home`).
  - **Listen to**: an exclusive switch (keys `1`–`3`). All tracks keep playing in sync; only the selected one is audible.
  - Volume.
  - Zoom −/fit/+ (keys `-` `0` `+`) and *follow*.
- **Numbers panel:**
  - **Cursor:** time, frequency and level under the mouse.
  - **Quality vs clean:** Separation (SI-SDR), Quality (PESQ) and Intelligibility (STOI) for noisy and processed, with a ▲/▼ change badge.
  - **Levels per track:** peak, RMS, max RMS, noise floor (how quiet the pauses are), clipped samples and DC offset.
- **Header menus:**
  - **Tracks:** display order and role assignment. Scores follow the roles, not the order.
  - **⚙ Display:** frequency scale, time/pitch detail, contrast, colours, waveform overlay, an optional spectrum-at-playhead card, and advanced STFT settings.
  - Dataset settings are in the collapsed sidebar (`>` at the top left).
- **Analysis & export** (collapsed drawer below the main view):

| Tab | Contents |
|---|---|
| Differences | `processed − clean` and `processed − noisy` in dB. Red means more energy (residual noise or artefacts); blue means less (removed speech) |
| Long-term spectrum | Average spectrum of each file |
| Loudness over time | 20 ms RMS; reveals gating and pumping |
| All metrics | Full table, including SNR, with notes |
| Export | Stacked figure as PNG/SVG, metrics CSV for this triplet, and batch CSV over the whole folder. Files are also written to `./exports` |

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
| `viewer/audio_io.py` | Loading and strict validation |
| `viewer/dsp.py` | STFT, mel, LTAS, envelope, stats |
| `viewer/metrics.py` | Metrics |
| `viewer/figures.py` | Plotly figures |
| `viewer/export.py` | Figure export and CSV |
| `viewer/player/` | Web Audio + Plotly.js player that runs in the browser |

`PLAN.md` is the accepted design plan.
