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

- **Slots:** each row has a role dropdown and ↑/↓ buttons. The display order is up to you.
  Metrics and difference views follow the *role*, not the position. If a role is assigned
  twice, those views are disabled.
- **Transport:**
  - Play/pause with the button or `space`; `Home` rewinds.
  - Click any plot to seek.
  - The **Monitor** buttons (keys `1`–`3`) are exclusive: selecting one mutes the
    previous one. All tracks keep running sample-synchronously, so switching never
    interrupts playback or moves the playhead.
  - **Follow playhead** pages the view while you are zoomed in.
- **Zoom:** drag on any panel. The time axes are linked, and double-click resets.
- **Spectrum at t** (right panel): the spectra of all three tracks at the playhead,
  overlaid. While paused, it follows the mouse over the spectrograms instead.
- **Difference spectrograms:** both inputs are floored at the colour-range minimum.
  - `processed − clean`: red means residual noise or artefacts; blue means speech energy
    that was removed (over-suppression).
  - `processed − noisy`: shows what the processor took away.
- **Sidebar:** FFT size, hop, window, linear/log/mel axis, dB colour range and colormap.

### Tabs

| Tab | Contents |
|---|---|
| Metrics | SNR, SI-SDR, PESQ and STOI for noisy→clean and processed→clean, plus the Δ between them |
| Long-term spectrum | Welch PSD overlay of the three files |
| Level envelope | 20 ms RMS in dBFS; reveals gating, pumping and level loss |
| Signal stats | Peak and RMS dBFS, crest factor, clipped samples, DC offset |
| Export | Stacked figure as PNG/SVG, metrics CSV for this triplet, and batch metrics CSV over the whole folder |

Exported files are also written to `./exports` on the host.

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
