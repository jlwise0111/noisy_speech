# Diagnostic Viewer — UI Redesign Plan (v2: "RX-style, spectrogram-first")

## Context
v1 (`diagnostic_viewer/PLAN.md`) is implemented and committed on `feature/diagnostic-viewer`. It works, but in use it is visually overwhelming:
- A track-role table sits above the player.
- Waveform and spectrogram are separate rows, each with its own axes.
- There is a giant dB colorbar and an always-on spectrum-slice chart.
- Difference spectrograms, a metrics table and long captions are inline.
- The sidebar is full of FFT jargon.

The goal is a **less technical** viewer modelled on iZotope RX:
- The spectrograms dominate the screen.
- The waveform is overlaid on each spectrogram.
- There is a dark theme.
- A compact numbers panel sits top right, like RX's *Waveform Statistics*.
- A transport bar runs along the bottom.

Everything technical stays available, but collapsed. Diagnostic-not-editing still holds.

**Scope of this step:** after approval, only save this plan into the project as `diagnostic_viewer/PLAN_v2_ui.md`. Implementation happens later, when you ask.

### Decisions you made
- **Waveform overlaid on the spectrogram**, as in RX, with an opacity control.
- **Dark, RX-like theme**: warm black→orange→yellow spectrogram, a bright playhead, and a bottom transport bar with a large time readout.
- **Top-right panel** with three blocks: the cursor readout, a quality scorecard, and level stats per track.
- **Technical material** goes in one collapsed **"Analysis & export" drawer** below the main view. FFT settings move into a small ⚙ popover.

## Target layout
```
┌ demo_000  ◀ ▶  [triplet ▾]                         [Tracks ▾] [⚙ Display ▾] ┐
├──────────────────────────────────────────────────────┬──────────────────────┤
│ 🎧 CLEAN · demo_000.wav                              │ CURSOR               │
│ ███ spectrogram + translucent waveform ███████████   │ 1.234 s · 818 Hz     │
│──────────────────────────────────────────────────────│ −22.7 dB · noisy     │
│ 🎧 NOISY · demo_000.wav                (monitored)   ├──────────────────────┤
│ ███████████████████████████████████████████████████  │ QUALITY  noisy  proc │
│──────────────────────────────────────────────────────│ Separation 0.9  14.9▲│
│ 🎧 PROCESSED · demo_000.wav                          │ Quality    1.10 1.66▲│
│ ███████████████████████████████████████████████████  │ Intellig.  .83  .98 ▲│
│ 0:00   0:01   0:02   0:03   0:04            h:m:s    ├──────────────────────┤
├──────────────────────────────────────────────────────┤ LEVELS  cln noi prc  │
│ 00:00:01.234  ⏮ ▶  Monitor:[clean|noisy|proc]  vol  │ Peak    -6  -5  -6   │
│ 16 kHz · 16-bit · mono · 4.000 s        zoom − fit + │ RMS ... Floor ...    │
└──────────────────────────────────────────────────────┴──────────────────────┘
▸ Analysis & export   (collapsed: differences · spectrum · level · metrics · export · batch)
```

## Changes by file (all inside `diagnostic_viewer/`)

1. **`.streamlit/config.toml`** (new): dark base theme.
   - Background `#14161b`, secondary `#1d2027`, text `#d6d9df`, primary (accent) orange `#f39c12`.
   - Set `initial_sidebar_state="collapsed"` in `app.py`; the sidebar keeps only dataset settings (folder, layout, aliases, rescan).

2. **`app.py`**: restructure, keeping the existing logic (discovery, resolve, strict validation, cached loaders).
   - **Header row:** the triplet name, ◀ ▶ buttons and the triplet selectbox.
   - **`st.popover("Tracks")`** replaces the always-visible slot table; the existing role dropdowns and ↑/↓ logic move inside unchanged.
   - **`st.popover("⚙ Display")`** with friendly labels:
     - "Frequency detail" maps to FFT size.
     - "Contrast" maps to the dB range.
     - "Frequency scale" defaults to **log**.
     - Colour theme.
     - Waveform opacity.
     - "Show spectrum at playhead" (off by default).
     - Hop and window go into an inner "Advanced" sub-expander.
   - **Main view:** the one `player(...)` component, which now contains the spectrograms, the right-hand panel and the transport bar.
   - **`st.expander("Analysis & export", expanded=False)`** holds the existing tabs:
     - Differences: moved out of the main figure into a separate static Plotly chart.
     - Long-term spectrum.
     - Level envelope.
     - Metrics detail: the full table with notes.
     - Export: figure, CSV and batch.
   - Captions become one short line each.

3. **`viewer/figures.py`**
   - **`stacked_figure` → one row per track:** a heatmap, plus a waveform `Scatter` on an overlaid secondary y-axis (`make_subplots(specs=[[{"secondary_y": True}]]*n)`) with a fixed symmetric range.
   - **Waveform style:** light blue `#4da3ff` at a user-set opacity, and no hover on the waveform, so hover reports the spectrogram.
   - **Panel labels:** subplot titles become small in-panel label annotations at the top left.
   - **Styling:**
     - Dark `paper_bgcolor`/`plot_bgcolor`.
     - Minimal gridlines.
     - A slim dB colorbar on the right (about 8 px).
     - An h:m:s-style time axis on the bottom row only.
     - Panel gaps of about 6 px.
   - **New default colorscale "Ember"** (black → deep blue → orange → yellow → near-white, approximating RX). Magma and Viridis remain selectable.
   - **New `difference_figure(diffs, ...)`:** the existing diff rows split out for the drawer.
   - **Export** keeps using `stacked_figure`, so the PNG/SVG matches the new look.

4. **`viewer/player/player.html`**: new CSS grid with the plot area, a right panel 300 px wide, and a bottom transport bar. All existing audio logic is unchanged: synced sources, XOR gains, seek, follow, and the keys space, Home and 1–3.
   - **Transport (bottom):**
     - A large `hh:mm:ss.mmm` readout.
     - ⏮ and ▶/⏸ buttons.
     - A segmented **Monitor** control (XOR, as now).
     - Volume.
     - Zoom −, fit and + buttons, which call `Plotly.relayout` on `xaxis.range`.
     - The file format line.
   - **Monitored panel:** its label chip in the plot turns accent orange with 🎧. Clicking a panel's label (or anywhere in a panel while holding Alt) selects it for monitoring.
   - **Right panel**, built from `DATA.stats` and `DATA.scorecard`:
     - **Cursor:** time, frequency and dB under the mouse, from `plotly_hover` on the heatmaps, plus the track name. It shows the playhead time while playing.
     - **Quality scorecard:** plain-language rows with the technical name as a tooltip, values for noisy and processed, and a ▲ green / ▼ red change badge:
       - Separation (SI-SDR, dB)
       - Quality (PESQ)
       - Intelligibility (STOI)
     - **Levels:** an RX-style table, one row per stat and one column per track:
       - Peak
       - Total RMS
       - Max RMS
       - Noise floor
       - Clipped
       - DC offset
   - **Spectrum-at-playhead chart:** only rendered if the Display toggle is on, as a small card under the panel.
   - Dark styling to match the theme.

5. **`viewer/player/__init__.py`**: `player(...)` gains these inputs:
   - `stats` (per-track dicts)
   - `scorecard` (from `metrics_df`, or `None` when roles are invalid; the panel then says "assign roles")
   - `wave_opacity`
   - `show_slice`
   - `format_line`

6. **`viewer/dsp.py`**: `signal_stats` adds two fields:
   - `max_rms_dbfs`: the maximum of the existing `rms_envelope`.
   - `noise_floor_dbfs`: the 10th percentile of `rms_envelope`. This is a plain proxy for "how quiet are the pauses", which is directly useful for judging denoising.

7. **`viewer/metrics.py`**: a new `scorecard(df)` helper picks out the three headline metrics and labels them in plain language.

8. **Tests:** update `tests/test_app.py` for the moved widgets (keys stay the same, so changes should be minimal). Add:
   - A test that `stacked_figure` produces one heatmap and one overlaid scatter per track.
   - A test for the new stats fields.
   - A Playwright check (the scratchpad script) that the right panel shows the scorecard and the cursor readout updates on hover.

## Verification (when implemented)
- `docker compose run --rm -v ./:/app viewer pytest -q` passes.
- `docker compose up --build`, then open localhost:8501 with the demo data. Check that:
  - The first screen shows only the header, the three large spectrogram panels with waveforms overlaid, the right panel and the transport. No tables or long captions are visible without opening the drawer.
  - Monitor XOR switching works mid-playback and the label highlight follows the monitored track.
  - The cursor readout updates on hover.
  - The scorecard ▲/▼ badges match the Metrics-detail table.
  - Opening the drawer shows the differences, spectrum, level, metrics and export, all still working.
  - A PNG export has the new dark look.
