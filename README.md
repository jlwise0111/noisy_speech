# noisy_speech

A speech-denoising project on the Valentini noisy-speech dataset: baselines, an evaluation suite and
a diagnostic viewer. The U-Net writes the same output layout, so the suite and viewer pick it up as "Neural network".

| Folder | What it is |
|---|---|
| `wiener_baseline/` | Classical Wiener-filter baseline (leading / minimum-statistics noise estimate, plus an oracle upper bound) |
| `linear_baseline/` | Linear ML baseline: per-frequency-bin Ridge regression on 3×3 spectral patches, with an α sweep |
| `speechprep/` | Shared 16 kHz preprocessing (resampling, STFT 512/128, speaker split, waveform cache), identical to the Ridge front end |
| `unet_model/` | 2D U-Net ratio-mask model with a silence-weighted loss (PyTorch) |
| `metric_suite/` | Scores every model run: PESQ-WB, STOI and SNR gain vs the noisy input |
| `diagnostic_viewer/` | Web app (Docker) to watch and listen to clean / noisy / processed triplets per system, loudness-matched |

Every model writes `<folder>/outputs/<run>/{clean,noisy,enhanced|processed}/<stem>.wav`. The metric suite and
the viewer find those runs automatically.

## Make targets

Run `make` for the full list. The main ones:

```bash
make pipeline         # train every model at the same time → score them all → open the viewer
make all              # checks → all tests → build viewer → collect data → evaluate → start viewer
make test             # the five test suites (make test-linear / test-wiener / test-metrics / test-unet / test-viewer)
make train-evaluate   # run every model (finished runs are skipped), then score them all
make data-unet        # train the U-Net on 28spk and enhance the test set (hours; not part of 'all')
make viewer           # build, start and open the viewer;  make viewer-close to shut it down
```

Tell make where the dataset is with `make … DATASET=/path/to/valentini_noisy_speech_database`, by setting
`VALENTINI_DIR`, or once per machine in an untracked `local.mk` containing
`DATASET := /path/to/valentini_noisy_speech_database`.

## Running everything at once

`make pipeline` runs `run_pipeline.py`, which does the scheduling in Python:

1. Every needed 48 kHz file is resampled to 16 kHz once into `.cache/speechprep`, in parallel. Every model
   then reads that cache instead of re-reading the dataset.
2. The U-Net, Ridge (28spk / 56spk / both) and Wiener (leading / minstats / oracle) runs train side by side.
   Each gets a share of the cores Python detects (`PIPELINE_ARGS="--cores N"` to override) as its worker /
   thread count, and BLAS / OpenMP pools are pinned to one thread per process, so the jobs never ask for
   more cores than there are. No GPU is needed; the U-Net uses one only if PyTorch finds it.
3. The metric suite scores all runs in one shared process pool (`metric_suite/results/summary.md`).
4. The diagnostic viewer starts and opens in the browser.

Finished runs are skipped and an interrupted U-Net run resumes, so re-running is safe; `FORCE=1` redoes
everything. Progress is printed as jobs start and finish; each job's full output is in `logs/<run>.log`.
`make pipeline-smoke` runs the same thing on a few files in a few minutes.
