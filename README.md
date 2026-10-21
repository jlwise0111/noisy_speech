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
make all              # checks → all tests → build viewer → collect data → evaluate → start viewer
make test             # the five test suites (make test-linear / test-wiener / test-metrics / test-unet / test-viewer)
make train-evaluate   # run every model (finished runs are skipped), then score them all
make data-unet        # train the U-Net on 28spk and enhance the test set (hours; not part of 'all')
make viewer           # build, start and open the viewer;  make viewer-close to shut it down
```

The dataset path defaults to the project machine's location. Override it with
`make … DATASET=/path/to/valentini_noisy_speech_database`, or set `VALENTINI_DIR`.
