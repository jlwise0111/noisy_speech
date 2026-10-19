# Linear ML Baseline: per-bin Ridge Regression

This is a linear baseline for the speech-denoising project. It is a **per-frequency-bin Ridge regression** on **3×3 local spectral patches** of the noisy log-magnitude spectrogram: frames t−1, t, t+1 × bins f−1, f, f+1. It includes a **sweep over the regularization penalty α**.

It trains on the Valentini training set, picks α on held-out speakers, enhances the Valentini test set, and writes clean/noisy/enhanced triplets that `../diagnostic_viewer/` loads as-is. The dataset is only ever read.

Full setup, usage, method and reference results: [docs/SETUP_AND_USAGE.md](docs/SETUP_AND_USAGE.md).

## Method
| Step | Detail |
|---|---|
| Audio | 48 kHz Valentini → 16 kHz mono (`resample_poly`) |
| STFT | Hann, `n_fft=512`, `hop=128` → 257 bins |
| Features | `log(|X| + 1e-6)` of noisy, 3×3 patch (9 values, edge-padded) per (t, f) |
| Target | clean `log(|S| + 1e-6)` at (t, f) |
| Model | 257 independent Ridge models (9 weights + intercept each), standardized features, unpenalized intercept |
| Fit | closed form from streamed sufficient statistics (Σx, Σxxᵀ, Σy, Σxy, Σy²) per bin: one pass over the audio for the whole sweep |
| α sweep | default `logspace(-6, 2, 17)`; global α (or `--per-bin-alpha`) chosen by exact validation MSE on held-out speakers, then refit on train + val |
| Resynthesis | `exp(L̂)` with the noisy phase, iSTFT, length matched to the clean file |

**α is a per-frame penalty.** The fit equals `StandardScaler` + `sklearn.linear_model.Ridge(alpha=α·n)` per bin, where n is the number of training frames (about 2–8 million). Scaling by n gives the same α grid the same meaning for `28spk`, `56spk` and `both`. Without it, every α up to 10⁴ is negligible and the sweep is flat.

## Usage
```bash
cd linear_baseline
pip install -r requirements.txt
pytest -q

DS="/path/to/valentini_noisy_speech_database"
python ridge_baseline.py --dataset "$DS" --train-set 28spk          # ~40 s on 16 cores (56spk: several minutes, disk-bound)
python ridge_baseline.py --dataset "$DS" --train-set 56spk
python ridge_baseline.py --dataset "$DS" --train-set both           # union of both training folders
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--alphas 1e-4,1e-3,0.01` | custom sweep grid |
| `--per-bin-alpha` | choose α per frequency bin instead of one global α |
| `--val-speakers p226,p287` | fixed validation speakers (default: about 10% of speakers, `--seed`) |
| `--limit-files N` / `--test-limit N` | quick smoke runs |
| `--model outputs/<run>/model.npz --skip-train` | rerun test enhancement only |
| `--n-fft`, `--hop`, `--sr`, `--workers`, `--out`, `--run-name` | |

## Output
```
outputs/<run_name>/                 # default: ridge_<train-set>_<YYYYmmdd_HHMMSS>
  clean/<stem>.wav                  # 16 kHz copy of the clean test file
  noisy/<stem>.wav                  # 16 kHz copy of the noisy test file
  enhanced/<stem>.wav               # Ridge output (same sr, channels, length: passes the viewer's strict check)
  sweep_results.csv                 # alpha, train_mse, val_mse (log-magnitude domain, all bins)
  sweep_per_bin.csv                 # per-bin val MSE for every α, best α per bin
  model.npz                         # W (257×9), b, alpha per bin, sr, n_fft, hop
  config.json                       # args, split speakers, chosen α, pass-through MSE, timings
```

## Viewing in the diagnostic viewer
```bash
cd ../diagnostic_viewer
DATA_DIR=../linear_baseline/outputs docker compose up
```
Each run folder shows up as `<run_name>/<stem>` triplets. The viewer's batch export computes SNR, SI-SDR, PESQ and STOI for a whole run.
