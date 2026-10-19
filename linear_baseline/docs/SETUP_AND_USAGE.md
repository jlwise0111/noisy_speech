# Linear Ridge Baseline: Setup and Usage Guide

This guide covers installing, running, checking and interpreting the **linear machine-learning baseline** for the noisy-speech project: a per-frequency-bin Ridge regression on 3×3 local spectral patches, with a sweep over the regularization penalty α. The baseline sets a reference point the neural model has to beat, and its outputs load directly into the project's `diagnostic_viewer/`.

---

## Contents
1. [What the baseline does](#1-what-the-baseline-does)
2. [Repository layout](#2-repository-layout)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Dataset setup](#5-dataset-setup)
6. [Running the baseline](#6-running-the-baseline)
7. [Command-line reference](#7-command-line-reference)
8. [Output files](#8-output-files)
9. [Viewing results in the diagnostic viewer](#9-viewing-results-in-the-diagnostic-viewer)
10. [Method in detail](#10-method-in-detail)
11. [Reading the α sweep](#11-reading-the-α-sweep)
12. [Reference results](#12-reference-results)
13. [Testing](#13-testing)
14. [Known limitations](#14-known-limitations)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What the baseline does

```
Valentini train set ──► 16 kHz ──► STFT ──► log|X| ──► 3×3 patches ─┐
  (noisy + clean pairs)                                             │  streamed sufficient
                                                                    ▼  statistics per bin
                                      α sweep on held-out speakers ◄── Σx, Σxxᵀ, Σy, Σxy, Σy²
                                                │
                                                ▼  refit on train + val at the best α
                                          model.npz (257 bins × 9 weights + bias)
                                                │
Valentini test set (824) ──► predict clean log|S| ──► + noisy phase ──► iSTFT
                                                │
                                                ▼
                         outputs/<run>/{clean,noisy,enhanced}/<stem>.wav  ──►  diagnostic_viewer
```

- **One Ridge model per frequency bin.** With `n_fft=512` at 16 kHz there are 257 independent models.
- **Input features:** the 3×3 neighbourhood of the noisy log-magnitude spectrogram around (t, f), covering frames t−1, t, t+1 and bins f−1, f, f+1. That gives 9 features.
- **Target:** the clean log-magnitude at (t, f).
- **Hyperparameter:** the ridge penalty α. The script sweeps a grid of α values and picks the one with the lowest validation error on held-out speakers.
- **Read-only dataset:** the script only ever *reads* the Valentini folders. All outputs go under `linear_baseline/outputs/`, and the script refuses to start if `--out` points inside the dataset.

---

## 2. Repository layout

```
linear_baseline/
├── ridge_baseline.py        # CLI entry point: train → sweep → refit → enhance test set
├── lrb/                     # "linear ridge baseline" package
│   ├── data.py              # dataset discovery, clean/noisy pairing, speaker split, load + resample
│   ├── features.py          # STFT / iSTFT, log-magnitude, 3×3 patch extraction
│   ├── ridge.py             # sufficient statistics, closed-form ridge, α sweep, exact SSE
│   └── synth.py             # per-file statistics, enhancement, writing viewer triplets
├── tests/                   # pytest suite (synthetic data, no dataset needed)
├── docs/SETUP_AND_USAGE.md  # this file
├── requirements.txt
├── README.md                # short summary
└── outputs/                 # run folders (git-ignored)
```

---

## 3. Requirements

| Item | Version / note |
|---|---|
| Python | 3.10 or newer (developed on 3.13; the viewer image uses 3.12) |
| numpy | 1.26.4 pinned to match the viewer (2.x also works) |
| scipy | 1.14.1 pinned (1.16 tested) |
| soundfile | 0.12.1 pinned (0.13 tested); needs `libsndfile`, which the pip wheels include on macOS and Windows |
| pytest | only for the test suite |
| CPU / RAM | multi-core recommended; peak memory is small (each worker holds one file at a time) |
| Disk | about 50 MB per run folder (824 triplets × 3 wavs at 16 kHz PCM_16) |
| Docker | only needed to *view* results with `diagnostic_viewer` |

scikit-learn and PyTorch are **not** required. The Ridge solution is computed in closed form with NumPy (see §10).

---

## 4. Installation

From the repository root:

```bash
cd linear_baseline
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                            # should print "11 passed"
```

Conda works too: `conda create -n ridge python=3.12 && conda activate ridge && pip install -r requirements.txt`.

---

## 5. Dataset setup

The script expects the standard folder names of the [Valentini-Botinhao noisy speech database](https://datashare.ed.ac.uk/handle/10283/2791) under one root:

```
valentini_noisy_speech_database/
├── clean_trainset_28spk_wav/   (11,572 wavs)    noisy_trainset_28spk_wav/   (11,572)
├── clean_trainset_56spk_wav/   (23,075 wavs)    noisy_trainset_56spk_wav/   (23,075)
└── clean_testset_wav/          (824 wavs)       noisy_testset_wav/          (824)
```

- Files are 48 kHz, mono, 16-bit PCM. They are resampled to 16 kHz in memory, and nothing is written back.
- Clean and noisy files are paired by **identical filename stem** (for example `p232_001.wav`). A stem that exists in only one folder is skipped.
- macOS `._*` resource-fork files are ignored.
- The speaker id is the filename prefix before the first underscore (`p232`). Validation splits always hold out **whole speakers**, so validation measures generalisation to unseen voices.

On this project's machine the dataset lives at:
```
/Volumes/Banton Glyph RAID 2TB/USF/Classes/06 Fall 2026/Machine Learning/Group Project/audio_data_sets/valentini_noisy_speech_database
```
Quote the path in the shell because it contains spaces. A convenient pattern:
```bash
export DS="/Volumes/Banton Glyph RAID 2TB/USF/Classes/06 Fall 2026/Machine Learning/Group Project/audio_data_sets/valentini_noisy_speech_database"
```

---

## 6. Running the baseline

All commands are run from `linear_baseline/`.

### 6.1 Smoke test (seconds)
```bash
python ridge_baseline.py --dataset "$DS" --limit-files 500 --test-limit 10 --run-name smoke
```
This trains on the first 500 pairs and enhances 10 test files. Use it to check the install and dataset path.

### 6.2 Full runs
```bash
python ridge_baseline.py --dataset "$DS" --train-set 28spk     # standard Valentini training set
python ridge_baseline.py --dataset "$DS" --train-set 56spk     # larger, different speakers
python ridge_baseline.py --dataset "$DS" --train-set both      # union of 28spk + 56spk (no shared speakers)
```
Every run trains, sweeps α, refits, and enhances all 824 test files. On a 16-core Mac reading from the external RAID, runs take about 40 s (28spk) to about 7 min (56spk, first read). The time is mostly file reading; the solve itself is instant (see §12).

### 6.3 Typical terminal output
```
[    0.0s] Run folder: .../outputs/ridge_28spk_20261019_132348
[    0.1s] Train set 28spk: 10436 train / 1136 val files (val speakers: p256, p279, p287)
[    1.9s] train stats: 400/10436 files
...
  Log-magnitude MSE vs clean (val, no processing): 6.262477
  alpha        train MSE    val MSE
  1e-06       2.852854     3.002251
  ...
  0.01        2.852925     3.002139  <- best
  ...
  100         3.864679     4.055416

[   32.9s] Global α = 0.01; val MSE 3.002139
[   38.2s] Done. 824 triplets in .../outputs/ridge_28spk_20261019_132348
View: cd diagnostic_viewer && DATA_DIR=.../outputs docker compose up
```

### 6.4 Common variations
```bash
# Custom α grid
python ridge_baseline.py --dataset "$DS" --alphas 1e-4,1e-3,3e-3,1e-2,3e-2,1e-1

# Choose α per frequency bin instead of one global α
python ridge_baseline.py --dataset "$DS" --per-bin-alpha

# Fixed validation speakers (reproducible across machines and train sets)
python ridge_baseline.py --dataset "$DS" --val-speakers p226,p287

# Different STFT resolution
python ridge_baseline.py --dataset "$DS" --n-fft 1024 --hop 256

# Re-run only test-set enhancement with a saved model
python ridge_baseline.py --dataset "$DS" --skip-train \
    --model outputs/ridge_28spk_20261019_132348/model.npz --run-name ridge_28spk_rerun
```

---

## 7. Command-line reference

| Flag | Default | Description |
|---|---|---|
| `--dataset PATH` | *(required)* | Valentini root folder. Only read. |
| `--train-set {28spk,56spk,both}` | `28spk` | Which training folders to use. `both` concatenates them. |
| `--alphas a,b,c` | `logspace(-6, 2, 17)` | α grid to sweep (per-frame penalty, see §10.4). |
| `--per-bin-alpha` | off | Pick the best α separately for each frequency bin. |
| `--val-speakers p1,p2` | none | Explicit held-out speakers. Overrides `--val-frac`. |
| `--val-frac F` | `0.1` | Fraction of speakers held out (at least 1) when `--val-speakers` is not set. |
| `--seed N` | `0` | Seed for the random speaker hold-out. |
| `--sr HZ` | `16000` | Processing sample rate. All output wavs use this rate. |
| `--n-fft N` | `512` | STFT window and FFT size (`n_fft/2 + 1` bins = number of Ridge models). |
| `--hop N` | `128` | STFT hop size. |
| `--workers N` | CPU count − 1 | Parallel processes for statistics and enhancement. |
| `--limit-files N` | all | Use only the first N training pairs (sorted by name). |
| `--test-limit N` | all (824) | Enhance only the first N test pairs. |
| `--out DIR` | `linear_baseline/outputs` | Parent folder for run folders. Must not be inside the dataset. |
| `--run-name NAME` | `ridge_<set>_<timestamp>` | Run folder name. Reusing a name overwrites files in it. |
| `--model PATH` | none | `model.npz` to load (with `--skip-train`). Its sr, n_fft and hop override the CLI values. |
| `--skip-train` | off | Skip training and only enhance the test set with `--model`. |

---

## 8. Output files

```
outputs/<run_name>/
├── clean/p232_001.wav … (824)      16 kHz mono PCM_16 copy of the clean test file
├── noisy/p232_001.wav … (824)      16 kHz mono PCM_16 copy of the noisy test file
├── enhanced/p232_001.wav … (824)   Ridge output: same sample rate, channels and length as clean/noisy
├── sweep_results.csv
├── sweep_per_bin.csv
├── model.npz
└── config.json
```

| File | Contents |
|---|---|
| `sweep_results.csv` | One row per α: `alpha, train_mse, val_mse`. MSE is in the log-magnitude domain, averaged over all bins and frames. |
| `sweep_per_bin.csv` | One row per bin: `bin, freq_hz, best_alpha, best_val_mse, chosen_alpha`, then `val_mse@<α>` for every α in the grid. Useful for plotting how regularization needs vary with frequency. |
| `model.npz` | `W` (257×9 weights in original feature units), `b` (257 intercepts), `alpha` (257, the penalty used per bin), and the `sr`, `n_fft`, `hop` the model was trained with. |
| `config.json` | Every CLI argument, plus file counts, validation speakers, the full sweep curve, chosen α, chosen validation MSE, `val_mse_passthrough` (no-processing reference) and timings. |

Feature order inside `W[f]`: index `3·dt + df` corresponds to (t−1+dt, f−1+df), so index 4 is the centre cell (t, f).

Loading a model in your own code:
```python
import numpy as np
from lrb.ridge import Model
z = np.load("outputs/<run>/model.npz")
model = Model(W=z["W"], b=z["b"], alpha=z["alpha"])
```

---

## 9. Viewing results in the diagnostic viewer

The viewer finds triplets by **matching filename stems in sibling `clean/`, `noisy/` and `enhanced/` folders**. It has no manifest. It also refuses any triplet whose three files differ in sample rate, channel count or length. The baseline writes exactly that layout with matched files, so no conversion is needed.

```bash
cd diagnostic_viewer
DATA_DIR=../linear_baseline/outputs docker compose up
# open http://localhost:8501
```

- Each run shows up as a set of keys `<run_name>/<stem>`, for example `ridge_28spk/p232_001`, so several runs can be compared side by side.
- To look at a single run, point `DATA_DIR` at that run folder instead, or change the root in the viewer's sidebar.
- The viewer computes SNR, SI-SDR, PESQ (wide-band at 16 kHz) and STOI live. Its **batch export** writes a CSV with these metrics for every triplet under `diagnostic_viewer/exports/`.
- `DATA_DIR` is mounted read-only into the container. The viewer never modifies the run folders.

---

## 10. Method in detail

### 10.1 Signal processing
1. Load the wav, average any channels to mono, and resample 48 kHz → 16 kHz with `scipy.signal.resample_poly` (polyphase, anti-aliased).
2. STFT with a Hann window, `n_fft=512` (32 ms) and `hop=128` (8 ms, 75% overlap), using `scipy.signal.stft`.
3. Log-magnitude `L = log(|X| + 1e-6)`. The ε avoids `log(0)` in silent frames.

### 10.2 Features
For each time-frequency cell (t, f), the feature vector is the 3×3 block of noisy `L` centred on it. At the spectrogram edges the block uses edge padding (the nearest value is repeated). Each bin f gets its own model, so the model for bin f sees bins f−1, f and f+1 at frames t−1, t and t+1.

### 10.3 Ridge regression per bin
For bin f, with features xₜ ∈ ℝ⁹ and target yₜ (clean `L[t, f]`):

- Standardize features using the training mean μ and standard deviation σ for that bin: zₜ = (xₜ − μ)/σ.
- Solve `min_β Σₜ (yₜ − ȳ − zₜᵀβ)² + λ‖β‖²`. The intercept is not penalized.
- Convert back to original units: `W = β/σ`, `b = ȳ − Wᵀμ`.

This is exactly `StandardScaler` followed by `sklearn.linear_model.Ridge(fit_intercept=True)` for each bin.

### 10.4 α scaling: why it is a "per-frame" penalty
The script uses `λ = α · n_f`, where `n_f` is the number of training frames for bin f. A full training set has millions of frames per bin but only 9 features, so a raw λ up to about 10⁴ barely changes the solution and the sweep looks flat. Scaling by n has two effects:
- the grid `logspace(-6, 2)` spans from almost no regularization to heavy shrinkage;
- an α picked on 28spk means the same thing on 56spk or `both`.

To reproduce in sklearn: `Ridge(alpha=α * n_frames)` on standardized features.

### 10.5 One-pass training with sufficient statistics
Ridge needs only second-order statistics. For every bin, the script accumulates, over all frames of all files:

```
n,  Σx (9),  Σxxᵀ (9×9),  Σy,  Σxy (9),  Σy²
```

From these:
- the centred Gram matrix and cross-moment come out directly, so a fit for **any** α is a 9×9 solve, and all 257 bins are solved together;
- the **exact** squared error of any model on any data set is
  `SSE = Σy² − 2WᵀΣxy − 2bΣy + WᵀΣxxᵀW + 2bWᵀΣx + n·b²`,
  so validation MSE for the whole α grid comes from the validation statistics, with no second pass over the audio;
- statistics from different files simply **add up**, so files are processed in parallel (`multiprocessing.Pool`) and the partial sums are combined. Training plus validation statistics give the final refit at no extra cost.

### 10.6 Model selection
1. Split the training pairs by speaker: about 10% of speakers go to validation (or the ones given with `--val-speakers`).
2. Accumulate separate train and validation statistics.
3. For each α, fit on train and compute train and validation MSE.
4. Pick α:
   - **global** (default): the α with the lowest validation MSE summed over all bins;
   - **per-bin** (`--per-bin-alpha`): for each bin, the α with the lowest validation MSE for that bin.
5. Refit on train + validation statistics at the chosen α and save `model.npz`.

### 10.7 Enhancement
1. Compute the noisy STFT, features and predicted log-magnitude `L̂`.
2. Rebuild the magnitude as `max(exp(L̂) − ε, 0)` and combine it with the **noisy phase**.
3. Run the inverse STFT, trim or pad to the exact sample count of the 16 kHz clean signal, clip to [−1, 1], and write PCM_16.

---

## 11. Reading the α sweep

- **Train MSE** rises steadily with α, because more shrinkage gives a worse fit to the training data.
- **Validation MSE** should be **U-shaped** or flat-then-rising. Its minimum is the chosen α.
- Compare against **"no processing"**: the MSE you get by using the noisy log-magnitude as the estimate (printed above the table and stored as `val_mse_passthrough`). Ridge should be well below it.
- A completely flat curve means the grid is too narrow. Widen `--alphas`.
- With only 9 features per model and millions of frames, the model is strongly over-determined. Expect a shallow minimum: regularization helps only slightly, and too much α hurts a lot.

---

## 12. Reference results

Measured on 2026-10-19: 16-core macOS, Python 3.13, dataset on an external RAID, default settings (`n_fft=512`, `hop=128`, `logspace(-6, 2, 17)` α grid, global α, ~10% held-out speakers with `--seed 0`).

### 12.1 Training and α selection

| Train set | Train / val files | Val speakers | Best α | Val log-mag MSE: no processing → Ridge | Run time |
|---|---|---|---|---|---|
| `28spk` | 10,436 / 1,136 | p256, p279, p287 | 0.01 | 6.262 → **3.002** | 38 s |
| `56spk` | 20,696 / 2,379 | p241, p272, p295, p299, p343, p363 | 0.01 | 5.916 → **2.686** | 400 s* |
| `both` | 30,951 / 3,696 | p233, p270, p275, p287, p293, p298, p307, p308, p313 | 0.0316 | 5.967 → **2.667** | 176 s* |

\*Mostly cold reads of the 56spk files from the external drive. The `both` run reused the OS file cache.

Every sweep is shallow and U-shaped. Validation MSE is flat for α ≤ 10⁻³, reaches its minimum at α ≈ 0.01–0.03, and rises steeply above α ≈ 1. More training data lowers validation error, and the optimal α barely moves, which is what the per-frame α scaling (§10.4) is meant to achieve.

### 12.2 Test set (824 files, speakers p232 and p257)

Checked with the diagnostic viewer's own `discover()` and `validate_strict()`. Every run gives **824/824 resolved triplets, 0 rejected, all 16 kHz mono**.

| Train set | Mean SI-SDR noisy → enhanced (dB) | Mean SNR noisy → enhanced (dB) |
|---|---|---|
| `28spk` | 8.45 → 4.96 | 8.45 → 1.61 |
| `56spk` | 8.45 → 4.64 | 8.45 → 1.50 |
| `both` | 8.45 → 4.74 | 8.45 → 1.51 |

The log-magnitude error roughly halves, but these waveform metrics go *down*. That comes from the log-domain resynthesis, not a training bug; see §14. Use the viewer's batch export for PESQ and STOI.

### 12.3 Other checks performed
- `--skip-train --model …/model.npz` reproduces the full run's enhanced wavs byte-for-byte.
- `--per-bin-alpha` picks 10 distinct α values across the 257 bins (median 0.0316).
- Invalid `--val-speakers` and `--out` inside the dataset both exit with a clear message, without creating a run folder.
- After all runs, `find <dataset> -newermt 2026-10-19` returns 0 files. The dataset (70,943 files) was not modified.

---

## 13. Testing

```bash
cd linear_baseline
pytest -q
```

The suite uses synthetic data and needs no dataset:

| Test file | What it checks |
|---|---|
| `test_features.py` | Patch shape `(T, F, 9)`; the centre feature equals the input; neighbour order and edge padding |
| `test_ridge.py` | Recovers known weights with α→0; matches an explicit standardized-ridge solve; SSE from statistics equals the direct SSE; statistics are additive; larger α shrinks ‖W‖ |
| `test_synth.py` | An identity model (centre weight 1) reconstructs the input to within 1e-4; output length is exact for odd lengths; written triplets share sample rate, channels and length; statistics work from 48 kHz wavs |

---

## 14. Known limitations

- **Waveform metrics can drop even though log-magnitude error halves.** MSE regression in the log domain predicts the *geometric* mean of the clean magnitude. That systematically under-estimates energy, and the estimate is not limited to the noisy magnitude. On the test set this lowers SI-SDR relative to the noisy input (§12). An offline check showed the same trained model gives:
  - noisy-level SI-SDR when a log-normal bias correction (`exp(L̂ + σ²/2)`) is applied;
  - **about +2.8 dB above noisy** when the prediction is applied as a gain capped at 1 on the noisy STFT.

  Those resynthesis variants are not implemented in the CLI yet. The current output is the pure log-magnitude mapping that was specified.
- **Linear and local only:** each bin sees 9 values from a 3×3 neighbourhood. There is no long-range context, no non-linearity, and no phase estimation. This is intentional for a baseline.
- **Noisy phase** is reused, which caps the achievable quality at low SNR.

---

## 15. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: Missing …/clean_trainset_28spk_wav` | `--dataset` must point at the Valentini **root**, the folder that contains the six `*_wav` folders. Quote paths that contain spaces. |
| `Empty split: N train / 0 val files` | With `--limit-files`, the sorted subset can cover only one or two speakers. Raise `--limit-files` or pass `--val-speakers` with a speaker that is in the subset. |
| `Validation speakers not in training set` | The speaker ids passed to `--val-speakers` do not occur in the chosen `--train-set` (28spk and 56spk have different speakers). |
| `--out must not be inside the dataset folder` | The safety check that keeps the dataset read-only. Choose another `--out`. |
| Flat α sweep | Widen the grid, e.g. `--alphas 1e-8,1e-6,1e-4,1e-2,1,100,1e4`. |
| Viewer shows ⚠ "missing processed" | Wrong `DATA_DIR`. Point it at `linear_baseline/outputs` or at a run folder, not the Valentini root. |
| Viewer rejects a triplet over sample rate or length | Files were mixed from different runs or edited by hand. Re-run the baseline into a fresh `--run-name`. |
| Slow on an external drive | Statistics collection is limited by reading files. Lower `--workers` if the drive is thrashing, or copy the dataset to a local SSD. |
| `RuntimeError` about `spawn` / `freeze_support` on Windows | Run the script as a file (`python ridge_baseline.py …`), not from an interactive session. |
