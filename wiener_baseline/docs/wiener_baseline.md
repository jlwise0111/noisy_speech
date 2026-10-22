# Wiener Filter Baseline: Setup and Usage

This guide covers the classical-DSP speech-enhancement baseline in `wiener_baseline/`:
- what it does and why it exists,
- how to install and run it,
- what it writes,
- how to view the results in `diagnostic_viewer`,
- how to plug the team's neural network into the same comparison.

---

## 1. Purpose

The project's goal is a neural network that isolates speech from background noise. To show
whether the network is any good, we need a **reference point that uses no learning at all**.
The Wiener filter is the textbook choice:

> A Wiener filter is a linear time-invariant (LTI) filter that estimates a target signal from a
> noisy or degraded observation. It works by minimizing the mean square error (MSE) between the
> estimated signal and the desired signal.

The baseline reads noisy/clean pairs from the Valentini (VoiceBank-DEMAND) dataset, enhances the
noisy file with a Wiener filter, and writes **clean / noisy / processed triplets**. The diagnostic
viewer can load these directly, side by side, with spectrograms and SI-SDR / PESQ / STOI scores.
The neural network's output is later dropped into the same `processed/` slot, so all three
(Wiener, NN and oracle) are compared the same way.

---

## 2. Repository layout

```
noisy_speech/
├── wiener_baseline/              ← branch: feature/wiener-baseline
│   ├── README.md                 short summary + commands
│   ├── docs/
│   │   └── wiener_baseline.md    ← this file (full setup & usage guide)
│   ├── requirements.txt          numpy, scipy, soundfile, pytest
│   ├── run_baseline.py           command-line runner (dataset → triplets)
│   ├── wiener/
│   │   ├── __init__.py           public API
│   │   ├── wiener.py             the DSP: STFT, noise estimate, Wiener gain, ISTFT
│   │   └── audio_io.py           dataset pairing + WAV read/write
│   ├── tests/
│   │   └── test_wiener.py        synthetic-signal unit tests
│   └── outputs/                  generated triplets (gitignored, not committed)
└── diagnostic_viewer/            ← branch: feature/diagnostic-viewer (not modified here)
```

The baseline and the viewer live on **separate branches**:

| Branch                       | Contains                                   |
|------------------------------|--------------------------------------------|
| `main`                       | initial commit only                        |
| `feature/diagnostic-viewer`  | the Streamlit viewer                       |
| `feature/wiener-baseline`    | this baseline (branched from `main`)       |

`wiener_baseline/outputs/` is gitignored, so generated audio **stays on disk when you switch
branches**. You can run the baseline on one branch, then switch to the viewer branch and open
the output.

> Switching to `feature/wiener-baseline` may show `diagnostic_viewer/` as untracked. It holds the
> viewer's local, gitignored `data/` folder, left behind by the branch switch. This is harmless:
> don't commit it.

---

## 3. The dataset

The baseline expects the Valentini noisy speech database, unpacked into one folder:

```
valentini_noisy_speech_database/
├── clean_testset_wav/            824 files     (speakers p232, p257)
├── noisy_testset_wav/            824 files
├── clean_trainset_28spk_wav/   11 572 files    (28 speakers)
├── noisy_trainset_28spk_wav/   11 572 files
├── clean_trainset_56spk_wav/   23 075 files    (56 speakers)
└── noisy_trainset_56spk_wav/   23 075 files
```

Keep it anywhere outside the repository and pass its path with `--dataset`, or set `VALENTINI_DIR`
(section 4.4). The examples use `/path/to/valentini_noisy_speech_database` as a placeholder.

All files are **48 kHz, mono, 16-bit PCM WAV**. The baseline resamples them to **16 kHz** on load and the filter
runs at 16 kHz only, the same rate every other model in the project uses. Noisy and clean files with the same name (e.g.
`p232_001.wav`) are the same utterance and the same length in samples.

The baseline **only reads** from this folder. `run_baseline.py` refuses to run if `--out` points
inside the dataset folder.

The `--split` option picks the folder pair:

| `--split` | noisy folder                | clean folder                |
|-----------|-----------------------------|-----------------------------|
| `test`    | `noisy_testset_wav`         | `clean_testset_wav`         |
| `train28` | `noisy_trainset_28spk_wav`  | `clean_trainset_28spk_wav`  |
| `train56` | `noisy_trainset_56spk_wav`  | `clean_trainset_56spk_wav`  |

Report final numbers on the **test** split. That is the standard benchmark split, and it is the one
the neural network never trains on. The Wiener filter has nothing to train, so running it on the
train splits only checks it at scale and gives a reference on the training data.

---

## 4. Setup

### 4.1 Get the code

```bash
cd noisy_speech          # wherever you cloned the repository
git pull
```

### 4.2 Python environment

You need Python 3.10 or newer (developed on 3.13). The only runtime dependencies are `numpy`, `scipy`
and `soundfile`; `pytest` is needed for the tests.

**Option A: existing conda/miniconda Python** (these packages are often already installed):

```bash
python3 -c "import numpy, scipy, soundfile; print('ok')"
```

**Option B: a dedicated virtual environment:**

```bash
cd wiener_baseline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`soundfile` bundles `libsndfile` in its macOS, Windows and Linux wheels, so you don't need to
install it separately.

### 4.3 Check the install

```bash
cd wiener_baseline
python -m pytest tests -q          # expect: 9 passed
```

### 4.4 (Optional) Set the dataset path once

`run_baseline.py` reads the `VALENTINI_DIR` environment variable when `--dataset` is not given:

```bash
export VALENTINI_DIR="/path/to/valentini_noisy_speech_database"
```

The examples below assume this is set. If not, add `--dataset "$VALENTINI_DIR"` (or the path).

---

## 5. Running the baseline

Run all commands from inside `wiener_baseline/`.

### 5.1 Quick smoke test (10 files, a few seconds)

```bash
python run_baseline.py --split test --limit 10 --run-name smoke
```

### 5.2 The standard runs

```bash
python run_baseline.py --split test                        # → outputs/wiener_leading_test
python run_baseline.py --split test --noise-est minstats   # → outputs/wiener_minstats_test
python run_baseline.py --split test --method oracle        # → outputs/oracle_test   (upper bound)
```

### 5.3 Train splits

```bash
python run_baseline.py --split train28                     # → outputs/wiener_leading_train28
python run_baseline.py --split train56                     # → outputs/wiener_leading_train56
```

### 5.4 Example output

```
824 file(s) from 'test' -> .../wiener_baseline/outputs/wiener_leading_test
  50/824  (2s)
  ...
  824/824  (8s)

mean over 824 files      noisy   processed   delta
  SNR      (dB)            8.39       15.91   +7.52
  SI_SDR   (dB)            8.39       15.76   +7.36

PESQ/STOI: make evaluate;  listen: make viewer
```

The SNR and SI-SDR summary is a quick check computed with numpy. For the full quality and
intelligibility scores (PESQ, STOI), run `make evaluate` (the metric suite).

### 5.5 Command-line reference

| Option              | Default                       | Meaning |
|---------------------|-------------------------------|---------|
| `--dataset PATH`    | `$VALENTINI_DIR`              | Root of the Valentini dataset |
| `--split`           | `test`                        | `test`, `train28` or `train56` |
| `--method`          | `wiener`                      | `wiener` (the baseline) or `oracle` (upper bound, uses clean audio) |
| `--noise-est`       | `leading`                     | Noise estimator for `wiener`: `leading` or `minstats` |
| `--limit N`         | all                           | Process only the first N files (sorted by name) |
| `--out DIR`         | `wiener_baseline/outputs`     | Where run folders are created |
| `--run-name NAME`   | `<method>_<noise-est>_<split>`| Output subfolder name (`oracle_<split>` for oracle) |
| `--jobs N`          | number of CPU cores           | Parallel worker processes |
| `--frame-ms`        | `32`                          | STFT window length (ms) |
| `--overlap`         | `0.75`                        | STFT overlap fraction |
| `--alpha`           | `0.98`                        | Decision-directed smoothing |
| `--gain-floor-db`   | `-20`                         | Minimum gain (dB) |
| `--xi-min-db`       | `-25`                         | Minimum a priori SNR (dB) |
| `--noise-init-s`    | `0.25`                        | `leading`: seconds at the start assumed to be noise only |
| `--ms-window-s`     | `1.5`                         | `minstats`: minimum-search window (s) |
| `--ms-smooth`       | `0.85`                        | `minstats`: periodogram smoothing factor |
| `--ms-bias`         | `2.0`                         | `minstats`: bias-compensation factor |

Re-running with the same run name overwrites the files in that folder.

---

## 6. How the algorithm works

All processing happens in the short-time Fourier transform (STFT) domain, one frequency bin at
a time. `Y(t, f)` is the noisy spectrum, `S` the clean speech and `N` the noise.

```
noisy wav ─► STFT Y ─► noise PSD N̂(f) ─► a-priori SNR ξ ─► gain G = ξ/(1+ξ) ─► Ŝ = G·Y ─► ISTFT ─► processed wav
```

1. **STFT.** 16 kHz input only. Hann window, 32 ms frames (512 samples), 75 % overlap (8 ms hop, 128 samples).

2. **Noise power estimate `N̂`.**
   - **`leading`** (default) averages `|Y|²` over the first `noise_init_s` = 0.25 s. Valentini
     clips almost always start with a short pause, so this is a clean noise sample. The estimate
     is fixed for the whole clip.
   - **`minstats`** is a simplified version of Martin's *minimum statistics*:
     1. Smooth `|Y|²` over time (`ms_smooth`).
     2. Take the minimum over a sliding 1.5 s window. Speech pauses pull this minimum down to the noise floor.
     3. Multiply by `ms_bias` to correct for the minimum lying below the mean.

     This follows noise that changes within a clip. It does not assume a quiet start.

3. **A priori SNR, decision-directed (Ephraim & Malah, 1984).**
   `ξ_t = α · |Ŝ_{t−1}|² / N̂ + (1 − α) · max(|Y_t|²/N̂ − 1, 0)`, then floored at `xi_min_db`.
   Mixing in the previous frame's estimate (α = 0.98) smooths ξ over time. That greatly reduces
   the "musical noise" (random twinkling tones) of naive spectral subtraction.

4. **Wiener gain.** `G = ξ / (1 + ξ)`. This is the gain that minimises
   `E[|S − G·Y|²]` in each bin when speech and noise are uncorrelated, which is the MSE
   criterion in the definition above. `G` is then floored at `gain_floor_db` (−20 dB ≈ 0.1).
   Removing a bin completely sounds unnatural, so a little residual noise is kept on purpose.

5. **Reconstruction.** `Ŝ = G·Y`, inverse STFT (overlap-add), trim or zero-pad to **exactly**
   the input length, clip to [−1, 1], and write as 16-bit PCM.

### Oracle mode (`--method oracle`)

Oracle mode builds the *ideal* Wiener mask from the true signals:
`G = |S|² / (|S|² + |N|²)`, where `N = noisy − clean`.

It needs the clean file, so it is **not a usable enhancer**. It shows the best result that any
time-frequency Wiener mask could reach on this data. That gives a useful bracket:

```
noisy  ≤  Wiener baseline  ≤  (your neural net?)  ≤  oracle Wiener
```

A neural network that predicts a mask is, in effect, trying to approximate this oracle.

### Tuning notes

| Change                         | Effect |
|--------------------------------|--------|
| Lower `--gain-floor-db` (e.g. −30) | More noise removed, more musical noise and speech damage |
| Higher `--gain-floor-db` (e.g. −12)| Gentler, more residual noise, fewer artefacts |
| Lower `--alpha` (e.g. 0.9)     | Responds faster to speech onsets, more musical noise |
| Longer `--frame-ms`            | Better frequency resolution, more smearing in time |
| `--noise-est minstats`         | Handles noise that changes over time, but untuned it underperforms `leading` on Valentini (see §9) |

Use the **test** split and the viewer's batch PESQ/STOI to judge tuning changes. SI-SDR alone can
favour aggressive settings that sound worse.

---

## 7. Output format

Each run creates one folder:

```
outputs/wiener_leading_test/
├── clean/p232_001.wav        dataset file resampled to 16 kHz
├── noisy/p232_001.wav        dataset file resampled to 16 kHz
├── processed/p232_001.wav    Wiener output
├── ...
├── params.json               method, split, dataset folder name, sample rate, file count, git commit, all parameters
└── quick_metrics.csv         per file: snr_noisy, snr_processed, si_sdr_noisy, si_sdr_processed
```

Guarantees for every `processed/` file, which the viewer's strict check needs:
- the same file name as its clean and noisy partners,
- the same sample rate (16 kHz), the same channel count (mono) and **the same length in samples**,
- values in [−1, 1], 16-bit PCM.

The clean and noisy files are **written** (resampled), not symlinked. The viewer's Docker container only sees the
mounted folder, and can't follow symlinks that point outside it. The cost is disk space:

| Split   | Files  | Approx. size per run folder |
|---------|-------:|----------------------------:|
| test    |    824 |                     ~0.2 GB |
| train28 | 11 572 |                       ~3 GB |
| train56 | 23 075 |                     ~6.5 GB |

Delete run folders you no longer need. They can always be regenerated.

---

## 8. Viewing results in the diagnostic viewer

The viewer finds triplets by folder name. A folder that contains `clean/`, `noisy/` and
`processed/` subfolders with matching file names is picked up automatically. Every run folder
above follows that layout.

### 8.1 With Docker (the viewer's standard way)

```bash
git switch feature/diagnostic-viewer        # outputs/ stays on disk
cd diagnostic_viewer
cp .env.example .env                        # first time only
```

Edit `diagnostic_viewer/.env` so `DATA_DIR` points at the run folder, or at `outputs/` to see every
run:

```
DATA_DIR=../wiener_baseline/outputs
```

An absolute path also works. Then run:

```bash
docker compose up --build
```

Open <http://localhost:8501>.

### 8.2 Without Docker

From `diagnostic_viewer/` with its requirements installed (`pip install -r requirements.txt`):

```bash
VIEWER_DATA_DIR=../wiener_baseline/outputs streamlit run app.py
```

To switch folders, change `DATA_DIR` / `VIEWER_DATA_DIR` and restart; to pick up new outputs, reload the page.

### 8.3 What you'll see

- If `DATA_DIR` is a single run folder, triplets are listed by name (`p232_001`, ...).
- If `DATA_DIR` is `outputs/`, triplets are prefixed with their run (`wiener_leading_test/p232_001`,
  `oracle_test/p232_001`, ...), so you can flip between methods for the same utterance.
- The viewer scores the file on screen (SNR, SI-SDR, PESQ, STOI). To score every file of every run,
  use `make evaluate` (the metric suite).

Things to look at:
- **Spectrograms:** energy left in the pauses of `processed` is residual noise; missing energy
  compared with `clean` is speech that was removed (often high-frequency consonants).
- **Loudness over time / noise floor:** how far the pauses dropped compared with noisy.
- **Listening:** musical noise (warbling tones in the pauses) is the classic Wiener artefact.

---

## 9. Verified results

All numbers below are from runs on the full Valentini dataset (all three splits) at 16 kHz with default
parameters. SNR and SI-SDR are means over all files, from `quick_metrics.csv`.

| Split   | Method                  | Files  | Noisy SI-SDR | Processed SI-SDR | Δ SI-SDR | Noisy SNR | Processed SNR | Run time |
|---------|-------------------------|-------:|-------------:|-----------------:|---------:|----------:|--------------:|---------:|
| test    | Wiener, `leading`       |    824 |         8.45 |            15.86 |    +7.41 |      8.45 |         16.01 |      ~3 s |
| test    | Wiener, `minstats`      |    824 |         8.45 |            13.12 |    +4.67 |      8.45 |         13.32 |      ~3 s |
| test    | Oracle Wiener           |    824 |         8.45 |            21.04 |   +12.59 |      8.45 |         21.08 |      ~3 s |
| train28 | Wiener, `leading`       | 11 572 |         6.12 |            11.26 |    +5.14 |      6.13 |         11.72 |   ~70 s |
| train56 | Wiener, `leading`       | 23 075 |         6.04 |            11.14 |    +5.10 |      6.04 |         11.61 |  ~2.5 min |

Run times are with a warm `--cache-dir` and are only a rough guide: they depend on the machine. `--jobs`
(default: every available core) sets how many files are processed at once, and `make pipeline` gives the
baseline its share of the cores automatically. The numbers differ slightly from the earlier 48 kHz runs
(e.g. test `leading` +7.36 dB then, +7.41 dB now) because the metrics are now computed on 16 kHz audio.

PESQ-WB and STOI on the test set, from `make evaluate`: noisy 1.97 / 0.921, `leading` 2.42 / 0.912,
`minstats` 2.31 / 0.908, oracle 3.84 / 0.982. The blind filters raise PESQ but lower STOI slightly:
removing noise also removes some of the speech detail that intelligibility depends on.

**Viewer compatibility.** The viewer's own discovery code (`viewer/dataset.py: discover`) was run
against every output folder, comparing sample rate (all 16 kHz) and length for each triplet:

| Run                      | Triplets found | Resolved | SR/length mismatches | NaN/Inf metrics |
|--------------------------|---------------:|---------:|---------------------:|----------------:|
| `wiener_leading_test`    |            824 |      824 |                    0 |               0 |
| `wiener_minstats_test`   |            824 |      824 |                    0 |               0 |
| `oracle_test`            |            824 |      824 |                    0 |               0 |
| `wiener_leading_train28` |         11 572 |   11 572 |                    0 |               0 |
| `wiener_leading_train56` |         23 075 |   23 075 |                    0 |               0 |

**Why the train splits score lower than test.** The train sets were mixed at harder SNRs
(0–15 dB) than the test set (2.5–17.5 dB): the mean input SNR is about 6 dB, against 8.5 dB for test.
Even so, the filter gives a solid +5 dB SI-SDR on both train sets, and the 28- and 56-speaker
results are almost the same. That is expected, because the filter has no training data to depend on.

**Files the filter makes worse (known limitation).** In these files SI-SDR goes down:

| Split   | Files worse than noisy | Median input SI-SDR of those files (vs. whole split) |
|---------|-----------------------:|-----------------------------------------------------:|
| test    |       7 / 824 (0.8 %)  | 16.5 dB (vs. 7.4 dB)                                 |
| train28 |  355 / 11 572 (3.1 %)  | 13.4 dB (vs. 6.6 dB)                                 |
| train56 |  949 / 23 075 (4.1 %)  | 13.6 dB (vs. 6.5 dB)                                 |

They are mostly **already-clean inputs**: there is little noise to remove, so any speech the
filter removes costs more than the noise it removes. The rest are usually clips where speech starts
within the first 0.25 s, which makes the `leading` noise estimate include speech, or clips whose noise
changes partway through. The average gain falls as the input gets cleaner (test split):

| Input SI-SDR | Files | Mean Δ SI-SDR |
|--------------|------:|--------------:|
| < 5 dB       |   218 |         +9.59 |
| 5–12 dB      |   400 |         +7.67 |
| > 12 dB      |   206 |         +4.58 |

These are exactly the cases a learned model should handle better. Sort `quick_metrics.csv` by
`si_sdr_processed − si_sdr_noisy` to find them and open them in the viewer.

---

## 10. Adding your neural network's output

To compare the neural network in the same viewer:

1. Run the model on the noisy files (normally `noisy_testset_wav`).
2. Write each output as `outputs/<your-run>/processed/<same stem>.wav`. It must have the same sample
   rate (16 kHz), mono channel and **exact** length as the noisy input. The viewer will not resample for you.
3. Put copies of the matching `clean/` and `noisy/` files next to it. The simplest way is to copy
   them from an existing Wiener run folder.
4. Point the viewer at `outputs/` to compare `wiener_leading_test`, `oracle_test` and your run
   side by side.

Only use **one** processed-type folder per run (`processed/`, `enhanced/`, `denoised/` or `output/`).
Two of them in the same parent make the viewer flag the triplet as ambiguous (⚠).

---

## 11. Using the filter from Python

```python
import soundfile as sf
from wiener import wiener_filter, wiener_oracle, WienerParams   # run from wiener_baseline/
from wiener.audio_io import read_wav                             # mono float32, resampled to 16 kHz

noisy, sr = read_wav("noisy.wav")                                    # sr is always 16000
enhanced = wiener_filter(noisy, sr)                                  # default: leading noise estimate
gentler  = wiener_filter(noisy, sr, "leading", WienerParams(gain_floor_db=-12))
tracked  = wiener_filter(noisy, sr, noise_method="minstats")

clean, _ = read_wav("clean.wav")
ceiling  = wiener_oracle(noisy, clean, sr)

sf.write("enhanced.wav", enhanced, sr, subtype="PCM_16")
```

Lower-level pieces in `wiener/wiener.py` (`analyze`, `synthesize`, `noise_psd_leading`,
`noise_psd_minstats`, `wiener_gain`) can be reused. For example, to plot the gain mask:

```python
from wiener.wiener import analyze, noise_psd_leading, wiener_gain
p = WienerParams()
Y = analyze(noisy, sr, p)
G = wiener_gain(Y, noise_psd_leading(abs(Y) ** 2, sr, p), p)   # shape (freq, frames), values in [0.1, 1]
```

---

## 12. Tests

```bash
cd wiener_baseline
python -m pytest tests -q
```

The tests use synthetic harmonic "speech" bursts in white noise and check that:
- the output length equals the input length, including odd lengths,
- the gain stays within [floor, 1],
- SNR improves by more than 3 dB with both noise estimators,
- the oracle beats the blind filter,
- silent input gives finite (zero) output,
- the output stays within [−1, 1] even for loud input,
- the filter refuses audio at any rate other than 16 kHz,
- 48 kHz input gives a 16 kHz clean / noisy / processed triplet of matching length.

---

## 13. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `--dataset is required` | Pass `--dataset PATH` or `export VALENTINI_DIR=...` |
| `missing dataset folder: ...noisy_trainset_28spk_wav` | Wrong dataset root, or that split wasn't downloaded or unzipped |
| `warning: N unmatched file(s) skipped` | A noisy file has no clean partner (or the reverse). Only matched pairs are processed |
| `noisy/clean mismatch` error | A pair has a different sample rate or length. The Valentini originals never do, so check the files aren't corrupted |
| `the Wiener filter runs at 16000 Hz only` | You passed audio at another rate to `wiener_filter` / `wiener_oracle`. Load it with `wiener.audio_io.read_wav`, which resamples to 16 kHz |
| `refusing to write inside the dataset folder` | `--out` points into the dataset. Choose another folder |
| Viewer shows "⚠ ... 2 candidates for processed" | The run folder contains two processed-type folders (e.g. `processed/` and `enhanced/`). Keep one |
| Viewer: sample-rate / length mismatch | `processed/` files from some other tool don't match the clean and noisy files. Wiener output always matches |
| Viewer in Docker shows no files | `DATA_DIR` path is wrong, or it relies on symlinks that point outside the mounted folder |
| Run is slow on the train splits | It is limited by disk reads. Use `--cache-dir ../.cache/speechprep` so each file is read and resampled only once across all models. `--jobs` above the core count won't help |
| macOS `._p232_001.wav` files | AppleDouble metadata files on non-APFS drives. Both the baseline and the viewer ignore them |

---

## 14. References

- Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral
  amplitude estimator," *IEEE Trans. ASSP*, 1984. (Source of the decision-directed a priori SNR.)
- R. Martin, "Noise power spectral density estimation based on optimal smoothing and minimum
  statistics," *IEEE Trans. Speech and Audio Processing*, 2001.
- P. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013.
- C. Valentini-Botinhao et al., "Noisy speech database for training speech enhancement algorithms
  and TTS models," University of Edinburgh, 2017. <https://doi.org/10.7488/ds/2117>
