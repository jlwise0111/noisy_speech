# Automated Metric Suite

A terminal script that scores **every model's test-set predictions** with **PESQ-WB**, **STOI** and **SNR gain**. It compares each model against the raw noisy input in two ways:

- **measured:** the noisy file vs clean, on the same files;
- **reference:** fixed noisy-baseline values. The defaults are **PESQ 1.50** and **STOI 0.83**; change them with `--ref-pesq` and `--ref-stoi`.

The metric code is extracted from `diagnostic_viewer/viewer/metrics.py`, and run discovery uses the same folder rules as the viewer. A number in this report therefore matches what the viewer shows for that file.

## Quick start (from the project folder)

```bash
make evaluate          # score every run already on disk → metric_suite/results/summary.md
make train-evaluate    # first train/run every model (finished runs are skipped), then score them all
make test-metrics      # unit tests (synthetic audio, no dataset needed)
```

The first `make evaluate` creates `metric_suite/.venv` with numpy, scipy, soundfile, `pesq` and `pystoi`, so your main Python environment is not changed. To pass options through: `make evaluate METRIC_ARGS="--force --include ridge"`.

Or run it directly:

```bash
cd metric_suite
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python evaluate_models.py                     # every run under the project folder
.venv/bin/python evaluate_models.py --include wiener    # only runs whose path matches a regex
.venv/bin/python evaluate_models.py --limit 50 --force  # quick rescore, 50 files per run
```

## What counts as a run

A run is any folder that has `clean/`, `noisy/` and one of `processed/`, `enhanced/`, `denoised/` or `output/`, each holding wavs with the same names. Every baseline already writes this layout:

| System (from folder words) | Example run |
|---|---|
| Neural network (`neural`, `nn`, `dnn`, `network`, `unet`) | `unet_model/outputs/unet_28spk/` |
| Wiener baseline (`wiener`, `oracle`) | `wiener_baseline/outputs/wiener_leading_test/` |
| Ridge baseline (`ridge`, `linear`) | `linear_baseline/outputs/ridge_28spk/` |

The U-Net (`unet_model/`) writes this layout too, so it is picked up automatically.

Some folders are skipped by default (override with `--exclude ''`):
- training-split outputs (paths containing `train`);
- `smoke` test runs;
- the viewer's synthetic demo data;
- the suite's own `results/`.

The three files of a triplet must share sample rate and length. A file that doesn't is listed with a `rejected:` status instead of scores.

## Metrics

| Metric | Definition |
|---|---|
| PESQ-WB | ITU-T P.862.2 wide-band, via `pesq` at 16 kHz. 48 kHz runs (Wiener) are resampled for scoring only. Range about 1.04–4.64 |
| STOI | Short-time objective intelligibility, via `pystoi` (resamples to 10 kHz internally). Range 0–1 |
| SNR gain | SNR(processed vs clean) − SNR(noisy vs clean), in dB. Plain SNR: it penalises gain changes and phase errors |
| SI-SDR gain | Scale-invariant version, also reported |

For PESQ and STOI:
- *Δ* = processed − noisy, measured on the same files;
- *vs ref* = processed − the fixed reference.

## Output (`metric_suite/results/`, git-ignored)

| File | Contents |
|---|---|
| `summary.md` | Report table per system and run, plus the measured noisy baseline per run |
| `summary.csv` | The same numbers, machine-readable (means over the files of each run) |
| `runs/<run>.csv` | One row per file: status, sample rate, duration, and noisy / processed / delta for every metric |

Per-run results are **cached**. A run is rescored only when its output files are newer than its CSV, when the file list changed, or when you pass `--force`.

## Results on the Valentini test set (824 files, 2026-10-19)

| System | Run | PESQ-WB | Δ | vs 1.50 | STOI | Δ | vs 0.83 | SNR gain |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Wiener (oracle, upper bound) | `oracle_test` | 3.83 | +1.87 | +2.33 | 0.983 | +0.062 | +0.153 | +12.65 dB |
| Wiener | `wiener_leading_test` | 2.42 | +0.46 | +0.92 | 0.912 | −0.010 | +0.082 | +7.52 dB |
| Wiener | `wiener_minstats_test` | 2.31 | +0.34 | +0.81 | 0.908 | −0.013 | +0.078 | +4.84 dB |
| Ridge | `ridge_28spk` | 1.61 | −0.35 | +0.11 | 0.913 | −0.008 | +0.083 | −6.84 dB |
| Ridge | `ridge_56spk` | 1.58 | −0.39 | +0.08 | 0.914 | −0.007 | +0.084 | −6.95 dB |
| Ridge | `ridge_both` | 1.59 | −0.38 | +0.09 | 0.914 | −0.007 | +0.084 | −6.94 dB |

The **measured** noisy input of this test set scores **PESQ-WB 1.97, STOI 0.921, SNR 8.4 dB**, in line with published Valentini figures. That is higher than the fixed 1.50 / 0.83 reference. So "vs ref" can look like an improvement where "Δ" (same files) shows none, as for Ridge. Use Δ to judge whether a model helps on this data.

## Files

| Path | Contents |
|---|---|
| `evaluate_models.py` | CLI |
| `msuite/metrics.py` | SNR, SI-SDR, PESQ-WB, STOI (extracted from the viewer) |
| `msuite/runs.py` | Run discovery, system naming, triplet loading and validation |
| `msuite/evaluate.py` | Parallel scoring, per-run cache, summary CSV / Markdown / terminal table |
| `tests/` | Metric values, run discovery and excludes, end-to-end CLI and cache invalidation |
