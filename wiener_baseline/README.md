# Wiener filter baseline (classical DSP)

A non-learned speech-enhancement baseline for the noisy-speech project. It runs a
Wiener filter over the Valentini (VoiceBank-DEMAND) noisy/clean pairs and writes
clean / noisy / processed triplets in the layout `diagnostic_viewer` reads.

**Full setup and usage guide: [`docs/wiener_baseline.md`](docs/wiener_baseline.md).**

## Algorithm

All processing is done per STFT bin. Defaults: Hann window, 32 ms frames, 75% overlap.

1. **Noise PSD estimate** `N(f)`
   - `leading` (default): mean power over the first 0.25 s. Valentini clips start with a short pause.
   - `minstats`: a simplified minimum-statistics tracker (smoothed periodogram → 1.5 s sliding minimum → bias correction). It follows noise that changes over time.
2. **A priori SNR** by the decision-directed rule (Ephraim–Malah, α = 0.98):
   `ξ_t = α·|Ŝ_{t-1}|²/N + (1−α)·max(|Y_t|²/N − 1, 0)`
3. **Wiener gain** `G = ξ / (1 + ξ)`, floored at −20 dB to limit musical noise. This gain is the per-bin estimator that minimises mean-square error.
4. `Ŝ = G·Y` → ISTFT → trimmed to the input length → clipped to [-1, 1].

`--method oracle` builds the *ideal* Wiener mask `|S|²/(|S|²+|N|²)` from the true clean
signal. It is **not** a baseline, only an upper bound on what any Wiener-type mask could reach.

## Run

Requires numpy, scipy and soundfile (`pip install -r requirements.txt`).

```bash
cd wiener_baseline
DATA="/Volumes/Banton Glyph RAID 2TB/USF/Classes/06 Fall 2026/Machine Learning/Group Project/audio_data_sets/valentini_noisy_speech_database"

python run_baseline.py --dataset "$DATA" --split test --limit 10      # quick smoke run
python run_baseline.py --dataset "$DATA" --split test                 # full 824-file test set
python run_baseline.py --dataset "$DATA" --split test --noise-est minstats
python run_baseline.py --dataset "$DATA" --split test --method oracle # upper bound
```

- `--split`: `test`, `train28` or `train56`.
- `--jobs`: number of parallel processes (default: all cores).
- Every `WienerParams` field can be set from the command line, e.g. `--gain-floor-db -15 --alpha 0.95`.
- The dataset is only read. The script refuses to write inside it.

Output, which is gitignored:

```
outputs/<method>_<noise-est>_<split>/
    clean/<stem>.wav      copied from the dataset
    noisy/<stem>.wav      copied from the dataset
    processed/<stem>.wav  Wiener output: 48 kHz mono PCM_16, same length as noisy
    params.json           settings + git commit
    quick_metrics.csv     per-file SNR / SI-SDR (numpy-only sanity check)
```

A full test-set run copies the clean and noisy files, so each run folder takes about 550 MB.

## View in diagnostic_viewer

The viewer lives on the `feature/diagnostic-viewer` branch. Set its sidebar data
folder (or `DATA_DIR` for Docker) to one run folder, e.g. `wiener_baseline/outputs/wiener_leading_test`,
and press **Rescan folder**. You can also point it at `outputs/` itself: the runs then show up as
`wiener_leading_test/p232_001`, `oracle_test/p232_001`, and so on. Use the viewer's batch export for PESQ / STOI.

To compare the neural net later, write its outputs as `outputs/<nn-run>/processed/<stem>.wav`,
next to copies of `clean/` and `noisy/`, in the same way.

## Results: Valentini test set (824 files, from `quick_metrics.csv`)

| Method                     | SNR (dB) | SI-SDR (dB) | Δ SI-SDR |
|----------------------------|---------:|------------:|---------:|
| Noisy input                |     8.39 |        8.39 |        — |
| Wiener, `leading` noise    |    15.91 |       15.76 |    +7.36 |
| Wiener, `minstats` noise   |    13.24 |       13.03 |    +4.64 |
| Oracle Wiener (upper bound)|    21.05 |       21.01 |   +12.61 |

Train splits (Wiener, `leading`): train28 SI-SDR 6.00 → 11.13 dB (+5.13, 11 572 files);
train56 5.93 → 11.02 dB (+5.09, 23 075 files). See the guide, section 9, for details.

## Tests

```bash
python -m pytest tests -q
```

The tests use synthetic signals only: they check output length, gain bounds, SNR improvement,
that the oracle beats the blind filter, that silent input stays finite, and that the output stays within [-1, 1].
