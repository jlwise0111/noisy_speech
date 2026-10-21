# U-Net ratio-mask model (Person B)

A 2D U-Net that predicts a time-frequency mask in [0, 1] from the noisy log-magnitude spectrogram. The mask
multiplies the noisy STFT (noisy phase kept) and is inverted back to audio. It is trained with a
silence-weighted mask loss and early stopping on held-out speakers.

Preprocessing comes from the shared [`../speechprep`](../speechprep) package. Its 16 kHz resampling, STFT
(Hann, `n_fft=512`, `hop=128` → 257 bins) and speaker split are identical to the Ridge baseline, and the
tests check this. U-Net and Ridge therefore see the same input, the same grid and the same validation speakers.

## Setup and usage

```bash
cd unet_model
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # or: make unet-venv
.venv/bin/python -m pytest -q                                          # 18 tests, synthetic data only

DS=/path/to/valentini_noisy_speech_database
.venv/bin/python train_unet.py --dataset "$DS" --limit-files 300 --epochs 2 --test-limit 20 --run-name unet_smoke
.venv/bin/python train_unet.py --dataset "$DS" --train-set 28spk --run-name unet_28spk      # full run
.venv/bin/python train_unet.py --dataset "$DS" --model outputs/unet_28spk/best.pt --skip-train  # re-enhance
```

`--resume` continues an interrupted run from `outputs/<run>/last.pt`. The first run caches 16 kHz waveforms
in `../.cache/speechprep` (about 4 GB for 28spk + test, under a minute on 16 cores). Run names must not contain
`train`, because the metric suite skips such folders.

## Pipeline

| Step | Detail |
|---|---|
| Input | noisy `log(|Y| + 1e-6)`, standardized per frequency bin with training-set mean/std (stored in the checkpoint) |
| Target | `irm` = `clip(|C|/|Y|, 0, 1)` (default) or `psm` = `clip(|C|/|Y|·cos(Δθ), 0, 1)` |
| Network | pad to multiples of 16 → 4 × [DoubleConv(3×3 conv, BN, ReLU ×2), MaxPool 2] → bottleneck → 4 × [ConvTranspose 2×2, concat skip, DoubleConv] → 1×1 conv → sigmoid → crop. Channels 16-32-64-128-256 (`--base-channels`), 1.94 M parameters |
| VAD | frame is silent if clean frame energy < utterance peak − `--silence-db` (30 dB → 28% of 28spk frames; 40 dB → 5%) |
| Loss | per-frame weighted MSE (or Huber, δ = 0.1): weight 1 speech, `--silence-weight` (0.25) silence, 0 padding, normalized by total weight. `--silence-weight 1` = plain MSE |
| Training | random 256-frame crops (2.05 s), batch 16, AdamW (lr 1e-3, wd 1e-2), ReduceLROnPlateau (×0.5, patience 2) or `--scheduler cosine`, grad clip 5, early stop after 6 stale epochs, seed 0 |
| Selection | best validation loss on held-out speakers (28spk seed 0: p256, p279, p287, same as Ridge). Unweighted speech/silent MSE is logged too, so runs with different silence weights stay comparable |
| Output | `outputs/<run>/{clean,noisy,enhanced}/<stem>.wav` at 16 kHz, plus `best.pt`, `last.pt`, `history.csv`, `config.json` |

Device: `--device auto` picks CUDA, then MPS, then CPU. On the project's Intel Mac the AMD GPU (MPS) runs a
16 × 256 batch in about 1.6 s, against 3.0 s on the CPU. A full 28spk epoch (652 batches + validation)
takes about 20 minutes.
