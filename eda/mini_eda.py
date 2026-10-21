import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load a paired sample (dataset folder at the repo root, wherever the script is run from)
DATA = Path(__file__).resolve().parent.parent / "valentini_eda"
clean_file = DATA / "clean_trainset_28spk_wav" / "p226_001.wav"
noisy_file = DATA / "noisy_trainset_28spk_wav" / "p226_001.wav"

# Enforces sampling rate of 16,000 samples/second (16 kHz)
y_clean, sr = librosa.load(clean_file, sr=16000)
y_noisy, _ = librosa.load(noisy_file, sr=16000)

# Compute STFT (Short-Time Fourier Transform)
# Transforms 1D waves into 2D 'images'
# This allows us to use standard 2D Convolutional Neural Networks to clean audio like an image-denoising task
D_clean = np.abs(librosa.stft(y_clean, n_fft=512, hop_length=256))
D_noisy = np.abs(librosa.stft(y_noisy, n_fft=512, hop_length=256))

# Calculate ground-truth Ideal Ratio Mask (IRM)
# Masking vs. Direct Synthesis:
# Instead of training a neural network to generate audio waveforms (prone to hallucinations and harsh digital artifacts),
# the network is trained as a suppressor/filter.
# How to interpret: if the pixel is 1.0: entirely clean voice, keep it.
# If the pixel is 0.0, no speech, only background noise, mute it
# Clipping enforces a valid ratio that can be trained against a standard Sigmoid activation function using MSE loss.
irm = np.clip(D_clean / (D_noisy + 1e-8), 0.0, 1.0)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 1. Waveforms
# Used for: checking clipping and saturation,
# observing the noise energy floor (should be 0 in clean),
# inspecting silence buffers
librosa.display.waveshow(y_clean, sr=sr, ax=axes[0, 0], color="blue")
axes[0, 0].set_title("Clean Waveform (Time Domain)")
librosa.display.waveshow(y_noisy, sr=sr, ax=axes[0, 1], color="red")
axes[0, 1].set_title("Noisy Waveform (Time Domain)")

# 2. Spectrograms
# Why this is done:
# Decibel Conversion: Converting to decibels makes subtle vocal formants clearly visible.
# Visualizing Acoustic Signatures
# Model Justification
img1 = librosa.display.specshow(
    librosa.amplitude_to_db(D_clean, ref=np.max),
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[1, 0],
)
axes[1, 0].set_title("Clean Spectrogram (dB)")
fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")

img2 = librosa.display.specshow(
    librosa.amplitude_to_db(D_noisy, ref=np.max),
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="hz",
    ax=axes[1, 1],
)
axes[1, 1].set_title("Noisy Spectrogram (dB)")
fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

# 3. Average Power Spectrum & Ideal Ratio Mask
# Why this is done:
# Quantifying Frequency Overlap: The gap between noisy and clean shows how much excess energy background noise injects into each frequency band.
# Validating Target Formulation:
# Because target values are skewed toward the extremes (especially 1.0 for high SNR bins, and 0.0 during non-speech pauses), standard MSE will work well, but we should evaluate whether Binary Cross-Entropy (BCE) or Huber/Smooth L1 Loss helps the model learn the subtle mid-range transition boundaries (0.2 to 0.8) better.
axes[2, 0].plot(
    np.mean(D_clean, axis=1), label="Clean Speech", color="blue", linewidth=1.5
)
axes[2, 0].plot(
    np.mean(D_noisy, axis=1),
    label="Noisy Speech",
    color="red",
    alpha=0.7,
    linewidth=1.5,
)
axes[2, 0].set_title("Power Spectral Density Across Frequency Bins")
axes[2, 0].set_xlabel("Frequency Bin (0 to 256)")
axes[2, 0].set_ylabel("Average Magnitude")
axes[2, 0].legend()

axes[2, 1].hist(irm.flatten(), bins=40, color="purple", edgecolor="black")
axes[2, 1].set_title("Distribution of Ideal Ratio Mask Values [0, 1]")
axes[2, 1].set_xlabel("Mask Value")
axes[2, 1].set_ylabel("Pixel Count")

plt.tight_layout()
plt.show()