"""2D U-Net that predicts a time-frequency ratio mask in [0, 1] from a noisy spectrogram.

    noisy log|Y| (B, 257, T)
      -> per-bin standardization (training-set mean/std, stored in the model)
      -> pad to multiples of 16 -> 4 x [DoubleConv, MaxPool 2x2] -> bottleneck DoubleConv
      -> 4 x [ConvTranspose 2x2, concat skip, DoubleConv] -> 1x1 conv -> sigmoid
      -> crop back to (B, 257, T) = mask M;   enhanced STFT = M * Y (noisy phase kept)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEVELS = 4                 # downsampling blocks; each halves freq and time
MULTIPLE = 2 ** N_LEVELS     # input sizes are padded to a multiple of this


class DoubleConv(nn.Sequential):
    """(3x3 conv -> BatchNorm -> ReLU) twice. Padding 1 keeps the size."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """2D U-Net on the noisy log-magnitude spectrogram, predicting a [0, 1] mask per bin."""
    def __init__(self, n_bins: int = 257, base_channels: int = 16):
        super().__init__()
        self.n_bins = n_bins
        self.base_channels = base_channels
        ch = [base_channels * 2 ** i for i in range(N_LEVELS + 1)]   # 16, 32, 64, 128, 256

        # Training-set log-magnitude mean/std per frequency bin (set with set_normalization).
        self.register_buffer("in_mean", torch.zeros(n_bins))
        self.register_buffer("in_std", torch.ones(n_bins))

        self.down = nn.ModuleList(DoubleConv(c_in, c_out) for c_in, c_out in zip([1] + ch[:-2], ch[:-1]))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(ch[-2], ch[-1])
        self.up = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(N_LEVELS)))
        self.up_conv = nn.ModuleList(DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(N_LEVELS)))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def set_normalization(self, mean, std) -> None:
        """Store the training-set per-bin mean/std; saved in the checkpoint with the weights."""
        self.in_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def forward(self, log_mag: torch.Tensor) -> torch.Tensor:
        """log_mag: (B, F, T) noisy log-magnitude -> mask (B, F, T) in [0, 1]."""
        _, n_f, n_t = log_mag.shape
        x = (log_mag - self.in_mean[:, None]) / self.in_std[:, None]
        # Zero padding after standardization means "an average bin", not silence.
        x = F.pad(x, (0, -n_t % MULTIPLE, 0, -n_f % MULTIPLE)).unsqueeze(1)

        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.up, self.up_conv, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))

        return torch.sigmoid(self.head(x))[:, 0, :n_f, :n_t]


def n_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
