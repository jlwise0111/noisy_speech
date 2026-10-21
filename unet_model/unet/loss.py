"""Voice-activity-weighted mask loss.

Every frame gets a weight: 1 for speech, `silence_weight` for silent frames, 0 for batch padding.
The loss is the weighted mean of the per-bin error, normalized by the total weight, so changing
`silence_weight` re-balances speech vs silence without changing the overall loss scale (and with
it the effective learning rate). silence_weight = 1 is exactly the plain (unweighted) loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def frame_weights(speech: torch.Tensor, valid: torch.Tensor, silence_weight: float) -> torch.Tensor:
    """speech, valid: (B, T) bool -> (B, T) float weights."""
    w = torch.where(speech, torch.ones_like(speech, dtype=torch.float32),
                    torch.full_like(speech, silence_weight, dtype=torch.float32))
    return w * valid.float()


def weighted_mask_loss(pred, target, speech, valid, silence_weight: float = 1.0,
                       kind: str = "mse", huber_delta: float = 0.1) -> torch.Tensor:
    """pred, target: (B, F, T) masks; speech, valid: (B, T) bool. Returns a scalar."""
    if kind == "mse":
        err = (pred - target) ** 2
    elif kind == "huber":
        # Mask errors live in [0, 1]; with the usual delta = 1 Huber would just be 0.5 * MSE.
        err = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    else:
        raise ValueError(f"unknown loss {kind!r} (mse or huber)")
    w = frame_weights(speech, valid, silence_weight)              # (B, T)
    total = (err.mean(dim=1) * w).sum()                           # mean over freq, weighted over time
    return total / w.sum().clamp_min(1e-8)


@torch.no_grad()
def masked_mse_parts(pred, target, speech, valid) -> dict:
    """Unweighted squared-error sums and frame counts for speech / silent frames (for logging).

    Logged for every run so runs with different silence weights can be compared on the same scale.
    """
    se = ((pred - target) ** 2).mean(dim=1)                       # (B, T)
    sp, si = speech & valid, ~speech & valid
    return {"se_speech": float(se[sp].sum()), "n_speech": int(sp.sum()),
            "se_silent": float(se[si].sum()), "n_silent": int(si.sum())}
