import numpy as np
import pytest
import torch

from speechprep import stft
from unet import UNet, n_parameters
from unet.loss import masked_mse_parts, weighted_mask_loss
from unet.targets import mask_target, speech_frames


@pytest.mark.parametrize("n_frames", [1, 17, 256, 301])
def test_mask_shape_and_range_for_any_length(n_frames):
    model = UNet(base_channels=4).eval()
    x = torch.randn(2, 257, n_frames) * 3 - 5
    with torch.no_grad():
        m = model(x)
    assert m.shape == (2, 257, n_frames)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0


def test_default_size_and_normalization_is_saved():
    model = UNet()
    assert 1.5e6 < n_parameters(model) < 2.5e6
    model.set_normalization(np.full(257, -4.0), np.full(257, 2.0))
    restored = UNet()
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored.in_mean, torch.full((257,), -4.0))


def test_can_overfit_one_example():
    """Gradients reach every layer: the network memorizes a single spectrogram/mask pair."""
    torch.manual_seed(0)
    model = UNet(base_channels=4)
    x, y = torch.randn(1, 257, 32), torch.rand(1, 257, 32)
    speech, valid = torch.ones(1, 32, dtype=torch.bool), torch.ones(1, 32, dtype=torch.bool)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = weighted_mask_loss(model(x), y, speech, valid)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.2 * first


def _batch():
    g = torch.Generator().manual_seed(0)
    pred, target = torch.rand(2, 257, 10, generator=g), torch.rand(2, 257, 10, generator=g)
    speech = torch.zeros(2, 10, dtype=torch.bool)
    speech[:, 3:7] = True
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 8:] = False
    return pred, target, speech, valid


def test_weight_one_is_plain_mse_over_valid_frames():
    pred, target, speech, valid = _batch()
    plain = ((pred - target) ** 2).mean(dim=1)[valid].mean()
    assert torch.isclose(weighted_mask_loss(pred, target, speech, valid, silence_weight=1.0), plain)


def test_padding_never_counts():
    pred, target, speech, valid = _batch()
    before = weighted_mask_loss(pred, target, speech, valid, 0.25)
    pred2 = pred.clone()
    pred2[1, :, 8:] = 1 - target[1, :, 8:]          # change only padded frames
    assert torch.isclose(before, weighted_mask_loss(pred2, target, speech, valid, 0.25))


def test_silence_weight_downweights_silent_errors():
    pred, target, speech, valid = _batch()
    pred = target.clone()
    pred[:, :, ~speech[0]] += 0.5                   # error only in silent frames
    full = weighted_mask_loss(pred, target, speech, valid, 1.0)
    down = weighted_mask_loss(pred, target, speech, valid, 0.1)
    assert down < full
    assert float(weighted_mask_loss(pred, target, speech, valid, 0.0)) == 0.0
    parts = masked_mse_parts(pred, target, speech, valid)
    assert parts["se_speech"] == 0.0 and parts["se_silent"] > 0


def test_huber_is_not_just_half_mse():
    pred, target, speech, valid = _batch()
    h = weighted_mask_loss(pred, target, speech, valid, 1.0, kind="huber")
    m = weighted_mask_loss(pred, target, speech, valid, 1.0, kind="mse")
    assert not torch.isclose(h, 0.5 * m)


def test_targets_and_vad():
    rng = np.random.default_rng(0)
    t = np.arange(32000) / 16000
    clean = np.sin(2 * np.pi * 200 * t) * (t > 1.0)
    noisy = clean + 0.01 * rng.standard_normal(len(t))
    C, Y = stft(clean), stft(noisy)
    irm, psm = mask_target(C, Y, "irm"), mask_target(C, Y, "psm")
    assert irm.min() >= 0 and irm.max() <= 1 and np.all(psm <= irm + 1e-6)
    loud = np.abs(Y) > 1e-3
    np.testing.assert_allclose(mask_target(Y, Y, "irm")[loud], 1.0, atol=1e-4)
    speech = speech_frames(C)
    assert speech[-50:].all() and not speech[:100].any()
