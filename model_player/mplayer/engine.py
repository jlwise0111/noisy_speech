"""Model registry and parallel processing.

Every system the pipeline trained is found under MODELS_ROOT and run with the project's own
inference code, on 16 kHz mono float32 audio. All outputs are forced to exactly the input length
and clipped to ±1, so every track has the same number of samples and can play in lock-step.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import MODELS_ROOT  # noqa: F401  (also sets sys.path for the imports below)

SR = 16000
TRAIN_SETS = ("28spk", "56spk", "both")


@dataclass(frozen=True)
class ModelSpec:
    key: str          # unique id, e.g. "unet_28spk"
    family: str       # ridge | wiener | unet | oracle
    label: str        # short name shown on the button / panel
    description: str  # longer text shown on the panel label
    path: Path        # weights file (or params.json for Wiener)
    trained_on: str | None = None  # "28spk" | "56spk" | "both" | None (not trained)

    @property
    def needs_clean(self) -> bool:
        return self.family == "oracle"


def registry(models_root: Path | str = MODELS_ROOT) -> tuple[list[ModelSpec], list[str]]:
    """(available models in display order, notes about expected runs that are missing)."""
    root = Path(models_root)
    want: list[ModelSpec] = []
    for t in TRAIN_SETS:
        want.append(ModelSpec(f"ridge_{t}", "ridge", f"Ridge {t}", f"ridge regression, trained on {t}",
                              root / "linear_baseline/outputs" / f"ridge_{t}" / "model.npz", t))
    want.append(ModelSpec("wiener_minstats", "wiener", "Wiener min-stats", "Wiener filter, minimum-statistics noise",
                          root / "wiener_baseline/outputs/wiener_minstats_test/params.json"))
    want.append(ModelSpec("wiener_leading", "wiener", "Wiener leading", "Wiener filter, leading-noise estimate",
                          root / "wiener_baseline/outputs/wiener_leading_test/params.json"))
    for t in TRAIN_SETS:
        want.append(ModelSpec(f"unet_{t}", "unet", f"U-Net {t}", f"U-Net ratio mask, trained on {t}",
                              root / "unet_model/outputs" / f"unet_{t}" / "best.pt", t))
    want.append(ModelSpec("oracle", "oracle", "Oracle", "ideal Wiener mask from the clean file (ceiling)",
                          root / "wiener_baseline/outputs/oracle_test/params.json"))
    found = [m for m in want if m.path.is_file()]
    missing = [f"{m.label} (no `{m.path.relative_to(root)}`)" for m in want if not m.path.is_file()]
    return found, missing


Enhancer = Callable[[np.ndarray, "np.ndarray | None"], np.ndarray]


def _wiener_params(path: Path):
    from wiener import WienerParams
    return WienerParams(**json.loads(path.read_text())["params"])


def load(spec: ModelSpec) -> Enhancer:
    """Load one model once; returns f(noisy, clean) -> enhanced (any length, any range)."""
    if spec.family == "ridge":
        from lrb import ridge, synth
        z = np.load(spec.path)
        model = ridge.Model(W=z["W"], b=z["b"], alpha=z["alpha"])
        n_fft, hop = int(z["n_fft"]), int(z["hop"])
        return lambda noisy, clean: synth.enhance(noisy, model, n_fft, hop)
    if spec.family == "wiener":
        from wiener import wiener_filter
        params = _wiener_params(spec.path)
        method = json.loads(spec.path.read_text()).get("noise_est", "leading")
        return lambda noisy, clean: wiener_filter(noisy, SR, method, params)
    if spec.family == "oracle":
        from wiener import wiener_oracle
        params = _wiener_params(spec.path)
        return lambda noisy, clean: wiener_oracle(noisy, clean, SR, params)
    if spec.family == "unet":
        import torch
        from unet import load_model
        from unet.enhance import enhance
        model = load_model(spec.path, "cpu")
        return lambda noisy, clean: enhance(model, noisy, "cpu")
    raise ValueError(f"unknown model family {spec.family!r}")


def load_all(specs: list[ModelSpec]) -> dict[str, Enhancer]:
    _set_torch_threads(len(specs))
    return {s.key: load(s) for s in specs}


def _set_torch_threads(n_workers: int) -> None:
    """Split the CPU between the models running at the same time (torch would grab every core each)."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // max(1, min(n_workers, 4))))


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Exactly n float32 samples in [-1, 1] (trim or zero-pad)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return np.clip(y, -1.0, 1.0)


def run_all(
    specs: list[ModelSpec],
    enhancers: dict[str, Enhancer],
    noisy: np.ndarray,
    clean: np.ndarray | None = None,
    on_done: Callable[[ModelSpec, int, int], None] | None = None,
    workers: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Run every model concurrently. Returns ({key: output}, {key: error message})."""
    noisy = np.asarray(noisy, dtype=np.float32)
    todo = [s for s in specs if s.key in enhancers and (clean is not None or not s.needs_clean)]
    out: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    workers = workers or min(len(todo), os.cpu_count() or 4) or 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model") as pool:
        futs = {pool.submit(enhancers[s.key], noisy.copy(), clean): s for s in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            s = futs[fut]
            try:
                out[s.key] = fit_length(fut.result(), len(noisy))
            except Exception as e:  # one broken model must not take the others down
                errors[s.key] = f"{type(e).__name__}: {e}"
            if on_done:
                on_done(s, i, len(todo))
    return {s.key: out[s.key] for s in todo if s.key in out}, errors
