"""Model player: run every trained model on one file and switch between the outputs live.

Importing this package puts the project's own packages on sys.path, so the models run with
exactly the code that trained and scored them (speechprep, unet, lrb, wiener) and the display
reuses the diagnostic viewer's DSP and figures (viewer).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Where the project's code lives: noisy_speech/ locally, /app in the Docker image.
CODE_ROOT = Path(os.environ.get("PLAYER_CODE_ROOT", Path(__file__).resolve().parents[2]))
# Where the trained runs live (*/outputs/*): the same folder locally, mounted at /models in Docker.
MODELS_ROOT = Path(os.environ.get("PLAYER_MODELS_DIR", CODE_ROOT))
# The Valentini dataset (optional: only needed to pick train/test files).
DATASET_ROOT = os.environ.get("PLAYER_DATASET_DIR") or os.environ.get("VALENTINI_DIR") or ""

for sub in ("", "unet_model", "linear_baseline", "wiener_baseline", "diagnostic_viewer"):
    p = str(CODE_ROOT / sub) if sub else str(CODE_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
