"""Find model output runs and their clean/noisy/processed file triplets.

A *run* is a folder with ``clean/``, ``noisy/`` and a processed folder (``processed``, ``enhanced``,
``denoised`` or ``output``) holding wavs with matching names, the layout every model in this project
writes and diagnostic_viewer reads. Systems are named with the same folder-word rules as
diagnostic_viewer/viewer/systems.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

CLEAN, NOISY = "clean", "noisy"
PROCESSED_ALIASES = ("processed", "enhanced", "denoised", "output")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
SYSTEMS = (
    ("Neural network", ("neural", "nn", "dnn", "network", "unet")),
    ("Wiener baseline", ("wiener", "oracle")),
    ("Ridge baseline", ("ridge", "linear")),
)
OTHER = "Other"
# Training-split outputs, smoke tests and the viewer's synthetic demo files are not evaluations.
DEFAULT_EXCLUDE = r"train|smoke|(^|/)diagnostic_viewer(/|$)|(^|/)metric_suite(/|$)"


@dataclass
class Run:
    path: Path  # absolute
    rel: str  # relative to the search root, posix
    system: str
    clean_dir: Path
    noisy_dir: Path
    processed_dir: Path

    def stems(self) -> list[str]:
        """Names present in all three folders, sorted."""
        def names(d):
            return {p.stem for p in d.glob("*.wav") if not p.name.startswith("._")}
        return sorted(names(self.clean_dir) & names(self.noisy_dir) & names(self.processed_dir))

    def files(self, stem: str) -> tuple[Path, Path, Path]:
        return (self.clean_dir / f"{stem}.wav", self.noisy_dir / f"{stem}.wav", self.processed_dir / f"{stem}.wav")

    def newest_mtime(self) -> float:
        return max((p.stat().st_mtime for p in self.processed_dir.glob("*.wav")), default=0.0)


def classify(rel: str) -> str:
    words = set(re.split(r"[_\-.\s/]+", rel.lower()))
    for name, keywords in SYSTEMS:
        if words & set(keywords):
            return name
    return OTHER


def find_runs(root: Path, include: str | None = None, exclude: str | None = DEFAULT_EXCLUDE) -> list[Run]:
    """All run folders under `root`, filtered by regexes on their relative path."""
    root = Path(root).resolve()
    runs = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        lower = {d.lower(): d for d in dirnames}
        proc = next((lower[a] for a in PROCESSED_ALIASES if a in lower), None)
        if CLEAN in lower and NOISY in lower and proc:
            path = Path(dirpath)
            rel = path.relative_to(root).as_posix()
            if (include is None or re.search(include, rel)) and not (exclude and re.search(exclude, rel)):
                runs.append(Run(path, rel, classify(rel), path / lower[CLEAN], path / lower[NOISY], path / proc))
            dirnames[:] = []  # a run's role folders contain only audio
    return runs


class TripletError(ValueError):
    pass


def load_triplet(clean: Path, noisy: Path, processed: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read three wavs as mono float32. They must share sample rate and length (as in the viewer)."""
    sigs, rates = [], []
    for p in (clean, noisy, processed):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        sigs.append(x.mean(axis=1))
        rates.append(sr)
    if len(set(rates)) != 1:
        raise TripletError(f"sample rates differ: {rates}")
    if len({len(x) for x in sigs}) != 1:
        raise TripletError(f"lengths differ: {[len(x) for x in sigs]}")
    return sigs[0], sigs[1], sigs[2], rates[0]
