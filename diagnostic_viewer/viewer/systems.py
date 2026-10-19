"""Group discovered triplets into processing systems (neural network, baselines) and their runs.

A *run* is the folder a triplet was found in, e.g. ``linear_baseline/outputs/ridge_28spk``. Each run
is assigned a *system* by looking for keywords in its path, so the viewer can offer
"System → Run → File" and keep the same file selected when switching between systems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .dataset import Triplet

# Display order. Keywords are matched against the words of every folder in the run's path
# (split on _ - . and spaces), so "linear_baseline/outputs/ridge_28spk" matches "linear" and "ridge".
SYSTEMS = (
    ("Neural network", ("neural", "nn", "dnn", "network", "unet")),
    ("Wiener baseline", ("wiener", "oracle")),
    ("Ridge baseline", ("ridge", "linear")),
)
OTHER = "Other"
SYSTEM_NAMES = tuple(name for name, _ in SYSTEMS)
ROOT_RUN = "."


@dataclass
class Run:
    path: str  # folder relative to the data root ("." for the root itself)
    system: str
    triplets: list[int] = field(default_factory=list)  # indices into the discovered triplet list

    @property
    def label(self) -> str:
        return "(data folder root)" if self.path == ROOT_RUN else self.path

    @property
    def name(self) -> str:
        """Short name for menus: the run's own folder, e.g. "ridge_28spk"."""
        return "(data folder root)" if self.path == ROOT_RUN else self.path.rstrip("/").rpartition("/")[2]


def split_key(key: str) -> tuple[str, str]:
    """Triplet key "a/b/p232_001" -> run "a/b", file stem "p232_001"."""
    run, _, stem = key.rpartition("/")
    return run or ROOT_RUN, stem


def classify(run_path: str) -> str:
    words = set(re.split(r"[_\-.\s/]+", run_path.lower()))
    for name, keywords in SYSTEMS:
        if words & set(keywords):
            return name
    return OTHER


def group_runs(triplets: list[Triplet]) -> dict[str, list[Run]]:
    """{system: [Run, ...]} for every system in SYSTEMS (empty list if no outputs) plus Other if used."""
    runs: dict[str, Run] = {}
    for i, t in enumerate(triplets):
        path, _ = split_key(t.stem)
        if path not in runs:
            runs[path] = Run(path, classify(path))
        runs[path].triplets.append(i)
    out: dict[str, list[Run]] = {name: [] for name in SYSTEM_NAMES}
    for run in sorted(runs.values(), key=lambda r: r.path):
        out.setdefault(run.system, []).append(run)
    return out
