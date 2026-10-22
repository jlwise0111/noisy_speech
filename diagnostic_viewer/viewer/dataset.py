"""Discovery of clean / noisy / processed .wav triplets in a dataset folder.

Two layouts are supported:

* ``subfolders``: ``root/<role folder>/<stem>.wav``  (e.g. clean/p232_001.wav)
* ``suffix``:     ``root/**/<stem>_<role suffix>.wav`` (e.g. p232_001_noisy.wav)

Each role accepts several aliases (e.g. processed = processed | enhanced), so a
stem can end up with zero or several candidate files for a role. Such triplets
are returned as unresolved and the UI asks the user to pick the files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("clean", "noisy", "processed")

DEFAULT_FOLDER_ALIASES = {
    "clean": ["clean", "clean_testset_wav", "clean_trainset_wav"],
    "noisy": ["noisy", "noisy_testset_wav", "noisy_trainset_wav"],
    "processed": ["processed", "enhanced", "denoised", "output"],
}
DEFAULT_SUFFIX_ALIASES = {
    "clean": ["clean"],
    "noisy": ["noisy"],
    "processed": ["processed", "enhanced", "denoised"],
}
SUFFIX_SEPARATORS = ("_", "-", ".")


@dataclass
class Triplet:
    stem: str
    layout: str
    candidates: dict[str, list[Path]] = field(default_factory=lambda: {r: [] for r in ROLES})

    @property
    def resolved(self) -> bool:
        return all(len(self.candidates[r]) == 1 for r in ROLES)

    @property
    def files(self) -> dict[str, Path | None]:
        return {r: (c[0] if len(c) == 1 else None) for r, c in self.candidates.items()}

    @property
    def issues(self) -> list[str]:
        out = []
        for r in ROLES:
            n = len(self.candidates[r])
            if n == 0:
                out.append(f"missing {r}")
            elif n > 1:
                out.append(f"{n} candidates for {r}")
        return out

    @property
    def label(self) -> str:
        if self.resolved:
            return f"✓ {self.stem}"
        return f"⚠ {self.stem} ({', '.join(self.issues)})"


SKIP_DIRS = {"__pycache__", "node_modules"}
# Runs written by `make pipeline-smoke` (a few files, two epochs): pipeline checks, not results.
# The metric suite leaves them out of its report too.
SKIP_PREFIXES = ("smoke_",)

# {relative dir ("" for the root): (sorted sub-folder names, sorted .wav file names)}
Tree = dict[str, tuple[list[str], list[str]]]


def scan(root: Path) -> Tree:
    """One pass over the folder tree with os.scandir.

    The directory entries already say whether each name is a file or a folder, so this needs no
    per-file stat calls. That matters on Docker bind mounts, where every stat is slow: a whole
    project of ~120k wavs scans in well under a second. Hidden entries (".git", macOS "._*"
    resource forks), cache folders and smoke-test runs are skipped.
    """
    root = Path(root)
    tree: Tree = {}
    stack = [""]
    while stack:
        rel = stack.pop()
        dirs, wavs = [], []
        try:
            with os.scandir(root / rel) as entries:
                for e in entries:
                    if e.name.startswith("."):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        if e.name not in SKIP_DIRS and not e.name.startswith(SKIP_PREFIXES):
                            dirs.append(e.name)
                    elif e.name.lower().endswith(".wav") and e.is_file():
                        wavs.append(e.name)
        except OSError:  # unreadable folder: skip it, like a missing one
            continue
        tree[rel] = (sorted(dirs), sorted(wavs))
        stack.extend(f"{rel}/{d}" if rel else d for d in dirs)
    return tree


def _join(rel: str, name: str) -> str:
    return f"{rel}/{name}" if rel else name


def list_wavs(root: Path, tree: Tree | None = None) -> list[Path]:
    """All .wav files under root (recursive), sorted. Used for manual picking."""
    root = Path(root)
    if not root.is_dir():
        return []
    tree = scan(root) if tree is None else tree
    return [root / rel / name for rel in sorted(tree) for name in tree[rel][1]]


def _discover_subfolders(root: Path, aliases: dict[str, list[str]], tree: Tree) -> dict[str, Triplet]:
    """Role folders may sit at any depth; triplets are grouped by the role folders' parent."""
    found: dict[str, Triplet] = {}
    for rel in sorted(tree):
        children = {name.lower(): name for name in tree[rel][0]}
        for role in ROLES:
            for alias in aliases.get(role, []):
                folder = children.get(alias.lower())
                if folder is None:
                    continue
                folder_rel = _join(rel, folder)
                for name in tree[folder_rel][1]:
                    stem = name[:-4]
                    key = _join(rel, stem)
                    t = found.setdefault(key, Triplet(key, "subfolders"))
                    t.candidates[role].append(root / folder_rel / name)
    return found


def _split_suffix(name: str, aliases: dict[str, list[str]]) -> tuple[str, str] | None:
    """'p232_001_noisy' -> ('p232_001', 'noisy')."""
    lower = name.lower()
    for role in ROLES:
        for alias in aliases.get(role, []):
            for sep in SUFFIX_SEPARATORS:
                tail = f"{sep}{alias.lower()}"
                if lower.endswith(tail) and len(name) > len(tail):
                    return name[: -len(tail)], role
    return None


def _discover_suffix(root: Path, aliases: dict[str, list[str]], tree: Tree) -> dict[str, Triplet]:
    found: dict[str, Triplet] = {}
    for rel in sorted(tree):
        for name in tree[rel][1]:
            split = _split_suffix(name[:-4], aliases)
            if split is None:
                continue
            stem, role = split
            # Keep stems unique across subdirectories by including the relative dir.
            key = _join(rel, stem)
            t = found.setdefault(key, Triplet(key, "suffix"))
            t.candidates[role].append(root / rel / name)
    return found


def discover(
    root: Path | str,
    layout: str = "auto",
    folder_aliases: dict[str, list[str]] | None = None,
    suffix_aliases: dict[str, list[str]] | None = None,
) -> list[Triplet]:
    """Find triplets under ``root``. ``layout`` is 'auto', 'subfolders' or 'suffix'."""
    root = Path(root)
    folder_aliases = folder_aliases or DEFAULT_FOLDER_ALIASES
    suffix_aliases = suffix_aliases or DEFAULT_SUFFIX_ALIASES
    if not root.is_dir():
        return []

    tree = scan(root)
    results: dict[str, Triplet] = {}
    if layout in ("auto", "subfolders"):
        results.update(_discover_subfolders(root, folder_aliases, tree))
    if layout in ("auto", "suffix"):
        for key, t in _discover_suffix(root, suffix_aliases, tree).items():
            if key in results:
                # Same stem found by both layouts: merge candidates, flag ambiguity.
                for r in ROLES:
                    results[key].candidates[r].extend(
                        p for p in t.candidates[r] if p not in results[key].candidates[r]
                    )
                results[key].layout = "mixed"
            else:
                results[key] = t
    return sorted(results.values(), key=lambda t: (not t.resolved, t.stem))
