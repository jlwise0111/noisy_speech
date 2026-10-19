"""Discovery of clean / noisy / processed .wav triplets in a dataset folder.

Two layouts are supported:

* ``subfolders``: ``root/<role folder>/<stem>.wav``  (e.g. clean/p232_001.wav)
* ``suffix``:     ``root/**/<stem>_<role suffix>.wav`` (e.g. p232_001_noisy.wav)

Each role accepts several aliases (e.g. processed = processed | enhanced), so a
stem can end up with zero or several candidate files for a role. Such triplets
are returned as unresolved and the UI asks the user to pick the files.
"""

from __future__ import annotations

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


def _is_wav(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".wav" and not p.name.startswith("._")


def list_wavs(root: Path) -> list[Path]:
    """All .wav files under root (recursive), sorted. Used for manual picking."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if _is_wav(p))


def _discover_subfolders(root: Path, aliases: dict[str, list[str]]) -> dict[str, Triplet]:
    """Role folders may sit at any depth; triplets are grouped by the role folders' parent."""
    found: dict[str, Triplet] = {}
    parents = [root] + sorted(d for d in root.rglob("*") if d.is_dir())
    for parent in parents:
        children = {c.name.lower(): c for c in parent.iterdir() if c.is_dir()}
        rel_dir = parent.relative_to(root)
        for role in ROLES:
            for alias in aliases.get(role, []):
                folder = children.get(alias.lower())
                if folder is None:
                    continue
                for wav in sorted(folder.iterdir()):
                    if _is_wav(wav):
                        key = wav.stem if str(rel_dir) == "." else f"{rel_dir.as_posix()}/{wav.stem}"
                        t = found.setdefault(key, Triplet(key, "subfolders"))
                        t.candidates[role].append(wav)
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


def _discover_suffix(root: Path, aliases: dict[str, list[str]]) -> dict[str, Triplet]:
    found: dict[str, Triplet] = {}
    for wav in list_wavs(root):
        split = _split_suffix(wav.stem, aliases)
        if split is None:
            continue
        stem, role = split
        # Keep stems unique across subdirectories by including the relative dir.
        rel_dir = wav.parent.relative_to(root)
        key = stem if str(rel_dir) == "." else f"{rel_dir.as_posix()}/{stem}"
        t = found.setdefault(key, Triplet(key, "suffix"))
        t.candidates[role].append(wav)
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

    results: dict[str, Triplet] = {}
    if layout in ("auto", "subfolders"):
        results.update(_discover_subfolders(root, folder_aliases))
    if layout in ("auto", "suffix"):
        for key, t in _discover_suffix(root, suffix_aliases).items():
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
