"""Write a few synthetic demo triplets into the data folder, in both supported layouts.

Usage (from diagnostic_viewer/):
    docker compose run --rm -v ./data:/out viewer python scripts/make_demo_data.py --out /out
"""

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewer.demo import make_triplet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("VIEWER_DATA_DIR", "data"))
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()
    root = Path(args.out) / "demo"

    # Layout 1: subfolders
    for i, snr_db in enumerate((0.0, 5.0)):
        sig = make_triplet(args.sr, 4.0, snr_db, seed=i)
        for role, x in zip(("clean", "noisy", "processed"), sig):
            p = root / "subfolders" / role / f"demo_{i:03d}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, x, args.sr)

    # Layout 2: suffixes in one folder (one triplet left incomplete on purpose)
    for i, snr_db in enumerate((-5.0, 10.0)):
        sig = make_triplet(args.sr, 3.0, snr_db, seed=10 + i)
        for role, x in zip(("clean", "noisy", "processed"), sig):
            if i == 1 and role == "processed":
                continue
            p = root / "suffix" / f"utt{i:02d}_{role}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, x, args.sr)
    print(f"Demo data written to {root}")


if __name__ == "__main__":
    main()
