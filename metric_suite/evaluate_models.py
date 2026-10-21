"""Automated metric suite: PESQ-WB, STOI and SNR gain for every model's test-set predictions.

Finds every run folder under the project (``<run>/{clean,noisy,enhanced|processed}/<stem>.wav``,
the layout the baselines and diagnostic_viewer use), scores processed and noisy against clean, and
compares the results with the raw noisy baseline: both as measured on the same files and against
fixed reference values (default PESQ-WB 1.50, STOI 0.83).

Usage (from metric_suite/):
    python evaluate_models.py                       # all runs under the project folder
    python evaluate_models.py --include ridge       # only runs whose path matches a regex
    python evaluate_models.py --limit 20 --force    # quick rescore of 20 files per run
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msuite import evaluate, runs  # noqa: E402

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE.parent, help="folder to search for runs (default: project)")
    ap.add_argument("--include", help="only runs whose relative path matches this regex")
    ap.add_argument("--exclude", default=runs.DEFAULT_EXCLUDE,
                    help="skip runs whose relative path matches this regex (default skips training-split, "
                         "smoke and demo outputs; pass '' to include everything)")
    ap.add_argument("--ref-pesq", type=float, default=1.50, help="fixed noisy reference for PESQ-WB")
    ap.add_argument("--ref-stoi", type=float, default=0.83, help="fixed noisy reference for STOI")
    ap.add_argument("--limit", type=int, help="score only the first N files of each run")
    ap.add_argument("--workers", type=int,
                    default=max(1, ((getattr(os, "process_cpu_count", None) or os.cpu_count)() or 2) - 1),
                    help="parallel scoring processes, shared by all runs (default: available cores minus one)")
    ap.add_argument("--force", action="store_true", help="rescore even if cached results are up to date")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    return ap.parse_args(argv)


def shown_root(root: Path) -> str:
    """A folder as printed in reports and on screen: relative to the project folder, never an absolute path."""
    root, project = root.resolve(), HERE.parent.resolve()
    if root == project:
        return "the project folder"
    try:
        return root.relative_to(project).as_posix()
    except ValueError:
        return root.name


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    found = runs.find_runs(args.root, args.include, args.exclude or None)
    if not found:
        print(f"No runs found under {shown_root(args.root)} (include={args.include!r}, exclude={args.exclude!r}).")
        return 1
    print(f"Found {len(found)} run(s) under {shown_root(args.root)}:")
    args.out.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        all_rows = evaluate.evaluate_runs(found, args.out, pool, args.limit, args.force)
    summaries = [evaluate.summarize(run, rows, args.ref_pesq, args.ref_stoi) for run, rows in zip(found, all_rows)]
    order = {name: i for i, (name, _) in enumerate(runs.SYSTEMS)}
    summaries.sort(key=lambda s: (order.get(s["system"], len(order)), s["run"]))

    evaluate.write_summary_csv(args.out / "summary.csv", summaries)
    (args.out / "summary.md").write_text(
        evaluate.markdown_report(summaries, args.ref_pesq, args.ref_stoi, shown_root(args.root)))
    print()
    print(evaluate.terminal_table(summaries, args.ref_pesq, args.ref_stoi))
    out = shown_root(args.out)
    print(f"\nWrote {out}/summary.md, {out}/summary.csv and per-file CSVs in "
          f"{out}/runs ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
