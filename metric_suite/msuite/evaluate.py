"""Score every file of every run, cache per-run results, and build the summary tables."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .metrics import score_file
from .runs import Run, TripletError, load_triplet

METRICS = ("pesq_wb", "stoi", "snr_db", "si_sdr_db")
FILE_COLUMNS = (["stem", "status", "sample_rate_hz", "duration_s"]
                + [f"{m}_{k}" for m in METRICS for k in ("noisy", "processed", "delta")])


def _score(stem: str, run: Run) -> dict:
    row = {"stem": stem}
    try:
        clean, noisy, processed, sr = load_triplet(*run.files(stem))
    except TripletError as e:
        return {**row, "status": f"rejected: {e}"}
    except Exception as e:
        return {**row, "status": f"error: {e}"}
    scores = score_file(clean, noisy, processed, sr)
    failed = [m for m in ("pesq_wb", "stoi") if math.isnan(scores[f"{m}_processed"])
              or math.isnan(scores[f"{m}_noisy"])]
    status = "ok" if not failed else f"partial: {', '.join(failed)} not computable"
    return {**row, "status": status, "sample_rate_hz": sr, "duration_s": round(len(clean) / sr, 3), **scores}


def cache_path(results_dir: Path, run: Run) -> Path:
    return results_dir / "runs" / (run.rel.replace("/", "__") + ".csv")


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FILE_COLUMNS[2:]:
            r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
    return rows


def _score_job(job: tuple[int, str, Run]) -> tuple[int, dict]:
    i, stem, run = job
    return i, _score(stem, run)


def evaluate_runs(found: list[Run], results_dir: Path, pool: Pool | None, limit: int | None = None,
                  force: bool = False, log=print) -> list[list[dict]]:
    """Per-file rows for every run, in the order of `found`.

    A run whose cached CSV is newer than its outputs is read back. Every other (run, file) pair goes
    into ONE shared work queue, so all runs are scored at the same time and the pool never idles
    waiting for the last files of one run before starting the next."""
    results: list[list[dict] | None] = [None] * len(found)
    jobs = []
    for i, run in enumerate(found):
        stems = run.stems()[:limit] if limit else run.stems()
        path = cache_path(results_dir, run)
        if not force and path.exists() and path.stat().st_mtime >= run.newest_mtime():
            rows = _read_csv(path)
            if [r["stem"] for r in rows] == stems:
                log(f"  ✓ {run.rel}: {len(rows)} files (cached; --force to rescore)")
                results[i] = rows
                continue
        log(f"  ▶ {run.rel}: scoring {len(stems)} files…")
        jobs += [(i, stem, run) for stem in stems]

    scored: dict[int, dict[str, dict]] = {}
    done_jobs = pool.imap_unordered(_score_job, jobs, chunksize=4) if pool else map(_score_job, jobs)
    for n, (i, row) in enumerate(done_jobs, 1):
        scored.setdefault(i, {})[row["stem"]] = row
        if n % 500 == 0 or n == len(jobs):
            log(f"    {n}/{len(jobs)} files scored")

    for i, by_stem in scored.items():
        run = found[i]
        stems = run.stems()[:limit] if limit else run.stems()
        rows = [by_stem[s] for s in stems]
        path = cache_path(results_dir, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, FILE_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        results[i] = rows
    return [r if r is not None else [] for r in results]


def evaluate_run(run: Run, results_dir: Path, pool: Pool | None, limit: int | None = None,
                 force: bool = False, log=print) -> list[dict]:
    """Per-file rows for one run, reusing the cached CSV when it is newer than the run's outputs."""
    return evaluate_runs([run], results_dir, pool, limit, force, log)[0]


def summarize(run: Run, rows: list[dict], ref_pesq: float, ref_stoi: float) -> dict:
    """Mean of every metric over the files where it could be computed."""
    s = {"system": run.system, "run": run.rel, "files": len(rows),
         "ok": sum(r["status"] == "ok" for r in rows)}
    for m in METRICS:
        for k in ("noisy", "processed", "delta"):
            vals = np.array([r.get(f"{m}_{k}", np.nan) for r in rows], dtype=float)
            s[f"{m}_{k}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    s["pesq_wb_vs_ref"] = s["pesq_wb_processed"] - ref_pesq
    s["stoi_vs_ref"] = s["stoi_processed"] - ref_stoi
    return s


SUMMARY_COLUMNS = ("system", "run", "files", "ok",
                   "pesq_wb_noisy", "pesq_wb_processed", "pesq_wb_delta", "pesq_wb_vs_ref",
                   "stoi_noisy", "stoi_processed", "stoi_delta", "stoi_vs_ref",
                   "snr_db_noisy", "snr_db_processed", "snr_db_delta",
                   "si_sdr_db_noisy", "si_sdr_db_processed", "si_sdr_db_delta")


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


def _f(v: float, d: int = 2, sign: bool = False) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def markdown_report(summaries: list[dict], ref_pesq: float, ref_stoi: float, root: Path | str) -> str:
    """`root` is shown as given: pass a path relative to the project so the report names no machine."""
    lines = [
        "# Metric suite report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{root}`.",
        "",
        "Each metric is scored against the clean reference and averaged over the files of the run.",
        "*Δ* = processed − noisy, measured on the same files. *vs ref* = processed − the fixed noisy",
        f"reference (PESQ-WB {ref_pesq:.2f}, STOI {ref_stoi:.2f}). SNR gain = SNR(processed) − SNR(noisy).",
        "",
        "| System | Run | Files | PESQ-WB | Δ | vs ref | STOI | Δ | vs ref | SNR (dB) | SNR gain | SI-SDR gain |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['system']} | `{s['run']}` | {s['ok']}/{s['files']} "
            f"| {_f(s['pesq_wb_processed'])} | {_f(s['pesq_wb_delta'], sign=True)} | {_f(s['pesq_wb_vs_ref'], sign=True)} "
            f"| {_f(s['stoi_processed'], 3)} | {_f(s['stoi_delta'], 3, True)} | {_f(s['stoi_vs_ref'], 3, True)} "
            f"| {_f(s['snr_db_processed'])} | {_f(s['snr_db_delta'], sign=True)} | {_f(s['si_sdr_db_delta'], sign=True)} |")
    lines += ["", "## Noisy input, as measured", "",
              "| Run | PESQ-WB | STOI | SNR (dB) |", "|---|--:|--:|--:|"]
    for s in summaries:
        lines.append(f"| `{s['run']}` | {_f(s['pesq_wb_noisy'])} | {_f(s['stoi_noisy'], 3)} | {_f(s['snr_db_noisy'])} |")
    lines += ["", f"Fixed reference: PESQ-WB {ref_pesq:.2f}, STOI {ref_stoi:.2f}.", ""]
    return "\n".join(lines)


def terminal_table(summaries: list[dict], ref_pesq: float, ref_stoi: float) -> str:
    head = (f"{'System':<16} {'Run':<44} {'PESQ-WB':>7} {'Δ':>6} {'vs ref':>7} "
            f"{'STOI':>6} {'Δ':>7} {'vs ref':>7} {'SNR gain':>9}")
    out = [head, "─" * len(head)]
    for s in summaries:
        out.append(
            f"{s['system']:<16} {s['run'][-44:]:<44} {_f(s['pesq_wb_processed']):>7} "
            f"{_f(s['pesq_wb_delta'], sign=True):>6} {_f(s['pesq_wb_vs_ref'], sign=True):>7} "
            f"{_f(s['stoi_processed'], 3):>6} {_f(s['stoi_delta'], 3, True):>7} {_f(s['stoi_vs_ref'], 3, True):>7} "
            f"{_f(s['snr_db_delta'], sign=True):>8} dB")
    out.append(f"vs ref = processed − fixed noisy reference (PESQ-WB {ref_pesq:.2f}, STOI {ref_stoi:.2f}); "
               "Δ = processed − noisy measured on the same files.")
    return "\n".join(out)
