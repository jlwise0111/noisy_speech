#!/usr/bin/env python3
"""Run every baseline in parallel, then train the U-Net on its own, score them all, then open the viewer.

    python run_pipeline.py --dataset /path/to/valentini_noisy_speech_database     # or: make pipeline
    python run_pipeline.py --smoke --skip-viewer                                   # minutes, tiny limits

Phases:
    1. check     dataset folders, model environments, Docker (warned about up front, not hours later)
    2. cache     read + resample every needed 48 kHz file to 16 kHz once (speechprep.WaveCache), so no model
                 re-reads the dataset drive and none of them resamples the same file again
    3. baselines Ridge (28spk / 56spk / both) and Wiener (leading / minstats / oracle) run side by side,
                 sharing the whole core budget
    4. U-Net     trains alone afterwards with every core: PyTorch fixes its thread count when it starts,
                 so it only gets the whole machine if nothing else is running
    5. evaluate  the metric suite scores every run in one shared process pool
    6. viewer    docker compose up, wait for the health check, open the browser

No hardware is assumed. The core budget is what Python reports for this process (`--cores` overrides it)
and no GPU is needed: the U-Net uses one only if PyTorch finds it. Each baseline gets a slice of the
budget, passed to it as its worker count, and every child process has its BLAS / OpenMP pools pinned to
one thread, so the processes together never ask for more cores than there are (no thrashing). A job only
starts when its cores are free; when a job ends, its cores go to the next waiting one. A failed baseline
does not stop the U-Net, but it does stop the pipeline before evaluation.

Runs that already finished are skipped (same markers as the Makefile); an interrupted U-Net run resumes
from last.pt. --force redoes everything. Logs: logs/<run>.log.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import webbrowser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOGS = REPO / "logs"
UNET_RUN = "unet_28spk"
HEARTBEAT_S = 300               # how often to print the latest log line of each running job

# Pin every BLAS / OpenMP pool to one thread inside child processes. Parallelism comes from the worker
# counts each job is given; a hidden thread pool per process on top of that is what oversubscribes the CPU.
SINGLE_THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}


def available_cores() -> int:
    """Cores this process may use (respects CPU affinity / container limits where Python can see them)."""
    count = getattr(os, "process_cpu_count", None) or os.cpu_count
    return max(1, count() or 1)


def venv_python(folder: Path) -> Path:
    """The interpreter of `folder`/.venv on any platform."""
    win = folder / ".venv" / "Scripts" / "python.exe"
    return win if win.exists() else folder / ".venv" / "bin" / "python"


@dataclass
class Job:
    name: str
    cwd: Path
    cmd: list[str]
    cores: int
    done: callable = field(repr=False)     # () -> bool: True if a previous run already finished
    needs: tuple[str, ...] = ()            # dataset folders whose 16 kHz cache this job reads
    stage: str = "baselines"               # jobs of one stage run together; stages run one after another


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("VALENTINI_DIR"),
                    help="Valentini root folder (default: $VALENTINI_DIR)")
    ap.add_argument("--cores", type=int, default=available_cores(),
                    help=f"core budget shared by all jobs (default: detected, {available_cores()} here)")
    ap.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "speechprep",
                    help="shared 16 kHz waveform cache (default: <project>/.cache/speechprep)")
    ap.add_argument("--only", help="run only the jobs whose name matches this regex, e.g. 'ridge|wiener'")
    ap.add_argument("--unet-args", default="", help="extra train_unet.py arguments, e.g. '--device cpu --epochs 10'")
    ap.add_argument("--force", action="store_true", help="redo finished runs and rescore everything")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: few files, 2 U-Net epochs, separate smoke_* runs and report")
    ap.add_argument("--skip-viewer", action="store_true", help="do not start the viewer at the end")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8501)), help="viewer port")
    args = ap.parse_args(argv)
    if not args.dataset:
        ap.error("--dataset is required (or set VALENTINI_DIR)")
    args.dataset = Path(args.dataset).expanduser()
    args.cores = max(1, args.cores)
    return args


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- jobs
def build_jobs(args) -> list[Job]:
    """Every model run, grouped into stages. Core shares are fractions of the budget, never fixed numbers."""
    cores, ds, cache = args.cores, str(args.dataset), str(args.cache_dir)
    prefix = "smoke_" if args.smoke else ""
    py = sys.executable

    # Baselines: three at a time, each with a third of the budget (one after another on a single core).
    # U-Net: runs alone afterwards, so its loader workers plus torch threads add up to the whole budget.
    base_cores = max(1, cores // 3)
    workers = min(4, max(0, (cores - 1) // 2))
    unet_par = ["--workers", str(workers), "--val-workers", "1" if cores >= 3 else "0",
                "--threads", str(max(1, cores - workers))]

    unet_dir, linear, wiener = REPO / "unet_model", REPO / "linear_baseline", REPO / "wiener_baseline"
    test = ("clean_testset_wav", "noisy_testset_wav")
    spk = {s: (f"clean_trainset_{s}_wav", f"noisy_trainset_{s}_wav") for s in ("28spk", "56spk")}

    unet_name = prefix + UNET_RUN
    unet_cfg = unet_dir / "outputs" / unet_name / "config.json"
    unet_cmd = [str(venv_python(unet_dir)), "-u", "train_unet.py", "--dataset", ds, "--train-set", "28spk",
                "--run-name", unet_name, "--cache-dir", cache, *unet_par]
    if args.smoke:
        unet_cmd += ["--limit-files", "300", "--epochs", "2", "--test-limit", "20"]
    if not args.force:
        unet_cmd.append("--resume")
    unet_cmd += shlex.split(args.unet_args)
    jobs = []

    for train_set in ("28spk", "56spk", "both"):
        name = f"{prefix}ridge_{train_set}"
        marker = linear / "outputs" / name / "config.json"
        cmd = [py, "-u", "ridge_baseline.py", "--dataset", ds, "--train-set", train_set, "--run-name", name,
               "--workers", str(base_cores), "--cache-dir", cache]
        if args.smoke:
            cmd += ["--limit-files", "500", "--test-limit", "20"]
        needs = (spk["28spk"] + spk["56spk"] if train_set == "both" else spk[train_set]) + test
        jobs.append(Job(name, linear, cmd, base_cores, marker.exists, needs))

    for name, extra in (("wiener_leading_test", ["--noise-est", "leading"]),
                        ("wiener_minstats_test", ["--noise-est", "minstats"]),
                        ("oracle_test", ["--method", "oracle"])):
        name = prefix + name
        marker = wiener / "outputs" / name / "quick_metrics.csv"
        cmd = [py, "-u", "run_baseline.py", "--dataset", ds, "--split", "test", *extra, "--run-name", name,
               "--jobs", str(base_cores), "--cache-dir", cache]
        if args.smoke:
            cmd += ["--limit", "20"]
        jobs.append(Job(name, wiener, cmd, base_cores, marker.exists, test))

    jobs.append(Job(unet_name, unet_dir, unet_cmd, cores,
                    lambda: unet_cfg.exists() and '"seconds"' in unet_cfg.read_text(), spk["28spk"] + test,
                    stage="U-Net"))

    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j.name)]
    return jobs


# --------------------------------------------------------------------------------------------- phases
def check(args, jobs: list[Job]) -> None:
    """Fail fast on anything that would stop the pipeline later; warn early if the viewer cannot start."""
    missing = [d for j in jobs for d in j.needs if not (args.dataset / d).is_dir()]
    if missing:
        sys.exit(f"✗ dataset folder(s) missing under {args.dataset}: {', '.join(sorted(set(missing)))}")
    for j in jobs:
        exe = Path(j.cmd[0])
        if exe.is_absolute() and not exe.exists():
            sys.exit(f"✗ {exe} not found. Create it with: make {'unet-venv' if 'unet' in j.name else 'metrics-venv'}")
    if not venv_python(REPO / "metric_suite").exists():
        sys.exit("✗ metric_suite/.venv not found. Create it with: make metrics-venv")
    if not args.skip_viewer and not docker_running():
        log("⚠ Docker is not running. Models will still train and be scored; start Docker before they "
            "finish for the viewer to open, or open it later with: make viewer")
    log(f"✓ checks passed: {len(jobs)} job(s), core budget {args.cores}")


def docker_running() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except OSError:
        return False


def warm_cache(args, jobs: list[Job]) -> None:
    """Resample every file the pending jobs read to 16 kHz once, on every core, before they start."""
    folders = sorted({d for j in jobs for d in j.needs})
    if not folders or args.smoke:          # smoke runs touch few files; the models fill the cache as they go
        return
    sys.path.insert(0, str(REPO))
    from speechprep import WaveCache

    cache = WaveCache(args.cache_dir)
    paths = [p for d in folders for p in (args.dataset / d).glob("*.wav") if not p.name.startswith("._")]
    t0 = time.time()
    log(f"▶ 16 kHz cache: checking {len(paths)} files in {len(folders)} folders ({args.cores} processes)")
    added = cache.warm(paths, args.cores)
    log(f"✓ 16 kHz cache: {added} file(s) added in {time.time() - t0:.0f}s")


def run_job(job: Job) -> tuple[Job, int, float]:
    """Run one job to completion with its output in logs/<name>.log (runs in a supervising thread)."""
    t0 = time.time()
    env = {**os.environ, **SINGLE_THREAD_ENV, "PYTHONUNBUFFERED": "1"}
    with open(LOGS / f"{job.name}.log", "w") as out:
        out.write(f"$ (cd {job.cwd.name} && {shlex.join(job.cmd)})\n\n")
        out.flush()
        proc = subprocess.Popen(job.cmd, cwd=job.cwd, stdout=out, stderr=subprocess.STDOUT, env=env)
        _running[job.name] = proc
        try:
            code = proc.wait()
        finally:
            _running.pop(job.name, None)
    return job, code, time.time() - t0


_running: dict[str, subprocess.Popen] = {}


def last_line(name: str) -> str:
    try:
        lines = (LOGS / f"{name}.log").read_text(errors="replace").strip().splitlines()
        return lines[-1][:120] if lines else ""
    except OSError:
        return ""


def pending_jobs(args, jobs: list[Job]) -> list[Job]:
    """The jobs still to run: finished ones are skipped unless --force."""
    pending = []
    for j in jobs:
        if not args.force and j.done():
            log(f"✓ skip {j.name}: already finished (--force to redo)")
        else:
            pending.append(j)
    return pending


def run_stages(args, jobs: list[Job]) -> list[str]:
    """Run each stage's jobs together, one stage after another, in build order. Returns failures."""
    failed = []
    for stage in dict.fromkeys(j.stage for j in jobs):
        group = [j for j in jobs if j.stage == stage]
        log(f"▶ stage: {stage} ({len(group)} job(s), core budget {args.cores})")
        failed += run_models(args, group)
    return failed


def run_models(args, pending: list[Job]) -> list[str]:
    """Start jobs in order whenever their cores are free (later, smaller jobs may fill gaps). Returns failures."""
    LOGS.mkdir(exist_ok=True)
    free, futures, failed = args.cores, {}, []
    last_beat = time.time()
    with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        try:
            while pending or futures:
                for j in list(pending):
                    cost = min(j.cores, args.cores)
                    if cost <= free:
                        pending.remove(j)
                        free -= cost
                        futures[pool.submit(run_job, j)] = cost
                        log(f"▶ start {j.name} on {cost} core(s)  (log: logs/{j.name}.log)")
                done, _ = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
                for f in done:
                    free += futures.pop(f)
                    job, code, secs = f.result()
                    if code == 0:
                        log(f"✓ done  {job.name} in {secs / 60:.1f} min")
                    else:
                        failed.append(job.name)
                        log(f"✗ FAILED {job.name} (exit {code}): {last_line(job.name)}  — see logs/{job.name}.log")
                if futures and time.time() - last_beat >= HEARTBEAT_S:
                    last_beat = time.time()
                    for name in sorted(_running):
                        log(f"  … {name}: {last_line(name)}")
        except KeyboardInterrupt:
            log("interrupted: stopping running jobs (an interrupted U-Net run resumes next time)")
            for proc in list(_running.values()):
                proc.terminate()
            raise
    return failed


def evaluate(args) -> int:
    metrics = REPO / "metric_suite"
    cmd = [str(venv_python(metrics)), "-u", "evaluate_models.py", "--workers", str(args.cores)]
    if args.smoke:
        cmd += ["--include", "smoke_", "--exclude", "", "--out", "results/smoke"]
    if args.force:
        cmd.append("--force")
    log("▶ evaluate: scoring every run (PESQ-WB, STOI, SNR, SI-SDR)")
    code = subprocess.run(cmd, cwd=metrics, env={**os.environ, **SINGLE_THREAD_ENV}).returncode
    report = "metric_suite/results/" + ("smoke/" if args.smoke else "") + "summary.md"
    log(f"✓ report: {report}" if code == 0 else f"✗ evaluation failed (exit {code})")
    return code


def viewer(args) -> int:
    if not docker_running():
        log("✗ Docker is not running, so the viewer was not started. Start Docker, then: make viewer")
        return 1
    url = f"http://localhost:{args.port}"
    compose = ["docker", "compose"]
    env = {**os.environ, "PORT": str(args.port)}
    cwd = REPO / "diagnostic_viewer"
    log("▶ viewer: building and starting the container")
    for step in (["build"], ["up", "-d"]):
        if subprocess.run(compose + step, cwd=cwd, env=env).returncode != 0:
            log("✗ viewer did not start; try: make viewer")
            return 1
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2):
                break
        except OSError:
            time.sleep(2)
    else:
        log("✗ viewer is not answering; see: make viewer-logs")
        return 1
    log(f"✓ viewer: {url}  (stop it with: make viewer-close)")
    webbrowser.open(url)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    jobs = build_jobs(args)
    check(args, jobs)
    pending = pending_jobs(args, jobs)
    warm_cache(args, pending)
    failed = run_stages(args, pending)
    if failed:
        log(f"✗ {len(failed)} job(s) failed: {', '.join(failed)}. Fix and re-run: finished runs are skipped.")
        return 1
    code = evaluate(args)
    if code == 0 and not args.skip_viewer:
        code = viewer(args)
    log(f"pipeline finished in {(time.time() - t0) / 60:.1f} min")
    return code


if __name__ == "__main__":
    sys.exit(main())
