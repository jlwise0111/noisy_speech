import csv
import os

import numpy as np
import soundfile as sf

import evaluate_models
from msuite.runs import OTHER, classify, find_runs
from tests.test_metrics import speech_like

SR = 16000


def write_run(root, rel, processed_name="enhanced", n=2, gain=0.02):
    rng = np.random.default_rng(0)
    for i in range(n):
        x = speech_like(seed=i)
        noisy = x + 0.08 * rng.normal(size=x.size)
        for role, y in (("clean", x), ("noisy", noisy), (processed_name, x + gain * rng.normal(size=x.size))):
            p = root / rel / role / f"p232_00{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, y, SR)


def test_classify_matches_viewer_rules():
    assert classify("linear_baseline/outputs/ridge_28spk") == "Ridge baseline"
    assert classify("wiener_baseline/outputs/oracle_test") == "Wiener baseline"
    assert classify("neural_network/outputs/crn") == "Neural network"
    assert classify("channel_tests") == OTHER


def test_find_runs_layout_and_default_excludes(tmp_path):
    write_run(tmp_path, "linear_baseline/outputs/ridge_a")
    write_run(tmp_path, "wiener_baseline/outputs/wiener_leading_test", processed_name="processed")
    write_run(tmp_path, "wiener_baseline/outputs/wiener_leading_train28", processed_name="processed")
    write_run(tmp_path, "diagnostic_viewer/data/demo/subfolders", processed_name="processed")
    (tmp_path / "lonely" / "clean").mkdir(parents=True)  # not a run: no noisy/processed
    rels = [r.rel for r in find_runs(tmp_path)]
    assert rels == ["linear_baseline/outputs/ridge_a", "wiener_baseline/outputs/wiener_leading_test"]
    assert len(find_runs(tmp_path, exclude=None)) == 4
    assert [r.rel for r in find_runs(tmp_path, include="ridge")] == ["linear_baseline/outputs/ridge_a"]


def test_cli_end_to_end_and_cache(tmp_path, capsys):
    write_run(tmp_path, "linear_baseline/outputs/ridge_a", gain=0.01)
    write_run(tmp_path, "wiener_baseline/outputs/wiener_x", processed_name="processed", gain=0.05)
    out = tmp_path / "results"
    argv = ["--root", str(tmp_path), "--out", str(out), "--workers", "2"]
    assert evaluate_models.main(argv) == 0
    rows = list(csv.DictReader(open(out / "summary.csv")))
    assert [r["system"] for r in rows] == ["Wiener baseline", "Ridge baseline"]
    for r in rows:
        assert r["ok"] == "2"
        assert float(r["pesq_wb_delta"]) > 0 and float(r["snr_db_delta"]) > 0
        assert float(r["pesq_wb_vs_ref"]) == round(float(r["pesq_wb_processed"]) - 1.50, 4)
    assert "| Ridge baseline |" in (out / "summary.md").read_text()
    per_file = out / "runs" / "linear_baseline__outputs__ridge_a.csv"
    assert len(list(csv.DictReader(open(per_file)))) == 2

    # Second run reuses the cache; touching an output invalidates it.
    capsys.readouterr()
    evaluate_models.main(argv)
    assert capsys.readouterr().out.count("(cached") == 2
    wav = tmp_path / "linear_baseline/outputs/ridge_a/enhanced/p232_000.wav"
    os.utime(wav, (per_file.stat().st_mtime + 10,) * 2)
    evaluate_models.main(argv)
    assert capsys.readouterr().out.count("(cached") == 1
