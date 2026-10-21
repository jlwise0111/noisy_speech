import csv
import json

import soundfile as sf

import train_unet


def test_train_then_enhance_writes_viewer_triplets(tiny_dataset, tmp_path):
    out = tmp_path / "outputs"
    train_unet.main([
        "--dataset", str(tiny_dataset), "--out", str(out), "--cache-dir", str(tmp_path / "cache"),
        "--run-name", "unet_tiny", "--val-frac", "0.25", "--epochs", "2", "--batch-size", "2",
        "--crop-frames", "64", "--base-channels", "4", "--workers", "0", "--stats-files", "4",
    ])
    run = out / "unet_tiny"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["n_test"] == 6 and cfg["val_speakers"] and cfg["best_epoch"] in (1, 2)
    assert len(list(csv.DictReader(open(run / "history.csv")))) == 2
    for stem in ("p232_001", "p257_003"):
        infos = [sf.info(str(run / role / f"{stem}.wav")) for role in ("clean", "noisy", "enhanced")]
        assert {(i.samplerate, i.channels, i.frames) for i in infos} == {(16000, 1, infos[0].frames)}

    # Re-enhancing from the saved checkpoint reproduces the files exactly.
    first = sf.read(str(run / "enhanced" / "p232_001.wav"))[0]
    train_unet.main(["--dataset", str(tiny_dataset), "--out", str(out), "--cache-dir", str(tmp_path / "cache"),
                     "--run-name", "unet_tiny", "--model", str(run / "best.pt"), "--skip-train",
                     "--workers", "0"])
    assert (sf.read(str(run / "enhanced" / "p232_001.wav"))[0] == first).all()
