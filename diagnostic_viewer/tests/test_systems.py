from pathlib import Path

from viewer.dataset import Triplet
from viewer.systems import OTHER, SYSTEM_NAMES, classify, group_runs, split_key


def test_classify_by_folder_words():
    assert classify("linear_baseline/outputs/ridge_28spk") == "Ridge baseline"
    assert classify("wiener_baseline/outputs/oracle_test") == "Wiener baseline"
    assert classify("neural_network/outputs/crn_v1") == "Neural network"
    assert classify("nn/run3") == "Neural network"
    assert classify("diagnostic_viewer/data/demo/subfolders") == OTHER
    assert classify("channel_tests") == OTHER  # "nn" only matches a whole word


def test_split_key():
    assert split_key("a/b/p232_001") == ("a/b", "p232_001")
    assert split_key("s0") == (".", "s0")


def test_group_runs_lists_every_system_even_without_outputs():
    keys = ["linear_baseline/outputs/ridge_a/s0", "linear_baseline/outputs/ridge_a/s1",
            "wiener_baseline/outputs/w/s0", "misc/s9"]
    triplets = [Triplet(k, "subfolders", {"clean": [Path("x")]}) for k in keys]
    g = group_runs(triplets)
    assert list(g)[:3] == list(SYSTEM_NAMES)
    assert g["Neural network"] == []
    assert [r.path for r in g["Ridge baseline"]] == ["linear_baseline/outputs/ridge_a"]
    assert g["Ridge baseline"][0].triplets == [0, 1]
    assert g["Wiener baseline"][0].triplets == [2]
    assert g[OTHER][0].path == "misc"
