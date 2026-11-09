"""The browser mixer (the AudioWorklet inside player.html), run under Node with a fake audio thread.

Checks lock-step mixing, the 2 ms equal-power solo crossfade and diff mode (second track minus the
polarity-inverted first). Skipped when Node is not installed (it is not in the Docker image).
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = (Path(__file__).resolve().parents[1] / "mplayer" / "player.html").read_text(encoding="utf-8")
NODE = shutil.which("node")

HARNESS = r"""
const SRC = %s;
globalThis.sampleRate = 16000; globalThis.currentTime = 0;
class AudioWorkletProcessor { constructor() { this.port = { postMessage: (m) => sent.push(m), onmessage: null }; } }
const sent = []; let Proc;
globalThis.AudioWorkletProcessor = AudioWorkletProcessor;
globalThis.registerProcessor = (name, cls) => { Proc = cls; };
new Function(SRC)();
const p = new Proc(); const send = (m) => p.port.onmessage({ data: m });

const n = 16000, sr = 16000;
const tone = (f, a) => Float32Array.from({ length: n }, (_, i) => a * Math.sin(2 * Math.PI * f * i / sr));
const A = tone(220, 0.5), B = Float32Array.from(A, (v, i) => v + 0.1 * Math.sin(2 * Math.PI * 3000 * i / sr)), A2 = A.slice();
send({ type: "load", tracks: [A, B, A2], n, sr, solo: 0, norm: [1, 1, 1], debug: true });
send({ type: "play" });

const out = []; let pos = 0;
const run = (blocks) => { for (let b = 0; b < blocks; b++) {
  const L = [new Float32Array(128), new Float32Array(128)]; p.process([], [L]);
  for (const v of L[0]) out.push(v); } };
const at = () => out.length;
const res = {};
run(10);                                                       // solo A
res.soloA = Math.max(...out.slice(0, at()).map((v, i) => Math.abs(v - A[i])));
const s1 = at(); send({ type: "mix", target: [0, 1, 0] }); run(10);   // solo switch A -> B
res.fade = sent.filter((m) => m.type === "fade").slice(-1)[0];
res.afterSwitch = Math.max(...out.slice(s1 + 32, at()).map((v, k) => Math.abs(v - B[s1 + 32 + k])));
const s2 = at(); send({ type: "mix", target: [-1, 0, 1] }); run(10);  // diff: A2 - A (identical) -> null
res.nullDiff = Math.max(...out.slice(s2 + 32, at()).map(Math.abs));
const s3 = at(); send({ type: "mix", target: [-1, 1, 0] }); run(10);  // diff: B - A
res.diffBA = Math.max(...out.slice(s3 + 32, at()).map((v, k) => Math.abs(v - (B[s3 + 32 + k] - A[s3 + 32 + k]))));
const s4 = at(); send({ type: "mix", target: [0, 1, 0] }); run(10);   // leave diff: solo B
res.leaveDiff = Math.max(...out.slice(s4 + 32, at()).map((v, k) => Math.abs(v - B[s4 + 32 + k])));
// stage 1, loudness gains, ahead of the mixer: solo B at gain 2, then diff 2B - 0.5A, then gains off
const s5 = at(); send({ type: "norm", target: [0.5, 2, 1] }); run(10);
res.normSolo = Math.max(...out.slice(s5 + 32, at()).map((v, k) => Math.abs(v - 2 * B[s5 + 32 + k])));
const s6 = at(); send({ type: "mix", target: [-1, 1, 0] }); run(10);
res.normDiff = Math.max(...out.slice(s6 + 32, at()).map((v, k) => Math.abs(v - (2 * B[s6 + 32 + k] - 0.5 * A[s6 + 32 + k]))));
const s7 = at(); send({ type: "norm", target: [1, 1, 1] }); run(10);
res.normOff = Math.max(...out.slice(s7 + 32, at()).map((v, k) => Math.abs(v - (B[s7 + 32 + k] - A[s7 + 32 + k]))));
let maxJump = 0;
for (let k = 1; k < out.length; k++) maxJump = Math.max(maxJump, Math.abs(out[k] - out[k - 1]));
res.maxJump = maxJump;
res.maxSignalStep = Math.max(...[A, B].map((x) => { let m = 0; for (let k = 1; k < n; k++) m = Math.max(m, Math.abs(x[k] - x[k - 1])); return m; }));
res.fadeLen = p.fadeLen;
console.log(JSON.stringify(res));
"""


@pytest.fixture(scope="module")
def result():
    if not NODE:
        pytest.skip("node is not installed")
    src = re.search(r"const WORKLET = `(.*?)`;", HTML, re.S).group(1)
    js = HARNESS % json.dumps(src)
    out = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_solo_plays_the_track_exactly(result):
    assert result["soloA"] < 1e-6


def test_solo_crossfade_is_2ms_and_equal_power(result):
    assert result["fadeLen"] == 32
    assert result["fade"]["samples"] == 32
    assert result["fade"]["maxPowerError"] < 1e-12
    assert result["afterSwitch"] < 1e-6


def test_diff_of_identical_tracks_nulls(result):
    assert result["nullDiff"] < 1e-7


def test_diff_is_second_minus_first(result):
    assert result["diffBA"] < 1e-6


def test_leaving_diff_solos_the_second_track(result):
    assert result["leaveDiff"] < 1e-6


def test_loudness_gain_is_applied_before_solo(result):
    assert result["normSolo"] < 1e-6


def test_diff_compares_the_loudness_matched_tracks(result):
    assert result["normDiff"] < 1e-6


def test_loudness_off_restores_original_levels(result):
    assert result["normOff"] < 1e-6


def test_no_clicks_on_any_change(result):
    # every mix change ramps over 32 samples: no sample-to-sample jump much beyond the signals' own
    assert result["maxJump"] < 2.5 * 2 * result["maxSignalStep"] + 0.05  # x2: the stage-1 gain of 2
