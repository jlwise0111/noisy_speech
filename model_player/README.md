# Model player

Load one recording, run it through **every trained model at once**, and flip between the outputs while
it plays, like soloing tracks in Pro Tools.

```bash
make player          # from noisy_speech/: build, start, open http://localhost:8502
make player-down     # stop it
make test-player     # tests, inside the image
```

The dataset picker needs `DATASET` (or `VALENTINI_DIR`, or a `local.mk`), as for the rest of the project.
Without it, the player still works with uploads.

## What it does

- **Source:** upload any `.wav` file. It can be any sample rate, mono or stereo, up to 60 s. You can also
  pick a file from the Valentini test set, train 28spk or train 56spk. Everything is converted to 16 kHz
  mono with the project's own loader (`speechprep.load_audio`). A training file gets a note saying which
  models may have seen it during training.
- **Tracks:**
  - Clean: dataset files only.
  - Noisy input.
  - Ridge 28spk / 56spk / both.
  - Wiener min-stats and Wiener leading.
  - U-Net 28spk / 56spk / both.
  - Oracle: dataset files only, because it needs the clean recording.

  Every model is found in its run folder (`*/outputs/*`) and run with the same code that trained and
  scored it. A missing run is skipped with a note.
- **Processing:** the models are loaded once. They run in parallel threads, and the results for the
  last 8 inputs are cached. Every output is forced to exactly the input's length.
- **Lock-step playback:** a single AudioWorklet holds all the tracks and reads them at one shared
  position, so they cannot drift apart. The browser's audio runs at 16 kHz where it can, so no
  resampling is needed. Otherwise the shared position steps by `16000 / rate` and every track is
  interpolated at that same point.
- **XOR solo:** exactly one track is audible at a time. Switching tracks crossfades over
  `round(0.002 × rate)` samples (32 at 16 kHz) with equal-power gains, `cos(πu/2)` out and `sin(πu/2)`
  in. If you switch again during a fade, a new fade starts from whatever mix is audible at that moment.
- **Diff mode** (button or `D`): a two-slot tracker, so never more than two tracks.
  - **Slots:** *new* (marked +) and *old* (marked Ø, polarity-inverted). You hear new minus old:
    exactly what differs between the two. Identical tracks null to silence.
  - **Picking:** click a track (or press its key) to make it the new track. It replaces the previous
    new one, so nothing has to be unclicked. Right-click a track (or ⇧ + its key) to make it the old
    track. Picking the track that already holds the other slot swaps the two.
  - **Entering and leaving:** the soloed track starts as the old one. You hear nothing until both slots
    are filled. Leaving diff mode solos the new track.
  - **Look:** the selected buttons, panel labels and levels rows turn blue-grey. `↑`/`↓` step the new
    track. The hint next to the Diff button is short and fixed-width (full text in its tooltip), so
    the control bar never grows.
  - **Crossfade:** every change of mix uses the same 2 ms crossfade.
- **Match loudness** (switch in the player, or `L`; starts on and is remembered by the browser): this
  is the **first stage of playback**. The chain for each track is loudness gain, then the solo/diff
  mixer, so solo and diff always compare tracks that are already loudness-matched.
  - The gains come from the diagnostic viewer's BS.1770 code: every track is brought to −20 LUFS, and
    all tracks are lowered together if needed so no peak goes above −1 dBFS.
  - Python computes the gains in `mplayer/loudness.py`; the tracks themselves are sent at their
    original level.
  - Flipping the switch ramps the gains over 2 ms without reloading the page, so the current solo or
    diff selection is kept.
  - The spectrograms, waveforms and levels panel change with the switch.
- **Display:** the diagnostic viewer's stacked spectrograms (Ember colormap, log frequency), with the
  blue waveform drawn on top.

### Controls

| Key or action | What it does |
|---|---|
| `1`–`9`, `0` | Solo a track (in diff mode: set the new track; with `⇧`: set the old track) |
| Right-click a track button or levels row | In diff mode: set the old track (Ø) |
| `↑` / `↓` | Previous / next track (in diff mode: step the new track) |
| `D` | Diff mode on / off |
| `L` | Match loudness on / off |
| `space` | Play / pause |
| `Home` | Back to the start |
| `+` / `−` | Zoom |
| Click a panel label or a levels row | Solo that track |
| Click the plot | Seek |

Open `http://localhost:8502/?debug` to log each crossfade's length and its equal-power error in the
browser console.

## Layout

| Path | What it is |
|---|---|
| `app.py` | Streamlit UI |
| `mplayer/engine.py` | Model registry, loading, parallel processing |
| `mplayer/source.py` | Upload decoding, dataset splits |
| `mplayer/component.py` | Builds the player iframe; reuses `viewer.figures` and `viewer.player` |
| `mplayer/player.html` | Transport, levels, and the AudioWorklet mixer |
| `Dockerfile`, `Dockerfile.dockerignore` | Build context is `noisy_speech/`, so the image gets the project's packages: `speechprep`, `unet`, `lrb`, `wiener` and `viewer` |
| `docker-compose.yml` | Mounts the project folder (trained runs) at `/models` and `$DATASET_DIR` at `/dataset`, both read-only |

To run it without Docker, use an environment with torch, streamlit and plotly, and run
`streamlit run app.py --server.enableStaticServing=true` from this folder. The player loads
`static/plotly.min.js`, which the Dockerfile writes; to create it yourself, run
`python -c "from plotly.offline import get_plotlyjs; open('static/plotly.min.js','w').write(get_plotlyjs())"`.
