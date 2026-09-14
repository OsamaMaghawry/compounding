# ReelForge

A local auto-editor for vertical short-form video. One command turns a raw talking-head
recording into a finished Reel: dead air cut, punch-ins on the emphatic moments, your own
b-roll dropped in on cue, and accurate Arabic karaoke captions burned in.

Everything runs on your machine. No upload, no server, no account, no subscription.

```bash
reelforge auto raw.mp4 --review
```

## Why this exists

CapCut and the rest are general-purpose and built around a timeline you drive by hand.
Two things they do not do well:

- **Arabic captions.** Shaping, right-to-left ordering and per-word karaoke timing are
  usually wrong or unavailable, and there is no way to teach the tool the names and terms
  you say in every video.
- **Your style.** Their "auto" presets are someone else's taste, and they never learn yours.

ReelForge does one job instead of all of them, and it keeps a local record of what you
accepted and what you rejected, so every edit makes the next one closer to what you'd have
done by hand.

## Install

```bash
# 1. ffmpeg  (must be built with libass - almost every distribution build is)
brew install ffmpeg           # macOS
sudo apt install ffmpeg       # Ubuntu/Debian
winget install Gyan.FFmpeg    # Windows

# 2. ReelForge
git clone <this repo> && cd reelforge
pip install -e .

# 3. Arabic fonts + a health check
reelforge setup

# 4. Speech recognition (skip and you get placeholder captions)
pip install -r requirements-asr.txt
```

`reelforge doctor` tells you what is missing at any point.

## Use it

```bash
reelforge auto raw.mp4                  # edit and export
reelforge auto raw.mp4 --review         # ...and open the review page first
reelforge auto raw.mp4 -p punchy        # faster pacing
reelforge auto raw.mp4 --preview        # half resolution, for a quick look
reelforge captions raw.mp4 --srt        # Arabic subtitles only
reelforge learn                         # what it has picked up from you so far
```

Useful flags: `--no-zoom`, `--no-broll`, `--no-cuts`, `--no-captions` to turn off a stage,
`--set zoom.max_factor=1.3` to override any setting, `--model small` for a faster/less
accurate transcript.

## What `auto` actually does

| Stage | What happens |
|---|---|
| Analyse | Silence intervals, an RMS energy envelope, and shot changes, straight from ffmpeg. ~1s for a 60s clip, cached by content hash. |
| Transcribe | faster-whisper with word-level timestamps, biased toward your learned vocabulary. Word boundaries are then snapped to local energy minima, which is what makes karaoke highlighting land on the beat. |
| Cut | Silences longer than a threshold are removed with a little padding either side. Everything downstream is re-timed onto the shortened timeline. |
| Zoom | Each phrase is scored for emphasis (loudness relative to the clip, word rate, position, whether it follows a cut). The strongest get a punch-in or pull-out. Moves are chained so the framing never snaps back. |
| B-roll | Phrases are matched against your own clip library by keyword and composited as a layer. |
| Caption | Words are grouped into short lines broken at natural pauses, then written as ASS and burned in with libass, which handles Arabic shaping and bidi properly. |
| Render | One ffmpeg pass: retime, reframe to 9:16, zoom curve, layers, captions, loudness normalisation to -14 LUFS. |

Every decision lands in an **EDL** — plain JSON in `.reelforge/runs/` with a stable id and an
`enabled` flag on each item. Read it, flip anything off, re-render. That file is also the
training signal: the difference between what was proposed and what you kept.

## How it gets better

`reelforge learn` shows the state of it. In short:

- **Vocabulary.** Every caption word you fix is stored. It comes back as an ASR bias prompt
  *and* a post-pass replacement, so a name you correct once stops being wrong.
- **Pacing.** Disable half the punch-ins and the threshold rises; keep them all and it drops.
  A handful of scalars converge on your taste within a few edits.
- **A zoom model.** After ~40 reviewed decisions, a small logistic regression trained on your
  own accept/reject choices replaces the hand-written emphasis score.
  `reelforge learn --retrain` shows which signals actually drive your choices.
- **B-roll preferences.** Assets you keep get promoted; ones you delete sink below threshold.

See [docs/TRAINING.md](docs/TRAINING.md) for the honest version of what does and does not
improve with use.

## Speed

Measured per 60 seconds of source. Analysis and transcription are cached, so everything
after the first pass is just the render.

| | GPU (RTX-class) | Modern laptop CPU |
|---|---|---|
| Analysis | ~1s | ~1-2s |
| Transcribe (`large-v3`) | ~10s | 2-5 min |
| Transcribe (`small`) | ~3s | ~40s |
| Render 1080x1920 | ~20s | ~40-60s |
| **Re-render after a tweak** | **~20s** | **~40s** |
| **Preview re-render** | **~6s** | **~12s** |

So: roughly a minute for the first pass on a GPU box, a few minutes on a laptop with
`--model small`, and a fast loop after that because nothing is recomputed. Straight
"seconds" happens on the preview loop and on captions-only runs with a GPU.

## B-roll library

Drop clips in `assets/broll/`. Filenames are the keywords, Arabic included:

```
assets/broll/الذكاء_الاصطناعي.mp4     matches الذكاء or الاصطناعي
assets/broll/فلوس.mp4                 matches فلوس / الفلوس
```

For more control add `assets/broll/library.yml` — see the README in that folder.

## Profiles

`profiles/*.yml` hold the numbers behind every decision. `default`, `punchy`, `calm`,
`captions_only` ship with it; copy one and edit to make your own. Anything in a profile can
also be set per run with `--set key.path=value`.

## Layout

```
reelforge/
  analysis.py   silence, energy, shot detection    render.py    EDL -> ffmpeg filtergraph
  speech.py     ASR backends, word timings         learn.py     feedback store, tuning, model
  arabic.py     normalisation, vocabulary          review.py    local review page
  captions.py   grouping, ASS/SRT writing          pipeline.py  orchestration
  brain.py      cuts, zooms, b-roll decisions      cli.py       command line
  edl.py        the edit format and retiming
```

## Limitations

- Reframing is centre-crop. Face tracking is not implemented yet — for off-centre framing
  use `--set reframe.mode=left|right` or `reframe.blur_background=true`.
- Transcription accuracy is Whisper's. It is strong on MSA, good on Egyptian and Gulf
  dialect, weaker on heavy dialect and noisy audio. The vocabulary loop is what closes
  the gap on your specific recurring terms.
- Music beat-sync, transitions and SFX are not implemented.
- Transcription has been verified against faster-whisper's API, but the model weights
  themselves were never downloaded during development. If your first run fails inside the
  ASR step, `reelforge doctor` and the error text will say why.

## Tests

```bash
python -m unittest discover -s tests -v
```

51 tests, no sample files or model downloads needed. The media tests build their own clip
with ffmpeg. The faster-whisper adapter is covered by integration tests that drive it with
the library's own `Segment`/`Word` types and assert every keyword argument we send is one
the installed version accepts — so a breaking change upstream fails the suite rather than
your first real render.
