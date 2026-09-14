# Architecture

```
video ──> analysis.py ──┐
                        ├──> brain.py ──> EDL (json) ──> render.py ──> reel.mp4
        speech.py ──────┘                   │  ▲
                                            │  │
                                  review.py │  │ learn.py
                                            ▼  │
                                    your corrections
```

## The EDL is the whole design

Everything the system decides becomes a row in a JSON file with a stable id and an `enabled`
flag, before anything is rendered. That single choice buys:

- **Inspectability.** The edit is readable before you spend a render on it (`--plan-only`).
- **Cheap iteration.** Toggling a decision costs one ffmpeg pass, not a re-analysis.
- **Training data for free.** Proposed EDL vs. the one you kept *is* the labelled dataset.
  No annotation step, no separate feedback UI to maintain.

## Time discipline

The one invariant that keeps this from falling apart: `src_*` fields are timestamps in the
original file, `out_*` fields are timestamps in the finished video. Once silences are cut
these diverge, and mixing them puts captions on the wrong words.

So: **only `cuts` knows about source time.** Every effect and caption is stored in output
time, remapped once through `Timeline.to_out()` when the EDL is built. `Timeline.to_out` also
handles the edge case of a word that straddles a cut, by clamping forward to the next
surviving frame.

## Module boundaries

| Module | Owns | Depends on |
|---|---|---|
| `ffmpeg.py` | Running the binaries, probing | - |
| `analysis.py` | Silence, energy envelope, shot changes, caching | ffmpeg |
| `speech.py` | ASR backends, word timings, energy snapping | analysis, arabic |
| `arabic.py` | Normalisation, similarity, vocabulary correction | - |
| `captions.py` | Word grouping, ASS/SRT writing | arabic, speech |
| `broll.py` | Local asset library and keyword matching | arabic |
| `brain.py` | Cut/zoom/overlay decisions | analysis, speech, captions, broll, edl |
| `edl.py` | The edit format and the timeline mapping | captions |
| `render.py` | Filtergraph construction and execution | edl, captions, ffmpeg |
| `learn.py` | Feedback store, parameter tuning, zoom model | edl, arabic |
| `pipeline.py` | Orchestration | all of the above |
| `review.py` / `cli.py` | Interfaces | pipeline |

`brain.py` never touches ffmpeg and `render.py` never makes a decision. That separation is
what lets the tests cover the decision logic without rendering anything.

## The render pass

One ffmpeg invocation:

```
trim/atrim per cut ──> concat ──> fps/setsar
  ──> scale+crop to an oversized 9:16 canvas     (zoom headroom, so punch-ins stay sharp)
  ──> zoompan with a piecewise-continuous curve
  ──> overlay per b-roll layer (enable=between(t,..))
  ──> ass (libass burns the captions)
  ──> format=yuv420p
audio: atrim/concat ──> optional ducking ──> loudnorm
```

The filtergraph is written to a file and passed with `-filter_complex_script`, so a long edit
never hits an argv limit.

**The zoom curve** is the subtle part. Moves are chained — each starts at the factor the
previous one ended on — and the factor holds between them. The generated expression is a
nested `if` over `in_time` with smoothstep easing, which makes it continuous: without this
the framing snaps back to 1.0 between punch-ins, which reads as a glitch.

**Zoom headroom** means the canvas is rendered `output x max_zoom_factor` before the zoom
crops into it, so a 1.22x punch-in on a 1080x1920 output is still sampling real pixels rather
than upscaling.

## Backends are pluggable on purpose

`speech.BACKENDS` maps a name to a function. `faster-whisper` is the default,
`whispercpp` suits machines without a Python ML stack, and `stub` produces deterministic fake
output so the entire pipeline — cuts, zooms, captions, render — can be developed and tested
with no model download. The test suite runs on `stub`, which is why it needs nothing but
ffmpeg.

Adding a backend (a local Arabic ASR service, say) means writing one function that returns a
`Transcript` and registering it.

## Extending it

- **A new effect**: add a dataclass in `edl.py`, a planner in `brain.py`, a filter chain in
  `render.py`, and a toggle row in `review.py`. Everything else — learning, persistence,
  review — follows from the EDL contract.
- **Face-aware reframing**: the honest next step. Detect a face track, store it as a
  per-time crop centre in the EDL, and use it for `zoompan`'s `x`/`y` instead of the current
  centre expressions. Nothing else changes.
