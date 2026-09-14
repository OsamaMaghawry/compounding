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
reelforge templates                     # see the ready-made looks
reelforge templates --preview           # ...as a picture, so you can pick by eye
reelforge auto raw.mp4 -t viral         # use one
reelforge auto raw.mp4 -t viral --review  # ...and check it before exporting
reelforge auto raw.mp4 --preview        # half resolution, for a quick look
reelforge captions raw.mp4 --srt        # Arabic subtitles only
reelforge fonts                         # Arabic fonts you can install
reelforge learn                         # what it has picked up from you so far
```

## Writing scripts

```bash
reelforge studio init                              # then edit the three files it makes
reelforge script "why most people never compound" -s SPX --since 2015-01-01
#   ... shoot it ...
reelforge auto clip.mp4 --script studio/scripts/001-....json
```

The studio holds who you are, how you sound and who is watching, as plain files you
edit. Scripts are written against those plus a **verified fact sheet** — and every
number in the finished script is audited back against your data, so an invented figure
gets flagged before you say it out loud.

Passing `--script` back to the editor is the point of keeping both halves in one tool:
the script becomes the transcription prior, so Arabic captions on footage you shot from
a script are markedly more accurate. See [docs/STUDIO.md](docs/STUDIO.md).

Writers: `claude` (best Arabic, needs an API key), `ollama` (fully local and free), or
`stub` (no model — gives you the empty structure). Pick per run with `--provider`.

## Market data

If you make market content, add your price history once and every script can cite real,
computed numbers:

```bash
reelforge market add spx.csv --symbol SPX --name "S&P 500" --currency USD
reelforge market facts SPX --since 2015-01-01 --monthly 100
reelforge market compare SPX GOLD
```

It reads what real exports actually contain — newest-first rows, `1.234,56` decimals,
Arabic headers and digits, ambiguous `15/03/2024` dates — and computes total return, CAGR,
drawdown, per-year returns and dollar-cost-averaging plans, giving you each fact phrased
for speech in both English and Arabic.

The rule it is built around: **arithmetic in Python, language in the model.** Nothing here
is generated, so nothing here can be hallucinated. See [docs/MARKET.md](docs/MARKET.md).

## Start here

```bash
reelforge templates --preview
```

That writes an image showing every template's captions so you can pick one by eye, then:

```bash
reelforge auto myclip.mp4 -t viral --review
```

| Template | |
|---|---|
| `viral` | Yellow box captions, tight cuts, constant movement. The default Reels look. |
| `bold` | Wide Alexandria with a pulse on every spoken word. Punchy, no boxes. |
| `clean` | Calm gold karaoke on white, no transitions. Good for teaching. |
| `word` | One huge word at a time. Highest attention - hooks and ads. |
| `elegant` | Full lines, soft blur transitions. Storytelling and long-form cutdowns. |
| `news` | Dark box, sober, minimal movement. Market and data content. |

Templates are just YAML in `templates/`. Copy one, change it, and it shows up in the list.

Useful flags: `--no-zoom`, `--no-broll`, `--no-cuts`, `--no-captions` to turn off a stage,
`--set zoom.max_factor=1.3` to override any setting, `--model small` for a faster/less
accurate transcript.

## Engagement effects

**Captions never get typed by you.** Your voice is transcribed automatically with per-word
timing; you only ever correct a word it mishears, and it remembers the correction.

*Caption styles* - `--caption-style` or `captions.style`:

| | |
|---|---|
| `karaoke` | The spoken word changes colour. Default. |
| `box` | The spoken word sits in a filled box. The CapCut look. |
| `pop` | The line pulses as each word lands. |
| `word` | One large word on screen at a time. |
| `plain` | Full lines, no per-word marking. |

*Important words stay marked* even when they are not being spoken - numbers, percentages
and a built-in list of Arabic emphasis words (`مجانا`, `أهم`, `احذر`, `سر`, ...). Add your own
with `captions.emphasis_words`.

*Transitions* on cuts - `--transitions` or `transitions.kind`:

| | |
|---|---|
| `auto` | Blur where the shot changed, flash where a long pause was cut, punch otherwise. Default. |
| `punch` | Quick zoom spike on the cut. |
| `flash` | Brief brightness lift. |
| `blur` | Short defocus. |
| `none` | Straight cuts. |

*Zoom* punches in on the phrases you emphasise and pulls out between them, chained so the
framing never snaps back.

*Fonts* - 14 popular Arabic families, all SIL Open Font License:

```bash
reelforge fonts                      # list, with what each is good for
reelforge fonts --install Changa     # or --install all
reelforge auto clip.mp4 --font Almarai
```

## What `auto` actually does

| Stage | What happens |
|---|---|
| Analyse | Silence intervals, an RMS energy envelope, and shot changes, straight from ffmpeg. ~1s for a 60s clip, cached by content hash. |
| Transcribe | faster-whisper with word-level timestamps, biased toward your learned vocabulary. Word boundaries are then snapped to local energy minima, which is what makes karaoke highlighting land on the beat. |
| Cut | Silences longer than a threshold are removed with a little padding either side. Everything downstream is re-timed onto the shortened timeline. |
| Zoom | Each phrase is scored for emphasis (loudness relative to the clip, word rate, position, whether it follows a cut). The strongest get a punch-in or pull-out. Moves are chained so the framing never snaps back. |
| B-roll | Phrases are matched against your own clip library by keyword and composited as a layer. |
| Caption | Words are grouped into short lines broken at natural pauses, styled per your template, then written as ASS and burned in with libass, which handles Arabic shaping and bidi properly. |
| Transition | A short punch, flash or blur is placed on cuts, sparsely enough that the edit does not feel like a template. |
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

## Where things live

| | |
|---|---|
| `studio/` | **Created on your machine** by `reelforge studio init`. Your background, voice, frameworks and scripts. Never committed, never uploaded. |
| `.reelforge/` | Also local. Cache, market database, learning history, render work files. |
| `studio_template/`, `frameworks/` | In this repo — the starting text `studio init` copies into your `studio/`. Edit these if you want different defaults for every new studio. |
| `templates/` | Video look presets (`viral`, `clean`, ...), in this repo. |

Nothing in `studio/` or `.reelforge/` leaves your computer. If you are browsing this
repo looking for `studio/background.md`, it is not here by design — run
`reelforge studio init` and it appears in your working folder.

## Layout

```
reelforge/
  analysis.py   silence, energy, shot detection    render.py    EDL -> ffmpeg filtergraph
  market.py     your price history and fact sheets  fonts.py     Arabic font catalog
  script.py     script writing and number auditing  llm.py       writing backends
  knowledge.py  your background, voice and frameworks
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
- No music beat-sync or SFX yet.
- Caption effects cannot reflow a line, so scaling a single word would grow it into its
  neighbour. `pop` pulses the whole line instead, and word emphasis is colour-only by
  default. `captions.emphasis_scale` raises it if you want the size change and accept
  the tighter spacing.
- Transcription has been verified against faster-whisper's API, but the model weights
  themselves were never downloaded during development. If your first run fails inside the
  ASR step, `reelforge doctor` and the error text will say why.

## Tests

```bash
python -m unittest discover -s tests -v
```

107 tests, no sample files or model downloads needed. The media tests build their own clip
with ffmpeg. The faster-whisper adapter is covered by integration tests that drive it with
the library's own `Segment`/`Word` types and assert every keyword argument we send is one
the installed version accepts — so a breaking change upstream fails the suite rather than
your first real render.
