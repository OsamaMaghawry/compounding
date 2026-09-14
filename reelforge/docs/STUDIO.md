# The studio: writing scripts

The editor half of ReelForge turns footage into a Reel. The studio half decides what
to shoot. They are connected in a way that matters — see "The loop" at the end.

## Setup

```bash
reelforge studio init
```

That creates three files. **Edit them** — the writer reads them verbatim:

| File | What goes in it |
|---|---|
| `studio/background.md` | Who you are, what you cover, why anyone should listen, what you never claim |
| `studio/voice.md` | Language and dialect, tone, sentence shape, words you use and avoid |
| `studio/audience.md` | Who is watching, what they already believe, what they fear |

They ship as empty prompts, not invented biography. A plausible-sounding fake
credential is worse than a blank, because a blank gets noticed and a fake gets
published.

## Writing

```bash
reelforge script "why most people never compound" -s SPX --since 2015-01-01 --monthly 100
```

You get a script broken into beats, each with its spoken line, a suggested on-screen
caption, b-roll keywords and a duration — saved as both `.json` (for the editor) and
`.md` (for you to read and correct).

Useful flags: `-f` picks a framework, `--seconds` sets the target length, `--lang en`
switches language, `-d` adds a one-off steer ("make the hook angrier").

## Frameworks

A framework is the structure of the video — the beats, in order, with the job each
one does. Five ship by default:

| | |
|---|---|
| `number-story` | Lead with one number that stops the scroll, then explain it. The default. |
| `myth-bust` | State a belief the audience holds, disprove it, replace it. |
| `mistake-fix` | Name an expensive mistake, quantify it, show the fix. |
| `compare` | Two options, same window, one verdict. |
| `story-lesson` | A short personal story that earns a general principle. |

They are YAML in `studio/frameworks/`. Edit them, or add your own — a file dropped in
that folder shows up in `reelforge script --list-frameworks` immediately, and a studio
copy overrides the built-in of the same name.

**These are a starting point, not a recommendation about your content.** Replace them
with how you actually structure a video.

## Numbers

The writing model never computes a figure. `market.py` computes them, they are handed
to the model as fixed text it may quote but not recalculate, and then **every number in
the finished script is audited back against the data**:

```
  CHECK THESE - not traceable to your data: 91.4, 5000
```

A figure counts as verified when some real value rounds to it at the precision it was
quoted at — so a real 5.66% may be said as "5.7%", but a real 99.0 does not excuse a
fabricated 99.9. The audit is deliberately strict: a false flag costs you a glance, a
missed one costs your credibility.

If you pass no market data, *any* statistic in the output gets flagged — which is the
correct behaviour, because nothing was there to support it.

## Choosing a writer

```bash
reelforge studio            # shows which writers are ready
```

| | |
|---|---|
| `claude` | Best Arabic by a wide margin. Needs `pip install anthropic` and `ANTHROPIC_API_KEY` (or `ant auth login`). A few cents per script. |
| `ollama` | Fully local and free. Weaker Arabic, needs a decent machine. `ollama serve`, then `ollama pull qwen2.5:14b`. |
| `stub` | No model. Produces the empty structure with your facts placed in it. |

Pick per run with `--provider`. The stub deliberately writes *nothing* rather than
filler — an obviously blank script is safer than one that reads well enough to be
published by mistake.

## The loop

```bash
reelforge script "..." -s SPX          # writes studio/scripts/001-....json
#   ... you shoot it ...
reelforge auto clip.mp4 --script studio/scripts/001-....json
```

Passing `--script` is not bookkeeping. The script becomes the **transcription prior**:
Whisper is told roughly what was said, and the distinctive words — your names, tickers,
figures — are seeded into the correction map. Arabic transcription on footage shot from
a script is markedly more accurate than on footage transcribed cold, and that accuracy
is free, because you already wrote the words.

That is the payoff for keeping writing and editing in one tool rather than two.
