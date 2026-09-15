# How this actually gets better (and what won't)

The goal is "it learns my style and gets fast". Worth being precise about which parts of
that are real, because the obvious interpretation — fine-tune a model on my videos — is the
one that does not work.

## What does not work

**Training a model on your finished Reels.** There is no model that maps "raw footage" to
"edited video" that you could fine-tune on a few hundred examples. The output space is
enormous, the supervision is weak, and you would need tens of thousands of paired examples.
Anyone promising this is describing a research programme, not a tool.

**Fine-tuning Whisper on your voice, as a first move.** It is possible, and covered below,
but it is the *last* thing to reach for. It costs a GPU, hours per run, and a few hundred
corrected sentences — and 90% of the accuracy you want from it is available immediately and
for free from vocabulary biasing.

## What does work

The insight is that the decisions are **low-dimensional**. "Where do I punch in" is not an
open-ended generative problem; it is a binary choice over a few numeric features. That is
learnable from tens of examples, not thousands.

### 1. Vocabulary — immediate, biggest win

Whisper mishears the same words every time: your name, your brand, the people you mention,
technical terms, dialect words. Each one you fix in review is stored as
`normalised wrong -> correct spelling` and applied two ways on every later run:

- as an `initial_prompt` that biases decoding toward those terms,
- as a post-pass replacement that catches whatever the bias missed.

Fix a name once, it is right forever. You can also seed it directly:

```bash
reelforge learn --add-vocab "اسامه=أسامة" --add-vocab "كومباوندنج=Compounding"
```

Ten minutes of seeding your recurring terms is the single highest-value thing you can do for
caption accuracy.

### 2. Pacing parameters — converges in a handful of edits

Five scalars are tuned from your behaviour with an exponential moving average:
`zoom.score_threshold`, `zoom.rate_per_min`, `broll.max_per_min`, `broll.min_score`,
`captions.max_words`.

The rule is just: the rate you *accept* is the rate it should have *proposed*. Disable four
of six punch-ins and the threshold rises. Keep them all and it falls. Bounds in
`learn.TUNABLE` stop any runaway. Three or four reviewed edits is enough to feel it.

### 3. A zoom model — takes over at ~40 decisions

Every proposed move is stored with its features and whether you kept it:

| feature | meaning |
|---|---|
| `rel_energy` | loudness of the phrase vs the clip average |
| `rel_peak` | peak loudness vs the clip average |
| `duration` | phrase length |
| `word_rate` | words per second — how fast you're delivering |
| `after_cut` | does the phrase start right after a jump cut |
| `position` | where in the video (hooks matter more) |
| `since_last_zoom` | seconds since the previous move |

Once there are enough labelled examples with both outcomes, a logistic regression trains on
them (pure Python, milliseconds) and replaces the hand-written emphasis score. Two or three
reviewed videos gets you there.

```bash
reelforge learn --retrain
```

It prints the learned weights, so you can see what actually drives your choices — for
instance that you punch in on *fast delivery* rather than on *volume*.

### 4. B-roll preferences

Keep an asset for a keyword and its weight rises; delete it and the weight falls until it
stops matching. Purely local, per asset-keyword pair.

### 5. Fine-tuning Whisper — only when the above is exhausted

If you still have systematic errors after seeding vocabulary, your corrected captions are
already a labelled dataset: audio segment + verified Arabic text, accumulating in
`.reelforge/history.db`. At a few hundred corrected sentences, a LoRA fine-tune of
`whisper-small` on your voice and dialect is worth it. Export with a short script over the
`runs` table, train with `peft` on a rented GPU, then point `asr.model` at the result.

This is the only component where "training a model" is the right description, and it is
optional.

## Where the speed comes from

Not from the model getting faster — from not recomputing:

- Analysis and transcript are cached by content hash. Re-render after a tweak skips both.
- The preview path renders at half resolution with `ultrafast`.
- The EDL is separate from the render, so changing your mind costs one ffmpeg pass, not a
  re-analysis.
- Better parameters mean fewer review rounds, which is the real time cost. A first pass that
  needs no correction is the fastest possible edit.

## Checking it is working

```bash
reelforge learn
```

Shows runs, reviewed count, accept rate per decision type, vocabulary size, and the model's
sample count and training accuracy. If the accept rate for zooms is climbing across runs,
it is learning your taste. If vocabulary is growing and captions still need the same fixes,
something is wrong — open an issue with the pair that keeps failing.

## Your data

All of it is `.reelforge/` next to where you run the command: SQLite plus JSON. Nothing
leaves the machine. Delete the folder and it forgets everything.
