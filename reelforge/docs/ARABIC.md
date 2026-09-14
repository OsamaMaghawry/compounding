# Getting Arabic captions right

Arabic subtitles break in three distinct ways. Only the third is about accuracy of the
transcript; the first two are rendering bugs that make correct text look broken.

## 1. Shaping

Arabic letters change form depending on position — `ع` is `عـ`, `ـعـ`, `ـع` or `ع`. Tools that
draw text glyph by glyph produce disconnected letterforms.

ReelForge renders through **libass**, which shapes with HarfBuzz. Check your ffmpeg has it:

```bash
reelforge doctor          # wants libass, libharfbuzz, libfribidi, libfreetype
```

## 2. Direction

Arabic runs right to left, and a line mixing Arabic with numbers or Latin words needs the
Unicode bidi algorithm. The classic broken workaround is to reverse the string manually.
Never do that: it produces text that looks right in a preview and is corrupt everywhere else.

ReelForge writes text in **logical (spoken) order** and lets FriBidi order it. `نسبة النجاح 95%
خلال 2026` comes out correct, numbers included. `arabic.clean_for_display` explicitly does not
reverse, and a test asserts that.

## 3. Accuracy

This is the real work.

**Model choice.** `large-v3` is meaningfully better on Arabic than `medium`, especially on
dialect. Use it if you have a GPU. On CPU, `small` is the practical floor for usable Arabic;
`tiny`/`base` are not worth burning in.

```bash
reelforge auto clip.mp4 --model large-v3      # GPU
reelforge auto clip.mp4 --model small         # CPU
```

**Vocabulary biasing.** The single biggest lever. Seed your recurring terms once:

```bash
reelforge learn --add-vocab "اسامه=أسامة" --add-vocab "انستجرام=إنستغرام"
```

They then bias decoding and get applied as a post-pass on every run. Every fix you make in
the review page is added automatically.

**Word timing.** Whisper's word timestamps drift 100-200 ms — enough to make karaoke
highlighting feel off. Each boundary is snapped to the nearest local energy minimum within
120 ms, which lands the highlight on the word. If your audio is very compressed (heavy
limiting, loud music bed) this helps less; lower `captions.karaoke` to `false` and use plain
lines rather than living with a highlight that lags.

**Hallucination on silence.** Whisper invents text over long silences. Two defences are on by
default: VAD filtering (`asr.vad`) and `condition_on_previous_text: false`, which stops the
model looping a phrase it just produced.

**Diacritics.** Whisper mostly outputs undiacritised text, which is what you want for
captions. If your source has them and you don't want them burned in, set
`captions.strip_diacritics: true`.

## Fonts

Arabic needs a font with Arabic coverage — most default sans fonts do not have it, and libass
will silently draw boxes. `reelforge setup` fetches Cairo, Tajawal and Almarai.

```bash
reelforge auto clip.mp4 --font Tajawal
```

```bash
reelforge fonts                      # the full list and what each suits
reelforge fonts --install all
```

Cairo is the safe default: clean, heavy enough for captions, good coverage. Almarai
ExtraBold reads well at large caption sizes. Alexandria is the best pick for
one-word-at-a-time captions.

One caveat found by testing rather than assumed: **Changa's `%` glyph crowds the word next
to it**, so avoid it if you say percentages a lot. Cairo, Almarai and Alexandria all render
percentages cleanly. If you prefer proper Arabic typography, `captions.arabic_percent: true`
renders `%` as `٪`.

Set `captions.font` in your template to make a choice permanent.

## Style

Reels captions want fewer words on screen than film subtitles: 3-4 at 90-100px, positioned
around 70% down the frame, clear of the platform UI. `captions.safe_area` enforces the bottom
margin. The defaults are tuned for 1080x1920; if you change the output size everything scales
with it.
