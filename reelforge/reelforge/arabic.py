"""Arabic text handling: normalisation, matching, and correction.

Two different "normal forms" live here and they must not be confused:

* `normalize_for_match` is lossy and only ever used for comparing words
  (b-roll keywords, learned corrections, dedupe).
* `clean_for_display` is what actually gets burned into the video.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanween, shadda, sukun, superscript alef, and the Quranic marks.
DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ࣓-ࣿ]")
TATWEEL = "ـ"
ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
PUNCT_STRIP = re.compile(r"[^\w؀-ۿ\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")

# Latin punctuation that should become its Arabic counterpart in burned captions.
PUNCT_MAP = {
    ",": "،",
    ";": "؛",
    "?": "؟",
}

# Presentation forms occasionally emitted by OCR/ASR pipelines; fold them back
# to plain letters so libass can do its own shaping.
_PRESENTATION_FOLD = str.maketrans({"ﻻ": "لا", "ﻷ": "لأ", "ﻹ": "لإ", "ﻵ": "لآ"})


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RANGE.search(text or ""))


def arabic_ratio(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ARABIC_RANGE.match(c)) / len(letters)


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", text or "")


def strip_tatweel(text: str) -> str:
    return (text or "").replace(TATWEEL, "")


def normalize_for_match(text: str) -> str:
    """Aggressive fold used only for comparison, never for display.

    Unifies alef/ya/ta-marbuta variants, drops diacritics, tatweel, punctuation
    and case, so `الذّكاء` and `الذكاء` compare equal.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_PRESENTATION_FOLD)
    text = strip_diacritics(strip_tatweel(text))
    text = (text
            .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
            .replace("ى", "ي").replace("ئ", "ي")
            .replace("ة", "ه")
            .replace("ؤ", "و"))
    text = PUNCT_STRIP.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip().lower()


def clean_for_display(text: str, *, strip_marks: bool = False,
                      normalize_punctuation: bool = True) -> str:
    """Tidy a transcript word/line for burning into the frame."""
    text = unicodedata.normalize("NFKC", text or "").strip()
    text = text.translate(_PRESENTATION_FOLD)
    text = strip_tatweel(text)
    if strip_marks:
        text = strip_diacritics(text)
    if normalize_punctuation and is_arabic(text):
        for latin, arabic in PUNCT_MAP.items():
            text = text.replace(latin, arabic)
    # ASS treats a literal newline as a line break directive; never emit one by accident.
    text = text.replace("\n", " ").replace("\r", " ")
    return WHITESPACE.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return [t for t in WHITESPACE.split(normalize_for_match(text)) if t]


def similarity(left: str, right: str) -> float:
    """Cheap 0..1 similarity over normalised forms (no external deps)."""
    a, b = normalize_for_match(left), normalize_for_match(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.86
    # token overlap, then character bigram overlap as a tiebreaker
    ta, tb = set(a.split()), set(b.split())
    if ta & tb:
        return 0.55 + 0.3 * (len(ta & tb) / max(len(ta), len(tb)))
    bigrams_a = {a[i:i + 2] for i in range(len(a) - 1)}
    bigrams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    if not bigrams_a or not bigrams_b:
        return 0.0
    return 0.5 * len(bigrams_a & bigrams_b) / max(len(bigrams_a), len(bigrams_b))


class VocabCorrector:
    """Applies learned `wrong -> right` word fixes to ASR output.

    This is the component that makes captions get better every time you fix a
    name the model keeps mishearing: one correction, applied forever after.
    """

    def __init__(self, pairs: dict[str, str] | None = None, *, min_similarity: float = 1.0):
        self.exact: dict[str, str] = {}
        self.min_similarity = min_similarity
        for wrong, right in (pairs or {}).items():
            key = normalize_for_match(wrong)
            if key:
                self.exact[key] = right

    def __len__(self) -> int:
        return len(self.exact)

    def correct_word(self, word: str) -> str:
        if not self.exact:
            return word
        key = normalize_for_match(word)
        if key in self.exact:
            return self.exact[key]
        if self.min_similarity < 1.0:
            best, score = None, 0.0
            for candidate, replacement in self.exact.items():
                value = similarity(key, candidate)
                if value > score:
                    best, score = replacement, value
            if best and score >= self.min_similarity:
                return best
        return word

    def correct_text(self, text: str) -> str:
        if not self.exact or not text:
            return text
        return " ".join(self.correct_word(part) for part in text.split(" "))

    def prompt_terms(self, limit: int = 220) -> str:
        """Build a Whisper `initial_prompt` that biases toward your vocabulary."""
        terms: list[str] = []
        length = 0
        for right in self.exact.values():
            if right in terms:
                continue
            if length + len(right) + 2 > limit:
                break
            terms.append(right)
            length += len(right) + 2
        return "، ".join(terms)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!؟?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]
