"""Speech recognition with word-level timings.

The backend is pluggable on purpose. `faster-whisper` is the default because it
gives word timestamps and runs locally; `whispercpp` suits machines without a
Python ML stack; `stub` produces deterministic fake output so the rest of the
pipeline can be developed and tested with no model downloads at all.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .analysis import Analysis
from .arabic import VocabCorrector, clean_for_display


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    backend: str
    model: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for segment in self.segments:
            out.extend(segment.words)
        return out

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def low_confidence(self, threshold: float = 0.55) -> list[Word]:
        """Words worth eyeballing in review - the cheapest accuracy win there is."""
        return [w for w in self.words if w.prob < threshold]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end,
                 "words": [asdict(w) for w in s.words]}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(
            language=data.get("language", "ar"),
            backend=data.get("backend", "unknown"),
            model=data.get("model", ""),
            segments=[
                Segment(
                    text=s.get("text", ""), start=s.get("start", 0.0), end=s.get("end", 0.0),
                    words=[Word(**w) for w in s.get("words", [])],
                )
                for s in data.get("segments", [])
            ],
        )


class ASRUnavailable(RuntimeError):
    pass


# ------------------------------------------------------------------ backends

def _pick_device(requested: str) -> tuple[str, str]:
    """Resolve ('auto','auto') into a concrete (device, compute_type)."""
    device = requested
    if device == "auto":
        device = "cpu"
        try:
            import ctranslate2  # noqa: PLC0415
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return device, compute


def transcribe_faster_whisper(audio: Path, profile, *, prompt: str | None) -> Transcript:
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError as exc:
        raise ASRUnavailable(
            "faster-whisper is not installed. Run:\n"
            "  pip install -r requirements-asr.txt\n"
            "or pick another backend with --asr-backend."
        ) from exc

    device, auto_compute = _pick_device(profile.get("asr.device"))
    compute_type = profile.get("asr.compute_type")
    if compute_type == "auto":
        compute_type = auto_compute

    model_name = profile.get("asr.model")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if profile.get("asr.temperature_fallback") else 0.0
    segments_iter, _info = model.transcribe(
        str(audio),
        language=profile.get("asr.language") or None,
        task="transcribe",
        beam_size=int(profile.get("asr.beam_size")),
        word_timestamps=bool(profile.get("asr.word_timestamps")),
        vad_filter=bool(profile.get("asr.vad")),
        vad_parameters={"min_silence_duration_ms": int(profile.get("asr.vad_min_silence_ms"))},
        initial_prompt=prompt or None,
        condition_on_previous_text=bool(profile.get("asr.condition_on_previous_text")),
        temperature=temperature,
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(text=(w.word or "").strip(), start=float(w.start), end=float(w.end),
                 prob=float(getattr(w, "probability", 1.0) or 0.0))
            for w in (getattr(seg, "words", None) or [])
            if (w.word or "").strip() and w.start is not None and w.end is not None
        ]
        text = (seg.text or "").strip()
        if not text and not words:
            continue
        segments.append(Segment(text=text, start=float(seg.start), end=float(seg.end), words=words))

    return Transcript(language=profile.get("asr.language"), backend="faster-whisper",
                      model=model_name, segments=segments)


def transcribe_whispercpp(audio: Path, profile, *, prompt: str | None) -> Transcript:
    """Drive a whisper.cpp binary and read its word-level JSON output."""
    binary = shutil.which("whisper-cli") or shutil.which("whisper-cpp") or shutil.which("main")
    if not binary:
        raise ASRUnavailable("whisper.cpp binary not found (looked for whisper-cli/whisper-cpp/main).")

    model_path = Path(profile.get("asr.model"))
    if not model_path.exists():
        raise ASRUnavailable(
            f"whisper.cpp needs a .bin model path in asr.model, got '{model_path}'."
        )

    out_prefix = audio.with_suffix("")
    args = [binary, "-m", str(model_path), "-f", str(audio),
            "-l", profile.get("asr.language") or "auto",
            "-ml", "1", "-oj", "-of", str(out_prefix)]
    if prompt:
        args += ["--prompt", prompt]
    proc = subprocess.run(args, capture_output=True, text=True)
    json_path = Path(f"{out_prefix}.json")
    if proc.returncode != 0 or not json_path.exists():
        raise ASRUnavailable(f"whisper.cpp failed: {(proc.stderr or '').strip()[-400:]}")

    data = json.loads(json_path.read_text("utf-8"))
    segments: list[Segment] = []
    for item in data.get("transcription", []):
        offsets = item.get("offsets", {})
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", 0)) / 1000.0
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # -ml 1 makes each transcription item a single token, so rebuild words by gaps.
        segments.append(Segment(text=text, start=start, end=end,
                                words=[Word(text=text, start=start, end=end, prob=1.0)]))
    merged = _merge_token_segments(segments)
    return Transcript(language=profile.get("asr.language"), backend="whispercpp",
                      model=model_path.name, segments=merged)


def _merge_token_segments(segments: list[Segment], gap: float = 0.6) -> list[Segment]:
    """Group single-token whisper.cpp items into sentence-ish segments."""
    out: list[Segment] = []
    for seg in segments:
        if out and seg.start - out[-1].end <= gap:
            current = out[-1]
            current.words.extend(seg.words)
            current.end = seg.end
            current.text = " ".join(w.text for w in current.words)
        else:
            out.append(Segment(text=seg.text, start=seg.start, end=seg.end, words=list(seg.words)))
    return out


_STUB_WORDS = [
    "مرحبا", "بكم", "في", "فيديو", "اليوم", "سنتحدث", "عن", "الذكاء", "الاصطناعي",
    "وكيف", "يمكنك", "استخدامه", "لتوفير", "الوقت", "في", "المونتاج", "خلال", "دقائق",
    "تابع", "معي", "حتى", "النهاية", "لأن", "الجزء", "الأخير", "هو", "الأهم",
]


def transcribe_stub(audio: Path, profile, *, analysis: Analysis | None = None,
                    **_: object) -> Transcript:
    """Deterministic fake transcript over the detected speech regions.

    Not a toy for its own sake: it lets the cut/zoom/caption/render path be tested
    on any machine with no model download, and it is what the test suite runs on.
    """
    intervals = [i for i in (analysis.speech if analysis else []) if i.duration > 0.2]
    if not intervals:
        total = analysis.duration if analysis else 10.0
        from .analysis import Interval  # noqa: PLC0415 - avoid cycle at import time
        intervals = [Interval(0.0, total)]

    seed = int(hashlib.sha1(str(audio).encode()).hexdigest()[:8], 16)
    segments: list[Segment] = []
    cursor = 0
    for interval in intervals:
        count = max(2, int(interval.duration / 0.42))
        step = interval.duration / count
        words: list[Word] = []
        for index in range(count):
            text = _STUB_WORDS[(seed + cursor) % len(_STUB_WORDS)]
            cursor += 1
            start = interval.start + index * step
            words.append(Word(text=text, start=round(start, 3),
                              end=round(start + step * 0.86, 3),
                              prob=0.55 + ((seed + index) % 40) / 100.0))
        segments.append(Segment(text=" ".join(w.text for w in words),
                                start=interval.start, end=interval.end, words=words))
    return Transcript(language=profile.get("asr.language"), backend="stub",
                      model="stub", segments=segments)


BACKENDS = {
    "faster-whisper": transcribe_faster_whisper,
    "whispercpp": transcribe_whispercpp,
    "stub": transcribe_stub,
}


def available_backend(requested: str) -> str:
    """Resolve 'auto' to the best backend actually installed."""
    if requested != "auto":
        return requested
    try:
        import faster_whisper  # noqa: F401,PLC0415
        return "faster-whisper"
    except ImportError:
        pass
    if shutil.which("whisper-cli") or shutil.which("whisper-cpp"):
        return "whispercpp"
    return "stub"


# ------------------------------------------------------------------ pipeline

def snap_words_to_energy(words: list[Word], analysis: Analysis, *,
                         max_shift: float = 0.12) -> list[Word]:
    """Nudge word boundaries toward nearby quiet points.

    Whisper's word timings drift by 100-200 ms, which is exactly enough to make
    karaoke highlighting feel out of sync. Snapping each boundary to the local
    energy minimum inside a small window fixes most of it for free.
    """
    if not analysis.energy or not words:
        return words
    hop = analysis.energy_hop
    span = max(1, int(max_shift / hop))

    def best_boundary(t: float) -> float:
        center = int(t / hop)
        lo, hi = max(0, center - span), min(len(analysis.energy), center + span + 1)
        if hi <= lo:
            return t
        window = analysis.energy[lo:hi]
        quietest = min(range(len(window)), key=lambda i: window[i])
        return round((lo + quietest) * hop, 3)

    snapped: list[Word] = []
    for index, word in enumerate(words):
        start = best_boundary(word.start)
        end = best_boundary(word.end)
        if end - start < 0.06:                      # snapping collapsed the word
            start, end = word.start, word.end
        if snapped and start < snapped[-1].end:     # never let words overlap
            start = snapped[-1].end
        if end <= start:
            end = max(word.end, start + 0.06)
        snapped.append(Word(text=word.text, start=start, end=end, prob=word.prob))
    return snapped


def transcribe(audio: Path, profile, *, analysis: Analysis | None = None,
               corrector: VocabCorrector | None = None, cache_dir: Path | None = None,
               cache_key: str | None = None, refresh: bool = False,
               on_status=None) -> Transcript:
    """Transcribe, apply learned corrections, and tighten word timings."""
    backend_name = available_backend(profile.get("asr.backend"))
    handler = BACKENDS.get(backend_name)
    if handler is None:
        raise ASRUnavailable(f"unknown ASR backend '{backend_name}' "
                             f"(choose from {', '.join(BACKENDS)})")

    cache_file = None
    if cache_dir and cache_key:
        digest = hashlib.sha1(
            f"{cache_key}|{backend_name}|{profile.get('asr.model')}|"
            f"{profile.get('asr.language')}|{len(corrector or {})}".encode()
        ).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"transcript-{digest}.json"
        if cache_file.exists() and not refresh:
            try:
                return Transcript.from_dict(json.loads(cache_file.read_text("utf-8")))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    prompt_parts = [p for p in [profile.get("asr.initial_prompt"),
                                corrector.prompt_terms(int(profile.get("learning.max_vocab_prompt")))
                                if corrector else ""] if p]
    prompt = " ".join(prompt_parts) or None

    if on_status:
        on_status(f"transcribing with {backend_name} ({profile.get('asr.model')})")

    if backend_name == "stub":
        transcript = transcribe_stub(audio, profile, analysis=analysis)
    else:
        transcript = handler(audio, profile, prompt=prompt)

    strip_marks = bool(profile.get("captions.strip_diacritics"))
    normalize_punct = bool(profile.get("captions.normalize_punctuation"))
    arabic_percent = bool(profile.get("captions.arabic_percent", False))
    for segment in transcript.segments:
        for word in segment.words:
            text = clean_for_display(word.text, strip_marks=strip_marks,
                                     normalize_punctuation=normalize_punct,
                                     arabic_percent=arabic_percent)
            word.text = corrector.correct_word(text) if corrector else text
        segment.text = clean_for_display(segment.text, strip_marks=strip_marks,
                                         normalize_punctuation=normalize_punct,
                                         arabic_percent=arabic_percent)
        if corrector:
            segment.text = corrector.correct_text(segment.text)
        if analysis is not None and segment.words:
            segment.words = snap_words_to_energy(segment.words, analysis)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
    return transcript
