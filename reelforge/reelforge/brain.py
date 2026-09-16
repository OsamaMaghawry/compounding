"""The editor brain: analysis + transcript in, EDL out.

Deliberately rule-first. Rules are legible, debuggable and good enough on day one;
the learned model in `learn.py` only takes over a decision once it has enough of
your own labelled examples to beat them. Nothing here downloads or calls anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import Analysis, Interval
from .broll import BrollLibrary
from .captions import CaptionLine, group_words
from .edl import EDL, Cut, Overlay, Timeline, Transition, Zoom, build_timeline
from .speech import Transcript, Word, drop_repeated_words, enforce_order


# ------------------------------------------------------------------- cutting

def _speech_spans(transcript: Transcript, analysis: Analysis, max_gap: float) -> list[Interval]:
    """Speech regions, preferring ASR word timings over raw silence detection."""
    words = [w for w in transcript.words if w.duration > 0] if transcript else []
    if words:
        spans: list[Interval] = []
        for word in sorted(words, key=lambda w: w.start):
            if spans and word.start - spans[-1].end <= max_gap:
                spans[-1].end = max(spans[-1].end, word.end)
            else:
                spans.append(Interval(word.start, word.end))
        return spans
    return [Interval(i.start, i.end) for i in analysis.speech]


def plan_cuts(analysis: Analysis, transcript: Transcript, profile) -> list[Cut]:
    """Decide which parts of the source survive."""
    duration = analysis.duration
    if not profile.get("cuts.enabled") or not analysis.has_audio:
        return build_timeline([(0.0, duration)])

    max_gap = float(profile.get("cuts.max_gap_keep"))
    pad_before = float(profile.get("cuts.pad_before"))
    pad_after = float(profile.get("cuts.pad_after"))
    min_segment = float(profile.get("cuts.min_segment"))
    keep_head = float(profile.get("cuts.keep_head"))

    spans = _speech_spans(transcript, analysis, max_gap)
    if not spans:
        return build_timeline([(0.0, duration)])

    padded: list[list[float]] = []
    if keep_head > 0:
        padded.append([0.0, min(keep_head, duration)])
    for span in spans:
        start = max(0.0, span.start - pad_before)
        end = min(duration, span.end + pad_after)
        if padded and start <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], end)
        else:
            padded.append([start, end])

    keep = [(s, e) for s, e in padded if e - s >= min_segment]
    if not keep:
        keep = [(0.0, duration)]
    return build_timeline(keep)


# -------------------------------------------------------------------- zooming

@dataclass
class Beat:
    """A speech phrase in output time - the unit a zoom is attached to."""
    start: float
    end: float
    words: int
    mean_db: float
    peak_db: float
    after_cut: bool
    position: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def features(self, clip_mean_db: float, since_last_zoom: float) -> dict:
        return {
            "rel_energy": round(self.mean_db - clip_mean_db, 2),
            "rel_peak": round(self.peak_db - clip_mean_db, 2),
            "duration": round(self.duration, 2),
            "word_rate": round(self.words / self.duration, 2) if self.duration > 0.1 else 0.0,
            "after_cut": 1.0 if self.after_cut else 0.0,
            "position": round(self.position, 3),
            "since_last_zoom": round(min(since_last_zoom, 30.0), 2),
        }


def _beats(lines: list[CaptionLine], analysis: Analysis, timeline: Timeline,
           out_duration: float) -> list[Beat]:
    boundaries = timeline.cut_boundaries()
    beats: list[Beat] = []
    for line in lines:
        src_start = timeline.to_src(line.start)
        src_end = timeline.to_src(line.end)
        if src_start is None or src_end is None:
            mean_db, peak_db = analysis.loudness_mean, analysis.loudness_peak
        else:
            mean_db, peak_db = analysis.energy_range(src_start, src_end)
        beats.append(Beat(
            start=line.start, end=line.end, words=len(line.words),
            mean_db=mean_db, peak_db=peak_db,
            after_cut=any(abs(line.start - b) < 0.3 for b in boundaries),
            position=(line.start / out_duration) if out_duration > 0 else 0.0,
        ))
    return beats


def rule_score(features: dict, profile) -> float:
    """Hand-written emphasis score in 0..1 - the fallback until the model learns."""
    energy = _clamp((features["rel_energy"] + 3.0) / 9.0)      # louder than average
    peak = _clamp((features["rel_peak"] + 1.0) / 12.0)
    pace = _clamp((features["word_rate"] - 1.2) / 2.8)         # rapid delivery
    recency = _clamp(features["since_last_zoom"] / 6.0)        # avoid clustering
    hook = 1.0 - _clamp(features["position"] * 3.0)            # the opening matters most
    length = _clamp(features["duration"] / 2.0)

    score = (0.30 * energy + 0.18 * peak + 0.16 * pace
             + 0.16 * recency + 0.12 * hook + 0.08 * length)
    if features["after_cut"]:
        score += float(profile.get("zoom.cut_bias"))
    return _clamp(score)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def plan_zooms(lines: list[CaptionLine], analysis: Analysis, timeline: Timeline,
               profile, *, scorer=None) -> list[Zoom]:
    """Place punch-ins and pull-outs on the most emphatic phrases.

    Moves are chained: each starts at the factor the previous one ended on, so the
    framing never snaps back. Between moves the factor simply holds.
    """
    if not profile.get("zoom.enabled") or not lines:
        return []

    out_duration = timeline.duration
    beats = _beats(lines, analysis, timeline, out_duration)
    if not beats:
        return []

    min_gap = float(profile.get("zoom.min_gap"))
    min_duration = float(profile.get("zoom.min_duration"))
    max_duration = float(profile.get("zoom.max_duration"))
    threshold = float(profile.get("zoom.score_threshold"))
    max_factor = float(profile.get("zoom.max_factor"))
    min_factor = float(profile.get("zoom.min_factor"))
    budget = max(1, int(round(float(profile.get("zoom.rate_per_min")) * out_duration / 60.0)))

    # Score every beat first, then take the strongest that respect spacing.
    scored: list[tuple[float, Beat, dict]] = []
    last_zoom_at = -30.0
    for beat in beats:
        features = beat.features(analysis.loudness_mean, beat.start - last_zoom_at)
        score = scorer(features) if scorer else rule_score(features, profile)
        scored.append((score, beat, features))

    chosen: list[tuple[float, Beat, dict]] = []
    for candidate in sorted(scored, key=lambda item: item[0], reverse=True):
        if candidate[0] < threshold:
            break
        if len(chosen) >= budget:
            break
        if any(abs(candidate[1].start - other[1].start) < min_gap for other in chosen):
            continue
        chosen.append(candidate)

    chosen.sort(key=lambda item: item[1].start)

    if profile.get("zoom.hook_punch") and beats:
        hook_window = float(profile.get("zoom.hook_window"))
        if not any(item[1].start < hook_window for item in chosen):
            first = beats[0]
            features = first.features(analysis.loudness_mean, 30.0)
            chosen.insert(0, (max(threshold, 0.7), first, features))

    zooms: list[Zoom] = []
    current = 1.0
    for index, (score, beat, features) in enumerate(chosen):
        duration = _clamp(beat.duration, min_duration, max_duration)
        start = beat.start
        end = min(out_duration, start + duration)
        if end - start < 0.2:
            continue

        magnitude = min_factor + (max_factor - min_factor) * _clamp(score)
        if profile.get("zoom.alternate") and current > 1.005:
            target, kind = 1.0, "pull_out"          # already pushed in - come back
        else:
            target, kind = magnitude, "punch_in"

        zooms.append(Zoom(
            id=f"z{index + 1}", out_start=round(start, 3), out_end=round(end, 3),
            start_factor=round(current, 4), end_factor=round(target, 4),
            kind=kind, score=round(float(score), 4), features=features,
        ))
        current = target
    return zooms


# ------------------------------------------------------------------- overlays

def plan_overlays(lines: list[CaptionLine], library: BrollLibrary, profile,
                  *, out_duration: float, weights: dict[str, float] | None = None) -> list[Overlay]:
    """Drop b-roll over phrases whose words match an asset in your library."""
    if not profile.get("broll.enabled") or not len(library) or not lines:
        return []

    min_duration = float(profile.get("broll.min_duration"))
    max_duration = float(profile.get("broll.max_duration"))
    photo_duration = float(profile.get("broll.photo_duration", 2.0))
    cooldown = float(profile.get("broll.cooldown"))
    head_guard = float(profile.get("broll.head_guard"))
    min_score = float(profile.get("broll.min_score"))
    budget = max(1, int(round(float(profile.get("broll.max_per_min")) * out_duration / 60.0)))

    overlays: list[Overlay] = []
    last_end = -999.0
    for line in lines:
        if len(overlays) >= budget:
            break
        if line.start < head_guard:            # never cover the hook
            continue
        if line.start - last_end < cooldown:
            continue
        match = library.match(line.text, weights=weights, min_score=min_score)
        if not match:
            continue
        asset, keyword, score = match
        from .broll import hold_seconds  # noqa: PLC0415
        asked = hold_seconds(asset,
                             cutaway=_clamp(line.duration, min_duration, max_duration),
                             photo=photo_duration)
        duration = asked if asked else _clamp(line.duration, min_duration, max_duration)
        end = min(out_duration, line.start + duration)
        if end - line.start < min_duration * 0.6:
            continue
        overlays.append(Overlay(
            id=f"o{len(overlays) + 1}", asset=str(asset.path),
            out_start=round(line.start, 3), out_end=round(end, 3),
            mode=profile.get("broll.mode"), opacity=float(profile.get("broll.opacity")),
            keyword=keyword, score=round(score, 3), asset_start=asset.start,
            audio=profile.get("broll.audio"),
        ))
        last_end = end
    return overlays


# ---------------------------------------------------------------- transitions

def plan_transitions(timeline: Timeline, analysis: Analysis, profile) -> list[Transition]:
    """Put a short effect on the cuts, so a jump cut reads as intentional.

    Kept sparse on purpose: a transition on every cut is what makes an edit feel
    like a template. `auto` picks by context - a blur where the shot actually
    changed, a flash where a long pause was removed, a punch otherwise.
    """
    if not profile.get("transitions.enabled"):
        return []

    boundaries = timeline.cut_boundaries()
    if not boundaries:
        return []

    # `kind: none` (or null) means no transitions. Note that profile overrides coerce
    # the string "none" to None, so both spellings have to land here.
    raw_kind = profile.get("transitions.kind")
    if raw_kind is None or str(raw_kind).lower() == "none":
        return []
    kind = str(raw_kind).lower()

    duration = float(profile.get("transitions.duration"))
    strength = float(profile.get("transitions.strength"))
    min_gap = float(profile.get("transitions.min_gap"))
    scene_only = bool(profile.get("transitions.scene_change_only"))
    budget = max(1, int(round(float(profile.get("transitions.max_per_min"))
                              * timeline.duration / 60.0)))

    transitions: list[Transition] = []
    last_at = -999.0
    for index, boundary in enumerate(boundaries):
        if len(transitions) >= budget or boundary - last_at < min_gap:
            continue

        src = timeline.to_src(boundary)
        scene_change = False
        removed = 0.0
        if src is not None:
            nearest = analysis.nearest_scene(src)
            scene_change = nearest is not None and abs(nearest - src) < 0.35
        if index + 1 <= len(timeline.cuts) - 1:
            # How much source time was dropped at this join.
            removed = timeline.cuts[index + 1].src_start - timeline.cuts[index].src_end

        if scene_only and not scene_change:
            continue

        if kind == "auto":
            chosen = "blur" if scene_change else ("flash" if removed > 1.2 else "punch")
        else:
            chosen = kind

        transitions.append(Transition(
            id=f"t{len(transitions) + 1}", out_time=round(boundary, 3), kind=chosen,
            duration=duration, strength=strength,
        ))
        last_at = boundary
    return transitions


# ---------------------------------------------------------------------- build

def label_cuts(cuts: list[Cut], transcript: Transcript) -> None:
    """Write what is said into each segment, in source time.

    Output time is no use here: a segment you have switched off has no output
    time at all, and a list of unlabelled durations is no way to decide whether
    you meant to cut it.
    """
    words = transcript.words
    for cut in cuts:
        said = [w.text for w in words
                if w.start >= cut.src_start - 0.15 and w.end <= cut.src_end + 0.15]
        text = " ".join(said).strip()
        cut.text = text if len(text) <= 120 else text[:117].rstrip() + "..."


def build_edl(source: str, analysis: Analysis, transcript: Transcript, profile,
              *, library: BrollLibrary | None = None, scorer=None,
              weights: dict[str, float] | None = None) -> EDL:
    """Run the whole decision pass and return an inspectable edit."""
    cuts = plan_cuts(analysis, transcript, profile)
    label_cuts(cuts, transcript)
    timeline = Timeline(cuts)
    out_duration = timeline.duration

    # Captions are authored in output time, so remap every word through the cuts.
    retimed: list[Word] = []
    for word in transcript.words:
        start = timeline.to_out(word.start)
        end = timeline.to_out(word.end)
        if start is None or end is None or end <= start:
            continue
        retimed.append(Word(text=word.text, start=start, end=end, prob=word.prob))

    # Order matters: the duplicate is detected by its overlap, which
    # enforce_order would otherwise remove first.
    retimed = enforce_order(drop_repeated_words(retimed))
    lines = group_words(retimed, profile) if profile.get("captions.enabled") else []
    zooms = plan_zooms(lines or _fallback_lines(out_duration), analysis, timeline,
                       profile, scorer=scorer)
    overlays = plan_overlays(lines, library or BrollLibrary(), profile,
                             out_duration=out_duration, weights=weights)
    transitions = plan_transitions(timeline, analysis, profile)

    return EDL(
        source=source,
        output={
            "width": int(profile.get("output.width")),
            "height": int(profile.get("output.height")),
            "fps": int(profile.get("output.fps")),
        },
        cuts=cuts,
        zooms=zooms,
        overlays=overlays,
        transitions=transitions,
        captions=lines,
        audio={
            "loudnorm": bool(profile.get("audio.loudnorm")),
            "target_lufs": float(profile.get("audio.target_lufs")),
            "music_path": profile.get("audio.music_path"),
            "music_db": float(profile.get("audio.music_db")),
            "duck_db": float(profile.get("audio.duck_db")),
        },
        meta={
            "profile": profile.get("name"),
            "source_duration": round(analysis.duration, 3),
            "asr_backend": transcript.backend,
            "asr_model": transcript.model,
            "scorer": "model" if scorer else "rules",
        },
    )


def _fallback_lines(out_duration: float) -> list[CaptionLine]:
    """Even with captions off we still want beats to hang zooms on."""
    lines: list[CaptionLine] = []
    step = 2.0
    position = 0.0
    while position < out_duration:
        end = min(out_duration, position + step)
        lines.append(CaptionLine(words=[Word(text="", start=position, end=end)],
                                 start=position, end=end))
        position = end
    return lines
