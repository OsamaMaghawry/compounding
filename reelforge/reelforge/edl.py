"""The Edit Decision List - the contract between "the AI decided" and "ffmpeg renders".

Every automatic decision lands here as plain, inspectable JSON with a stable id and
an `enabled` flag. That single design choice is what makes the rest possible:
you can read the edit, flip any decision off, re-render in seconds, and the diff
between what was proposed and what you kept is exactly the training signal.

Time discipline: `src_*` fields are timestamps in the original file, `out_*` fields
are timestamps in the finished video. Once silences are cut these differ, so every
effect is stored in OUTPUT time and only the segment list knows about source time.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .captions import CaptionLine

EDL_VERSION = 1


@dataclass
class Cut:
    """A slice of the source that survives into the output."""
    src_start: float
    src_end: float
    out_start: float
    out_end: float
    kind: str = "speech"
    id: str = ""
    enabled: bool = True
    text: str = ""          # what is said here, so a segment can be read before it is cut

    @property
    def duration(self) -> float:
        return max(0.0, self.src_end - self.src_start)

    def to_dict(self) -> dict:
        return {"src_start": round(self.src_start, 3), "src_end": round(self.src_end, 3),
                "out_start": round(self.out_start, 3), "out_end": round(self.out_end, 3),
                "kind": self.kind, "id": self.id, "enabled": self.enabled,
                "text": self.text}


@dataclass
class Zoom:
    id: str
    out_start: float
    out_end: float
    start_factor: float
    end_factor: float
    kind: str = "punch_in"
    score: float = 0.0
    enabled: bool = True
    features: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.out_end - self.out_start)

    def to_dict(self) -> dict:
        return {"id": self.id, "out_start": round(self.out_start, 3),
                "out_end": round(self.out_end, 3),
                "start_factor": round(self.start_factor, 4),
                "end_factor": round(self.end_factor, 4), "kind": self.kind,
                "score": round(self.score, 4), "enabled": self.enabled,
                "features": self.features}


@dataclass
class Transition:
    """A short effect sitting on a cut, in output time."""
    id: str
    out_time: float
    kind: str = "punch"          # punch | flash | blur
    duration: float = 0.18
    strength: float = 0.7
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"id": self.id, "out_time": round(self.out_time, 3), "kind": self.kind,
                "duration": round(self.duration, 3), "strength": round(self.strength, 3),
                "enabled": self.enabled}


@dataclass
class Overlay:
    id: str
    asset: str
    out_start: float
    out_end: float
    mode: str = "cover"
    opacity: float = 1.0
    keyword: str = ""
    score: float = 0.0
    asset_start: float = 0.0
    audio: str = "mute"
    enabled: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.out_end - self.out_start)

    def to_dict(self) -> dict:
        return {"id": self.id, "asset": self.asset, "out_start": round(self.out_start, 3),
                "out_end": round(self.out_end, 3), "mode": self.mode,
                "opacity": self.opacity, "keyword": self.keyword,
                "score": round(self.score, 4), "asset_start": round(self.asset_start, 3),
                "audio": self.audio, "enabled": self.enabled}


class Timeline:
    """Maps source time to output time across a list of kept cuts."""

    def __init__(self, cuts: list[Cut]):
        self.cuts = sorted(cuts, key=lambda c: c.src_start)
        self._starts = [c.src_start for c in self.cuts]

    @property
    def duration(self) -> float:
        return self.cuts[-1].out_end if self.cuts else 0.0

    def to_out(self, src_t: float, *, clamp: bool = True) -> float | None:
        """Translate a source timestamp into output time.

        Returns None for time that was cut away, unless `clamp` is set, in which
        case it snaps to the nearest surviving frame - what you want for the edges
        of a word that straddles a cut.
        """
        if not self.cuts:
            return None
        index = bisect_right(self._starts, src_t) - 1
        if index < 0:
            return self.cuts[0].out_start if clamp else None
        cut = self.cuts[index]
        if src_t <= cut.src_end:
            return cut.out_start + (src_t - cut.src_start)
        if not clamp:
            return None
        if index + 1 < len(self.cuts):
            return self.cuts[index + 1].out_start
        return cut.out_end

    def to_src(self, out_t: float) -> float | None:
        for cut in self.cuts:
            if cut.out_start <= out_t <= cut.out_end:
                return cut.src_start + (out_t - cut.out_start)
        return None

    def to_source_spans(self, out_start: float, out_end: float) -> list[tuple[float, float]]:
        """The stretches of original footage behind a span of finished video.

        A selection made while watching the edit can cover several segments with
        already-removed footage between them. Mapping only its two ends would
        silently claim that footage as well, so each surviving segment inside the
        selection is converted on its own.
        """
        spans: list[tuple[float, float]] = []
        for cut in self.cuts:
            low = max(out_start, cut.out_start)
            high = min(out_end, cut.out_end)
            if high - low > 0.01:
                spans.append((cut.src_start + (low - cut.out_start),
                              cut.src_start + (high - cut.out_start)))
        return spans

    def cut_boundaries(self) -> list[float]:
        """Output-time positions where a jump cut happens."""
        return [c.out_start for c in self.cuts[1:]]

    def removed_duration(self, source_duration: float) -> float:
        return max(0.0, source_duration - self.duration)


@dataclass
class EDL:
    source: str
    output: dict
    cuts: list[Cut] = field(default_factory=list)
    zooms: list[Zoom] = field(default_factory=list)
    overlays: list[Overlay] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    captions: list[CaptionLine] = field(default_factory=list)
    audio: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    version: int = EDL_VERSION

    @property
    def timeline(self) -> Timeline:
        return Timeline(self.active_cuts())

    def active_cuts(self) -> list[Cut]:
        return [c for c in self.cuts if c.enabled and c.duration > 0.01]

    @property
    def duration(self) -> float:
        return self.timeline.duration

    def active_zooms(self) -> list[Zoom]:
        return [z for z in self.zooms if z.enabled and z.duration > 0.05]

    def active_overlays(self) -> list[Overlay]:
        return [o for o in self.overlays if o.enabled and o.duration > 0.05]

    def active_transitions(self) -> list[Transition]:
        return [t for t in self.transitions if t.enabled and t.duration > 0.01]

    def summary(self) -> dict:
        src_duration = float(self.meta.get("source_duration") or 0.0)
        return {
            "source_duration": round(src_duration, 2),
            "output_duration": round(self.duration, 2),
            "removed": round(max(0.0, src_duration - self.duration), 2),
            "cuts": len(self.active_cuts()),
            "zooms": len(self.active_zooms()),
            "overlays": len(self.active_overlays()),
            "transitions": len(self.active_transitions()),
            "caption_lines": len(self.captions),
            "words": sum(len(line.words) for line in self.captions),
        }

    def retime(self, enabled: dict[str, bool] | None = None, adjust=None) -> None:
        """Switch segments on or off, and move every effect to match.

        Effects and captions are stored in OUTPUT time, so dropping a segment
        shifts all of them. Leaving them where they were would put a zoom on the
        wrong sentence and captions out of sync with the voice - the two things
        this tool exists to get right. So each one is converted back to source
        time against the old timeline and forward again against the new one.

        Anything whose moment was in the removed segment has nowhere to go and is
        taken out. It is removed rather than switched off because switching off
        means "I did not want this", which is a thing the editor learns from, and
        trimming a segment says nothing about the zoom that happened to sit in it.

        `adjust` may move the segment boundaries themselves - lengthening or
        shortening the pause at a join. It runs here, between reading where
        everything was and working out where it goes, because that is the only
        moment at which both answers are available.
        """
        old = self.timeline

        def source_of(out_t: float) -> float | None:
            return old.to_src(out_t)

        # Read every effect's position before anything moves.
        zoom_src = [(z, source_of(z.out_start), source_of(z.out_end)) for z in self.zooms]
        overlay_src = [(o, source_of(o.out_start), source_of(o.out_end)) for o in self.overlays]
        transition_src = [(t, source_of(t.out_time)) for t in self.transitions]
        caption_src = [
            (line, [(w, source_of(w.start), source_of(w.end)) for w in line.words])
            for line in self.captions
        ]

        if enabled is not None:
            for cut in self.cuts:
                if cut.id in enabled:
                    cut.enabled = bool(enabled[cut.id])
        if adjust is not None:
            adjust(self.cuts)

        # Rebuild output time. A dropped segment collapses to nothing rather than
        # keeping a stale span that would quietly still map.
        cursor = 0.0
        for cut in sorted(self.cuts, key=lambda c: c.src_start):
            if cut.enabled and cut.duration > 0.01:
                cut.out_start, cut.out_end = cursor, cursor + cut.duration
                cursor = cut.out_end
            else:
                cut.out_start = cut.out_end = cursor

        new = self.timeline

        def moved(src_t: float | None) -> float | None:
            if src_t is None:
                return None
            return new.to_out(src_t, clamp=False)

        def respan(items):
            kept = []
            for item, start, end in items:
                out_start, out_end = moved(start), moved(end)
                if out_start is None or out_end is None or out_end - out_start <= 0.01:
                    continue
                item.out_start, item.out_end = out_start, out_end
                kept.append(item)
            return kept

        self.zooms = respan(zoom_src)
        self.overlays = respan(overlay_src)

        transitions = []
        for transition, src_t in transition_src:
            out_t = moved(src_t)
            if out_t is not None:
                transition.out_time = out_t
                transitions.append(transition)
        self.transitions = transitions

        captions = []
        for line, words in caption_src:
            kept = []
            for word, start, end in words:
                out_start, out_end = moved(start), moved(end)
                if out_start is None or out_end is None or out_end <= out_start:
                    continue
                word.start, word.end = out_start, out_end
                kept.append(word)
            if not kept:
                continue
            line.words = kept
            line.start, line.end = kept[0].start, kept[-1].end
            captions.append(line)
        self.captions = captions

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "output": self.output,
            "cuts": [c.to_dict() for c in self.cuts],
            "zooms": [z.to_dict() for z in self.zooms],
            "overlays": [o.to_dict() for o in self.overlays],
            "transitions": [t.to_dict() for t in self.transitions],
            "captions": [line.to_dict() for line in self.captions],
            "audio": self.audio,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EDL":
        version = int(data.get("version", EDL_VERSION))
        if version > EDL_VERSION:
            raise ValueError(
                f"this EDL was written by a newer ReelForge (v{version} > v{EDL_VERSION})"
            )
        return cls(
            source=data["source"],
            output=data.get("output", {}),
            cuts=[Cut(**c) for c in data.get("cuts", [])],
            zooms=[Zoom(**z) for z in data.get("zooms", [])],
            overlays=[Overlay(**o) for o in data.get("overlays", [])],
            transitions=[Transition(**t) for t in data.get("transitions", [])],
            captions=[CaptionLine.from_dict(c) for c in data.get("captions", [])],
            audio=data.get("audio", {}),
            meta=data.get("meta", {}),
            version=version,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EDL":
        return cls.from_dict(json.loads(Path(path).read_text("utf-8")))


def build_timeline(keep: list[tuple[float, float]]) -> list[Cut]:
    """Turn (src_start, src_end) keeps into cuts carrying output timestamps."""
    cuts: list[Cut] = []
    cursor = 0.0
    for index, (src_start, src_end) in enumerate(sorted(keep)):
        duration = max(0.0, src_end - src_start)
        if duration <= 0:
            continue
        cuts.append(Cut(src_start=src_start, src_end=src_end,
                        out_start=cursor, out_end=cursor + duration,
                        id=f"seg{index:03d}"))
        cursor += duration
    return cuts
