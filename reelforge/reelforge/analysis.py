"""Signal analysis: where is the speech, where is the energy, where does the shot change.

Everything here is derived from ffmpeg output or raw PCM - no ML, no downloads - so it
runs in a second or two on a 60s clip and is cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import subprocess
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ffmpeg import MediaInfo, extract_pcm, ffmpeg_bin, probe, run

SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")
PTS_RE = re.compile(r"pts_time:(-?[\d.]+)")
SCORE_RE = re.compile(r"lavfi\.scd\.score=([\d.]+)")


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, other: "Interval", pad: float = 0.0) -> bool:
        return self.start < other.end + pad and other.start < self.end + pad


@dataclass
class Analysis:
    """Everything we know about a source clip before any AI is involved."""

    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    silences: list[Interval] = field(default_factory=list)
    speech: list[Interval] = field(default_factory=list)
    scenes: list[float] = field(default_factory=list)
    energy_hop: float = 0.05
    energy: list[float] = field(default_factory=list)   # dBFS per hop
    loudness_mean: float = -30.0
    loudness_peak: float = -10.0

    def energy_at(self, t: float) -> float:
        if not self.energy:
            return self.loudness_mean
        idx = min(len(self.energy) - 1, max(0, int(t / self.energy_hop)))
        return self.energy[idx]

    def energy_range(self, start: float, end: float) -> tuple[float, float]:
        """(mean, peak) dBFS over a time range."""
        if not self.energy or end <= start:
            return self.loudness_mean, self.loudness_mean
        lo = max(0, int(start / self.energy_hop))
        hi = min(len(self.energy), max(lo + 1, int(math.ceil(end / self.energy_hop))))
        window = self.energy[lo:hi]
        if not window:
            return self.loudness_mean, self.loudness_mean
        return sum(window) / len(window), max(window)

    def nearest_scene(self, t: float) -> float | None:
        if not self.scenes:
            return None
        return min(self.scenes, key=lambda s: abs(s - t))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["silences"] = [asdict(i) for i in self.silences]
        data["speech"] = [asdict(i) for i in self.speech]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Analysis":
        data = dict(data)
        data["silences"] = [Interval(**i) for i in data.get("silences", [])]
        data["speech"] = [Interval(**i) for i in data.get("speech", [])]
        return cls(**data)


# ---------------------------------------------------------------- detection

def detect_silences(path: str | Path, *, noise_db: float = -32.0,
                    min_silence: float = 0.35, duration: float | None = None) -> list[Interval]:
    """Parse ffmpeg's silencedetect output into concrete silent intervals."""
    proc = run([
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence:.3f}",
        "-f", "null", "-",
    ], check=False)
    text = (proc.stderr or "") + (proc.stdout or "")

    silences: list[Interval] = []
    pending: float | None = None
    for line in text.splitlines():
        if "silence_start" in line:
            match = SILENCE_START_RE.search(line)
            if match:
                pending = max(0.0, float(match.group(1)))
        if "silence_end" in line:
            match = SILENCE_END_RE.search(line)
            if match and pending is not None:
                end = float(match.group(1))
                if end > pending:
                    silences.append(Interval(pending, end))
                pending = None
    if pending is not None and duration and duration > pending:
        silences.append(Interval(pending, duration))
    return silences


def invert_intervals(intervals: list[Interval], duration: float) -> list[Interval]:
    """Complement of a set of intervals over [0, duration]."""
    out: list[Interval] = []
    cursor = 0.0
    for iv in sorted(intervals, key=lambda i: i.start):
        if iv.start > cursor:
            out.append(Interval(cursor, min(iv.start, duration)))
        cursor = max(cursor, iv.end)
    if cursor < duration:
        out.append(Interval(cursor, duration))
    return [i for i in out if i.duration > 1e-3]


def detect_scenes(path: str | Path, *, threshold: float = 0.32,
                  max_scenes: int = 400) -> list[float]:
    """Timestamps where the picture changes enough to call it a new shot."""
    proc = run([
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
        "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
        "-an", "-f", "null", "-",
    ], check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    times: list[float] = []
    for line in text.splitlines():
        match = PTS_RE.search(line)
        if match:
            value = float(match.group(1))
            if value > 0.05 and (not times or value - times[-1] > 0.15):
                times.append(value)
        if len(times) >= max_scenes:
            break
    return times


def energy_envelope(path: str | Path, *, hop: float = 0.05,
                    rate: int = 16000) -> tuple[list[float], float, float]:
    """RMS envelope in dBFS, one value per `hop` seconds, computed from raw PCM."""
    try:
        pcm = extract_pcm(path, rate=rate)
    except Exception:
        return [], -30.0, -10.0
    if not pcm:
        return [], -30.0, -10.0

    samples = array("h")
    usable = len(pcm) - (len(pcm) % 2)
    samples.frombytes(pcm[:usable])

    window = max(1, int(hop * rate))
    envelope: list[float] = []
    for start in range(0, len(samples), window):
        chunk = samples[start:start + window]
        if not chunk:
            continue
        total = 0
        for value in chunk:
            total += value * value
        rms = math.sqrt(total / len(chunk))
        db = 20.0 * math.log10(rms / 32768.0) if rms > 0 else -90.0
        envelope.append(round(max(-90.0, db), 2))

    if not envelope:
        return [], -30.0, -10.0
    voiced = [d for d in envelope if d > -55.0] or envelope
    return envelope, sum(voiced) / len(voiced), max(envelope)


# ---------------------------------------------------------------- caching

def content_key(path: str | Path, extra: str = "") -> str:
    """Cheap content fingerprint: size plus head and tail bytes."""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha1()
    digest.update(str(stat.st_size).encode())
    digest.update(extra.encode())
    with path.open("rb") as handle:
        digest.update(handle.read(1 << 20))
        if stat.st_size > (2 << 20):
            handle.seek(-(1 << 20), 2)
            digest.update(handle.read(1 << 20))
    return digest.hexdigest()[:16]


def analyze(path: str | Path, profile, *, cache_dir: Path | None = None,
            info: MediaInfo | None = None, refresh: bool = False) -> Analysis:
    """Run (or reuse) the full signal analysis for a clip."""
    path = Path(path)
    info = info or probe(path)

    cache_file = None
    if cache_dir is not None:
        key = content_key(path, extra=profile.fingerprint())
        cache_file = Path(cache_dir) / f"analysis-{key}.json"
        if cache_file.exists() and not refresh:
            try:
                return Analysis.from_dict(json.loads(cache_file.read_text("utf-8")))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass  # corrupt cache - just recompute

    silences: list[Interval] = []
    envelope: list[float] = []
    mean_db, peak_db = -30.0, -10.0
    if info.has_audio:
        silences = detect_silences(
            path,
            noise_db=profile.get("cuts.noise_db"),
            min_silence=min(0.2, profile.get("cuts.min_silence")),
            duration=info.duration,
        )
        envelope, mean_db, peak_db = energy_envelope(path)

    analysis = Analysis(
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        silences=silences,
        speech=invert_intervals(silences, info.duration) if info.has_audio else [Interval(0.0, info.duration)],
        scenes=detect_scenes(path),
        energy=envelope,
        loudness_mean=round(mean_db, 2),
        loudness_peak=round(peak_db, 2),
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(analysis.to_dict()), encoding="utf-8")
    return analysis
