"""ReelForge as a web app, so editing does not depend on one computer.

Same pipeline as the CLI - this only adds the parts a machine you are not
sitting at needs: uploads, a job queue, progress you can watch from a phone,
and a password, because anything reachable from a hotel wifi will be found.

Renders are CPU-bound, so exactly one job runs at a time; a second upload
queues rather than fighting the first for cores.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import broll
from .edl import EDL, Cut
from .fonts import resolve as font_catalog_resolve
from .pipeline import PACKAGE_ROOT, AutoEditor
from .profile import StyleProfile

# Imported at module level, not inside create_app: `from __future__ import
# annotations` makes every annotation a string, and FastAPI resolves those
# against module globals - a name imported inside a function is invisible to it
# and silently becomes a query parameter.
try:
    from fastapi import (BackgroundTasks, Depends, FastAPI, Form, HTTPException,
                         Request, UploadFile)
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
    HAVE_FASTAPI = True
except ImportError:                                  # the core tool works without it
    HAVE_FASTAPI = False

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
COOKIE = "reelforge_session"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024      # 4 GB per job


# ---------------------------------------------------------------- look panel
#
# The dials worth turning between takes, described once so the browser can build
# the panel itself. Anything not listed here is deliberately out: the profile has
# roughly ninety settings and a phone screen has room for the dozen that change
# how the Reel actually feels.
#
# Every key is a real profile path, so a choice made here is the same thing as
# `--set captions.style=word` on the command line.
LOOK_FIELDS: list[dict] = [
    {"group": "Captions", "key": "captions.style", "label": "Style", "type": "select",
     "help": "How the word you are saying is marked.",
     "options": [
         ("karaoke", "Karaoke — the spoken word changes colour"),
         ("box", "Box — the spoken word sits in a filled box (the CapCut look)"),
         ("pop", "Pop — the whole line pulses on each word"),
         ("word", "One word — a single big word at a time"),
         ("plain", "Plain — no per-word marking"),
     ]},
    {"group": "Captions", "key": "captions.font", "label": "Font", "type": "font",
     "help": "Arabic families only. One that is not downloaded yet is fetched when you apply."},
    {"group": "Captions", "key": "captions.max_words", "label": "Words per line",
     "type": "number", "min": 1, "max": 8, "step": 1,
     "help": "Reels read best at three or four. Set 1 for one-word-at-a-time."},
    {"group": "Captions", "key": "captions.font_size", "label": "Size", "type": "number",
     "min": 40, "max": 170, "step": 2},
    {"group": "Captions", "key": "captions.y_pct", "label": "Height on screen",
     "type": "number", "min": 0.25, "max": 0.92, "step": 0.01,
     "help": "0 is the top, 1 the bottom. 0.72 clears the Instagram buttons."},
    {"group": "Captions", "key": "captions.line_gap", "label": "Pause between lines",
     "type": "number", "min": 0.0, "max": 0.6, "step": 0.02,
     "help": "Blank time between one line and the next, in seconds."},
    {"group": "Captions", "key": "captions.highlight", "label": "Spoken-word colour",
     "type": "color"},
    {"group": "Captions", "key": "captions.primary", "label": "Text colour", "type": "color"},
    {"group": "Captions", "key": "captions.emphasis", "label": "Mark important words",
     "type": "bool", "help": "Keeps key words coloured even when you are past them."},
    {"group": "Captions", "key": "captions.emphasis_color", "label": "Important-word colour",
     "type": "color"},

    {"group": "Motion", "key": "zoom.enabled", "label": "Punch in while speaking",
     "type": "bool"},
    {"group": "Motion", "key": "zoom.rate_per_min", "label": "Moves per minute",
     "type": "number", "min": 0, "max": 40, "step": 1,
     "help": "How often the camera pushes in or pulls out."},
    {"group": "Motion", "key": "zoom.max_factor", "label": "Strongest punch",
     "type": "number", "min": 1.0, "max": 1.6, "step": 0.01},
    {"group": "Motion", "key": "transitions.kind", "label": "Transition", "type": "select",
     "help": "What happens at a cut.",
     "options": [
         ("auto", "Auto — pick per cut"),
         ("punch", "Punch — a fast scale kick"),
         ("flash", "Flash — a brief brightness lift"),
         ("blur", "Blur — a short smear"),
         ("none", "None — hard cuts only"),
     ]},
    {"group": "Motion", "key": "transitions.duration", "label": "Transition length",
     "type": "number", "min": 0.05, "max": 0.6, "step": 0.01,
     "help": "Seconds. A transition you notice is too long."},
    {"group": "Motion", "key": "transitions.strength", "label": "Transition strength",
     "type": "number", "min": 0.0, "max": 1.0, "step": 0.05},
    {"group": "Motion", "key": "transitions.max_per_min", "label": "Transitions per minute",
     "type": "number", "min": 0, "max": 60, "step": 1},

    {"group": "B-roll", "key": "broll.enabled", "label": "Cut in b-roll", "type": "bool",
     "help": "Covers you with a clip from your library when you say a word it matches."},
    {"group": "B-roll", "key": "broll.max_per_min", "label": "Clips per minute",
     "type": "number", "min": 0, "max": 20, "step": 1},
    {"group": "B-roll", "key": "broll.min_score", "label": "Match confidence",
     "type": "number", "min": 0.3, "max": 1.0, "step": 0.05,
     "help": "How sure the word match must be. Raise it if the wrong clips appear."},
    {"group": "B-roll", "key": "broll.mode", "label": "How it appears", "type": "select",
     "options": [
         ("cover", "Cover — fills the screen"),
         ("pip", "Corner — a smaller inset over you"),
         ("band", "Band — a strip across the middle"),
     ]},
    {"group": "B-roll", "key": "broll.max_duration", "label": "Longest clip",
     "type": "number", "min": 0.6, "max": 6.0, "step": 0.2,
     "help": "Seconds. Short is the point - it is a cutaway, not a scene."},

    {"group": "Pacing", "key": "cuts.enabled", "label": "Cut dead air", "type": "bool"},
    {"group": "Pacing", "key": "cuts.min_silence", "label": "Shortest silence to cut",
     "type": "number", "min": 0.15, "max": 1.5, "step": 0.05,
     "help": "Seconds. Lower is more aggressive."},
    {"group": "Pacing", "key": "cuts.max_gap_keep", "label": "Pause left behind",
     "type": "number", "min": 0.05, "max": 1.0, "step": 0.01,
     "help": "Seconds of breath kept where a silence was removed."},
]

LOOK_KEYS = {field["key"] for field in LOOK_FIELDS}
LOOK_LABELS = {field["key"]: f"{field['group'].lower()}: {field['label'].lower()}"
               for field in LOOK_FIELDS}


def _remaining(edl: EDL, drops: list) -> list[tuple[float, float]]:
    """Source stretches still kept, once these drops are taken out."""
    ranges = [(float(a), float(b)) for a, b in drops if float(b) > float(a)]
    kept: list[tuple[float, float]] = []
    for cut in edl.cuts:
        if not cut.enabled:
            continue
        pieces = [(cut.src_start, cut.src_end)]
        for low, high in ranges:
            nxt = []
            for start, end in pieces:
                if high <= start or low >= end:
                    nxt.append((start, end))
                    continue
                if start < low:
                    nxt.append((start, low))
                if high < end:
                    nxt.append((high, end))
            pieces = nxt
        kept += pieces
    return kept


MIN_PIECE = 0.12          # never shave a segment down to nothing


def apply_pauses(edl: EDL, pauses: list) -> None:
    """Move one edge of one piece, lengthening or shortening the pause after it.

    Each entry is [anchor, seconds, side]: which join, how far, and which of the
    two pieces meeting there it belongs to. `before` moves the end of the piece
    on the left, `after` moves the start of the piece on the right, and neither
    ever touches the other - dragging the tail of one shot has no business
    trimming the head of the next.

    Positive seconds give back footage the cut removed; negative take more away,
    eating into the padding and then into the shot, but never past the point
    where it would have nothing left.

    A two-item entry is from before the sides were separate and still means the
    old both-at-once behaviour, so edits saved then still open.
    """
    wanted = []
    for entry in (pauses or []):
        if len(entry) >= 3:
            wanted.append((float(entry[0]), float(entry[1]), str(entry[2])))
        elif len(entry) == 2:
            wanted.append((float(entry[0]), float(entry[1]), "both"))
    wanted = [w for w in wanted if abs(w[1]) > 0.005]
    if not wanted:
        return

    def shift(cuts: list[Cut]) -> None:
        live = sorted([c for c in cuts if c.enabled and c.duration > 0.01],
                      key=lambda c: c.src_start)
        # Midpoints are read before anything moves: they are what the stored
        # entries were matched against, and each edit would otherwise be
        # matched against a join the edit before it had already shifted.
        joins = [((before.src_end + after.src_start) / 2.0, before, after)
                 for before, after in zip(live, live[1:])]
        for middle, before, after in joins:
            for anchor, delta, side in wanted:
                if abs(anchor - middle) > 0.35:
                    continue
                # Each edge is bounded twice, and the bound against the other
                # piece is the outer one: footage that played twice would be
                # worse than a piece trimmed shorter than intended.
                if side == "before":
                    before.src_end = min(after.src_start,
                                         max(before.src_start + MIN_PIECE,
                                             before.src_end + delta))
                elif side == "after":
                    after.src_start = max(before.src_end,
                                          min(after.src_end - MIN_PIECE,
                                              after.src_start - delta))
                else:                                   # saved before the split
                    give = delta / 2.0
                    before.src_end = min(after.src_start,
                                         max(before.src_start + MIN_PIECE,
                                             before.src_end + give))
                    after.src_start = max(before.src_end,
                                          min(after.src_end - MIN_PIECE,
                                              after.src_start - give))

    edl.retime(adjust=shift)


def apply_beats(edl: EDL, beats: list) -> None:
    """Set the length of the transition at a particular join.

    The Look panel sets one length for the whole video, which is right until one
    cut wants to land harder than the rest. Matched in source time so the tweak
    survives the next plan.
    """
    wanted = [(float(mid), float(seconds)) for mid, seconds in (beats or [])]
    if not wanted:
        return
    timeline = edl.timeline
    for transition in edl.transitions:
        where = timeline.to_src(transition.out_time)
        if where is None:
            continue
        match = next((s for mid, s in wanted if abs(mid - where) <= 0.35), None)
        if match is None:
            continue
        transition.duration = max(0.0, min(1.5, match))
        transition.enabled = transition.duration > 0.01


def apply_drops(edl: EDL, drops: list) -> None:
    """Take trimmed-away stretches of the original footage out of the edit.

    A selection made while watching rarely lines up with the segments the silence
    cutter produced - it starts halfway through one and ends halfway through
    another. So segments are split at the edges of what you selected first, and
    only the pieces actually inside it are dropped. Splitting does not change the
    edit: two halves of a segment play exactly as the whole did.

    Matched by overlap in source time rather than by segment id, so a trim holds
    even when a new plan divides the footage differently - which it does the
    moment the pacing settings change.
    """
    ranges = [(float(start), float(end)) for start, end in (drops or [])
              if float(end) > float(start)]
    if not ranges:
        return

    edges = sorted({edge for span in ranges for edge in span})
    pieces: list[Cut] = []
    for cut in edl.cuts:
        points = sorted({cut.src_start, cut.src_end}
                        | {e for e in edges if cut.src_start < e < cut.src_end})
        cursor = cut.out_start
        for start, end in zip(points, points[1:]):
            pieces.append(Cut(src_start=start, src_end=end,
                              out_start=cursor, out_end=cursor + (end - start),
                              kind=cut.kind, id=f"seg{len(pieces):03d}",
                              enabled=cut.enabled, text=cut.text))
            cursor += end - start
    edl.cuts = pieces

    wanted = {}
    for cut in edl.cuts:
        middle = (cut.src_start + cut.src_end) / 2.0
        wanted[cut.id] = cut.enabled and not any(low <= middle <= high
                                                 for low, high in ranges)
    if any(wanted.values()):
        edl.retime(wanted)


def _differs(new: object, old: object) -> bool:
    """Whether a posted setting is actually a change from the template's value.

    The panel posts every control on every apply, so without this the first
    Apply would pin all twenty settings and the template would stop meaning
    anything. Colours come back from the browser lower-cased and numbers as
    floats, neither of which is a change.
    """
    if isinstance(new, str) and isinstance(old, str):
        return new.strip().lower() != old.strip().lower()
    if (isinstance(new, (int, float)) and isinstance(old, (int, float))
            and not isinstance(new, bool) and not isinstance(old, bool)):
        return abs(float(new) - float(old)) > 1e-9
    return new != old


@dataclass
class Job:
    id: str
    title: str
    created: float
    status: str = "queued"          # queued | working | ready | exporting | done | error
    stage: str = "waiting to start"
    progress: list[str] = field(default_factory=list)
    error: str = ""
    sources: list[str] = field(default_factory=list)
    template: str = ""
    model: str = "small"
    run_id: int | None = None
    summary: dict = field(default_factory=dict)
    output_name: str = ""
    overrides: dict = field(default_factory=dict)   # look settings, changed after upload
    prepared: str = ""                              # the joined (or single) source
    # Trimmed-away stretches, as (start, end) in SOURCE time. Source time because
    # it is the one frame of reference that survives everything else: change the
    # caption style, the pacing, the model, and 0:14 to 0:19 of your footage is
    # still 0:14 to 0:19. Segment numbers are not - they renumber on every plan.
    drops: list = field(default_factory=list)
    # Pauses you lengthened or shortened by hand, as [midpoint, seconds] pairs.
    # The midpoint is in source time for the same reason drops are: it still
    # points at the same join after the next plan renumbers everything.
    pauses: list = field(default_factory=list)
    # Per-join transition lengths, [midpoint, seconds], keyed the same way.
    beats: list = field(default_factory=list)
    percent: float = 0.0        # 0-100, so a long wait has a shape
    heartbeat: float = 0.0      # last sign of life, so a stuck job can be told apart
    started: float = 0.0        # when the current run began
    patience: float = 90.0      # how long this step may go quiet before it is odd

    def to_dict(self) -> dict:
        data = asdict(self)
        data["progress"] = self.progress[-40:]
        return data


class JobStore:
    """Jobs on disk, so a restart does not lose what you uploaded."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        for path in sorted(self.root.glob("*/job.json")):
            try:
                data = json.loads(path.read_text("utf-8"))
                job = Job(**{k: v for k, v in data.items() if k in Job.__annotations__})
                if job.status in ("queued", "working", "exporting"):
                    job.status, job.stage = "error", "interrupted by a restart"
                    job.error = "the server restarted while this job was running"
                self._jobs[job.id] = job
            except Exception:
                continue

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, title: str, template: str, model: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], title=title, created=time.time(),
                  template=template, model=model)
        with self._lock:
            self._jobs[job.id] = job
        self.dir(job.id).mkdir(parents=True, exist_ok=True)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        try:
            (self.dir(job.id) / "job.json").write_text(
                json.dumps(job.to_dict(), ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(self.dir(job_id), ignore_errors=True)
        return True

    def update(self, job: Job, **changes) -> None:
        for key, value in changes.items():
            setattr(job, key, value)
        self.save(job)


class Runner:
    """One worker thread. Renders saturate the CPU, so they do not overlap."""

    def __init__(self, store: JobStore, fonts_dir: Path, broll_dir: Path):
        self.store = store
        self.fonts_dir = fonts_dir
        # Beside the jobs, not inside the package: the package sits in the repo
        # and is replaced on every update, which is no place to keep your footage.
        self.broll_dir = Path(broll_dir)
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.editors: dict[str, AutoEditor] = {}
        self.edls: dict[str, EDL] = {}
        self._recent: list[str] = []
        # Restoring happens on whichever thread asked for the edit, which is the
        # request thread as often as the worker.
        self._lock = threading.Lock()
        self._proxy_lock = threading.Lock()
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _owner(self, job: Job) -> int:
        """A stable handle for whatever this job is running, so it can be stopped."""
        return abs(hash(job.id)) % (2 ** 31)

    def stop(self, job: Job) -> bool:
        """Give up on a job. Kills its ffmpeg if one is running."""
        from .ffmpeg import stop_all  # noqa: PLC0415
        killed = stop_all(self._owner(job))
        self.store.update(job, status="error", stage="stopped",
                          error="you stopped this one", percent=0.0)
        return bool(killed)

    def submit(self, job_id: str, task: str = "plan") -> None:
        """Queue work. Rendering can take minutes, which is far too long to hold
        an HTTP request open - every proxy in between would give up first."""
        self.queue.put((task, job_id))

    def _remember(self, job_id: str, editor: AutoEditor, edl: EDL) -> None:
        with self._lock:
            self.editors[job_id] = editor
            self.edls[job_id] = edl
            if job_id in self._recent:
                self._recent.remove(job_id)
            self._recent.append(job_id)
            while len(self._recent) > 20:                 # a server runs for months
                stale = self._recent.pop(0)
                self.editors.pop(stale, None)
                self.edls.pop(stale, None)

    # -- keeping the edit ------------------------------------------------
    #
    # The plan lives in memory, twenty at a time. That was fine until you could
    # leave: a Codespace stops after thirty idle minutes, and coming back to a
    # video you can watch but cannot export - under a message telling you to
    # upload it again, which is not true - is the kind of thing that loses an
    # evening. So the working edit goes to disk beside the preview, and is read
    # back the moment anything asks for it.
    #
    # Beside, not instead of: `project/runs/run-N.edl.json` is what the editor
    # proposed and must stay untouched, because the difference between that and
    # what you kept is the whole training signal.

    # Roughly what each phase costs, so the bar moves for the right reasons.
    # Transcribing dominates; rendering is the next longest thing.
    PHASES = (("joining the clips", 0.06), ("analysing audio and shots", 0.10),
              ("transcribing", 0.55), ("planning the edit", 0.60),
              ("rendering", 0.98))

    def beat(self, job: Job, *, percent: float | None = None,
             stage: str | None = None) -> None:
        """Say we are still here, and how far along.

        Without this a job that has died and a job that is merely slow look the
        same from the page: both say "working" and neither moves.
        """
        changes = {"heartbeat": time.time()}
        if percent is not None:
            # Never goes backwards: a bar that retreats reads as a fault.
            changes["percent"] = round(max(job.percent, min(99.0, percent)), 1)
        if stage is not None:
            changes["stage"] = stage
            changes["patience"] = self._patience_for(job, stage)
        self.store.update(job, **changes)

    def _patience_for(self, job: Job, message: str) -> float:
        """How long this step may reasonably say nothing.

        One number cannot cover it. Whisper on a machine with no graphics card
        goes quiet for minutes at a time between chunks - that is the model
        thinking, not a hang - while a stalled render is odd after one. Judging
        both by the same stopwatch is how a working job gets called dead.
        """
        if message.startswith("transcribing"):
            big = any(name in str(job.model) for name in ("large", "medium"))
            return 900.0 if big else 300.0
        if message.startswith("joining") or message.startswith("rendering"):
            return 180.0
        return 120.0

    def _phase_percent(self, message: str) -> float | None:
        for prefix, share in self.PHASES:
            if message.startswith(prefix[:12]):
                return share * 100.0
        return None

    def _working_path(self, job_id: str) -> Path:
        return self.store.dir(job_id) / "edl.json"

    def keep(self, job_id: str) -> None:
        """Write the current edit to disk. Called after anything changes it."""
        edl = self.edls.get(job_id)
        if edl is None:
            return
        try:
            edl.save(self._working_path(job_id))
        except OSError:
            pass                        # a full disk must not lose the render too

    def edl_for(self, job: Job) -> EDL | None:
        """The current edit, read back from disk if the server restarted."""
        edl = self.edls.get(job.id)
        if edl is not None:
            return edl

        path = self._working_path(job.id)
        if not path.exists():
            # An edit made before the working copy was saved. The proposal is
            # still there, so offer that rather than nothing.
            runs = sorted((self.store.dir(job.id) / "project" / "runs").glob("run-*.edl.json"))
            if not runs:
                return None
            path = max(runs, key=lambda p: p.stat().st_mtime)
        try:
            edl = EDL.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        if not Path(edl.source).exists():
            return None                 # the footage is gone; nothing to render
        editor = AutoEditor(self.profile_for(job),
                            project_dir=self.store.dir(job.id) / "project",
                            fonts_dir=self.fonts_dir)
        self._remember(job.id, editor, edl)
        return edl

    def _loop(self) -> None:
        while True:
            task, job_id = self.queue.get()
            job = self.store.get(job_id)
            if job is None:
                continue
            try:
                if task == "plan":
                    self._process(job)
                elif task == "rerender":
                    self._rerender(job)
                elif task == "replan":
                    self._replan(job)
                elif task == "replan_only":
                    self._replan(job, render=False)
                elif task == "export":
                    self._export(job)
            except Exception as exc:                      # a failed job must not kill the worker
                self.store.update(job, status="error", stage="failed",
                                  error=f"{type(exc).__name__}: {exc}"[:400])

    def _process(self, job: Job) -> None:
        directory = self.store.dir(job.id)

        def note(message: str) -> None:
            job.progress.append(message)
            self.beat(job, stage=message, percent=self._phase_percent(message))

        self.store.update(job, status="working", stage="starting", error="",
                          percent=0.0, heartbeat=time.time(), started=time.time())
        profile = self.profile_for(job)
        self._ensure_font(profile, note)
        editor = AutoEditor(profile, project_dir=directory / "project",
                            fonts_dir=self.fonts_dir, on_status=note,
                            on_progress=lambda f: self.beat(job, percent=10 + 45 * f))
        clips = [Path(p) for p in job.sources]
        if len(clips) > 1:
            from .join import join_clips  # noqa: PLC0415
            source = join_clips(clips, editor.work_dir / "joined.mp4",
                                max_height=int(profile.get("output.height")
                                               * float(profile.get("output.zoom_headroom"))),
                                on_status=note,
                                # Joining is the longest step on phone footage and
                                # said nothing while it ran, so the page called it
                                # stuck when it was working perfectly well.
                                on_fraction=lambda f: self.beat(job, percent=2 + 8 * f),
                                owner=self._owner(job))
        else:
            source = clips[0]

        self.store.update(job, prepared=str(source))
        note("preparing the footage for playback")
        self._ensure_proxy(job, source)
        self._plan_into(job, source, editor, note)

    def profile_for(self, job: Job, values: dict | None = None) -> StyleProfile:
        """Template, then the model, then the look - saved, or merely proposed."""
        profile = StyleProfile.resolve(job.template or None,
                                       [PACKAGE_ROOT / "templates", PACKAGE_ROOT / "profiles"])
        overrides = [f"asr.model={job.model}",
                     f"broll.library={self.broll_dir}"]
        backend = os.environ.get("REELFORGE_ASR_BACKEND")
        if backend:
            overrides.append(f"asr.backend={backend}")
        chosen = dict(job.overrides or {})
        chosen.update(values or {})
        overrides += [f"{key}={value}" for key, value in chosen.items()]
        return profile.apply_overrides(overrides)

    def _ensure_font(self, profile: StyleProfile, note) -> None:
        """Fetch the caption font if picking it is the first time it is used.

        Choosing a font in the browser and getting tofu boxes back would be a
        baffling way to learn that fonts are downloaded on demand, so download it
        here. A failure is only a warning - libass falls back, and a fallback
        preview is more useful than no preview.
        """
        from . import fonts as font_catalog  # noqa: PLC0415
        from .render import check_font       # noqa: PLC0415
        if not profile.get("captions.enabled"):
            return
        if check_font(profile, self.fonts_dir) is None:
            return
        entry = font_catalog.resolve(profile.get("captions.font") or "")
        if entry is None:
            return
        note(f"downloading {entry.family}")
        _changed, message = font_catalog.download(entry, self.fonts_dir)
        note(message)

    def _ensure_proxy(self, job: Job, source: Path | None = None) -> Path | None:
        """The small copy the browser plays. Built once, then kept.

        Also reachable from the request that wants it, because edits made before
        there was a player have no proxy: without this they would open to an
        empty black box, which looks a great deal like the edit being gone.
        """
        from .render import build_proxy  # noqa: PLC0415
        proxy = self.store.dir(job.id) / "proxy.mp4"
        if proxy.exists() and proxy.stat().st_size > 0:
            return proxy
        if source is None:
            source = Path(job.prepared) if job.prepared else None
        if not source or not source.exists():
            return None
        # A video element asks for several ranges at once; one build, not five.
        with self._proxy_lock:
            if proxy.exists() and proxy.stat().st_size > 0:
                return proxy
            try:
                return build_proxy(source, proxy)
            except Exception:
                return None

    def broll_proxy(self, asset: Path) -> Path | None:
        """The small copy of a library clip, built at most once.

        Under the lock because a video element asks for several ranges at once:
        without it, a clip with no proxy yet would start an encode per request,
        and a handful of those is enough to bring the machine to its knees.
        """
        dest = self.broll_dir / ".proxies" / f"{asset.name}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        with self._proxy_lock:
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            return broll.make_proxy(asset, dest)

    def _plan_into(self, job: Job, source: Path, editor: AutoEditor, note,
                   *, render: bool = True) -> None:
        """Decide the edit for `source`, and render a preview only if asked.

        Not rendering is the normal case now: the browser plays the proxy and
        draws the edit over it, so a new decision is something you see rather
        than something you queue.
        """
        result = editor.plan(source)
        apply_pauses(result.edl, job.pauses)
        apply_drops(result.edl, job.drops)
        apply_beats(result.edl, job.beats)
        self._remember(job.id, editor, result.edl)
        for warning in result.warnings:
            note(warning)
        self.keep(job.id)
        if render:
            note("rendering preview")
            editor.render(result.edl, self.store.dir(job.id) / "preview.mp4", preview=True,
                          on_fraction=lambda f: self.beat(job, percent=75 + 24 * f),
                          owner=self._owner(job))
        self.store.update(job, status="ready", stage="ready to review",
                          run_id=result.run_id, summary=result.edl.summary(),
                          percent=100.0, heartbeat=time.time())

    def plan_preview(self, job: Job, values: dict, drops: list,
                     pauses: list | None = None, beats: list | None = None) -> dict:
        """What the edit would be with these settings - committing nothing.

        This is what makes the panel feel live: the answer comes back in about a
        second, nothing is stored, and nothing is rendered. Saving is a separate
        act, which is the point - you can try six caption styles and keep none.
        """
        source = Path(job.prepared) if job.prepared else None
        if not source or not source.exists():
            raise RuntimeError("this edit is no longer loaded - upload it again")
        profile = self.profile_for(job, values)
        editor = AutoEditor(profile, project_dir=self.store.dir(job.id) / "project",
                            fonts_dir=self.fonts_dir)
        result = editor.plan(source, record=False)
        apply_pauses(result.edl, pauses if pauses is not None else job.pauses)
        apply_drops(result.edl, drops)
        apply_beats(result.edl, beats if beats is not None else job.beats)
        return {"edl": result.edl.to_dict(), "summary": result.edl.summary(),
                "look": profile.section("captions")}

    def _replan(self, job: Job, *, render: bool = True) -> None:
        """Re-decide the edit with new look settings.

        Fast, because the analysis and the transcript are cached against the
        source: changing the caption style or the transitions never re-listens to
        the audio. What it does cost is the preview render, and the enable
        checkboxes - a different rate gives you different moves, so there is
        nothing for the old ticks to attach to.
        """
        source = Path(job.prepared) if job.prepared else None
        if not source or not source.exists():
            raise RuntimeError("this edit is no longer loaded - upload it again")

        def note(message: str) -> None:
            job.progress.append(message)
            self.beat(job, stage=message, percent=self._phase_percent(message))

        fixes = self._typed_fixes(job)
        self.store.update(job, status="working", stage="applying the new look", error="")
        profile = self.profile_for(job)
        self._ensure_font(profile, note)
        editor = AutoEditor(profile, project_dir=self.store.dir(job.id) / "project",
                            fonts_dir=self.fonts_dir, on_status=note,
                            on_progress=lambda f: self.beat(job, percent=10 + 45 * f))
        self._plan_into(job, source, editor, note, render=render)
        if fixes:
            self._apply_fixes(job, fixes)
            self.keep(job.id)
            self.store.update(job, summary=self.edls[job.id].summary())

    def _typed_fixes(self, job: Job) -> dict[str, str]:
        """Word corrections typed into the current captions, as wrong -> right.

        Re-planning rebuilds the captions from the transcript, which would quietly
        throw away a name you just spelled correctly. Feeding them back through
        the ASR would be the thorough fix, but that means transcribing again -
        minutes, for a caption-style change. Replaying them over the new words
        costs nothing and keeps what you typed.
        """
        from .learn import _diff_caption_words  # noqa: PLC0415
        edl = self.edl_for(job)                  # a restart must not lose them either
        if edl is None or not job.run_id:
            return {}
        proposed_path = self.editors[job.id].runs_dir / f"run-{job.run_id}.edl.json"
        if not proposed_path.exists():
            return {}
        try:
            proposed = EDL.load(proposed_path)
        except Exception:
            return {}
        return dict(_diff_caption_words(proposed, edl))

    def _apply_fixes(self, job: Job, fixes: dict[str, str]) -> None:
        from .arabic import VocabCorrector  # noqa: PLC0415
        corrector = VocabCorrector(fixes)
        for line in self.edls[job.id].captions:
            for word in line.words:
                word.text = corrector.correct_word(word.text)

    def _rerender(self, job: Job) -> None:
        edl = self.edl_for(job)
        if edl is None:
            raise RuntimeError("this edit is no longer loaded - upload it again")
        editor = self.editors[job.id]
        self.store.update(job, status="working", stage="re-rendering the preview",
                          percent=0.0, heartbeat=time.time(), started=time.time())
        editor.render(edl, self.store.dir(job.id) / "preview.mp4", preview=True,
                      on_fraction=lambda f: self.beat(job, percent=100 * f),
                      owner=self._owner(job))
        self.store.update(job, status="ready", stage="ready to review",
                          summary=edl.summary())

    def _export(self, job: Job) -> None:
        edl = self.edl_for(job)
        if edl is None:
            raise RuntimeError("this edit is no longer loaded - upload it again")
        editor = self.editors[job.id]
        self.store.update(job, status="exporting", stage="rendering the final video",
                          percent=0.0, heartbeat=time.time(), started=time.time())
        name = f"{Path(job.title).stem or 'reel'}-reel.mp4"
        editor.render(edl, self.store.dir(job.id) / name, preview=False,
                      on_fraction=lambda f: self.beat(job, percent=100 * f),
                      owner=self._owner(job))
        if job.run_id:
            try:
                editor.accept(job.run_id, edl)
            except Exception:
                pass                                       # learning must never block a download
        self.store.update(job, status="done", stage="finished", output_name=name)


# ------------------------------------------------------------------ security

def _sign(secret: str, value: str) -> str:
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:32]


def make_token(secret: str) -> str:
    issued = str(int(time.time()))
    return f"{issued}.{_sign(secret, issued)}"


def valid_token(secret: str, token: str | None, max_age: int = 30 * 86400) -> bool:
    if not token or "." not in token:
        return False
    issued, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(secret, issued), signature):
        return False
    try:
        return (time.time() - int(issued)) < max_age
    except ValueError:
        return False


# -------------------------------------------------------------------- routes

def create_app(data_dir: str | Path = "data", password: str | None = None,
               secret: str | None = None):
    if not HAVE_FASTAPI:
        raise RuntimeError(
            "the web app needs a few extra packages. Install them with:\n"
            "  pip install -e \".[web]\"")

    data_dir = Path(data_dir)
    store = JobStore(data_dir / "jobs")
    runner = Runner(store, PACKAGE_ROOT / "assets" / "fonts", data_dir / "broll")

    password = password or os.environ.get("REELFORGE_PASSWORD") or secrets.token_urlsafe(9)
    secret = secret or os.environ.get("REELFORGE_SECRET") or secrets.token_hex(16)
    app = FastAPI(title="ReelForge", docs_url=None, redoc_url=None)
    app.state.password = password
    # Exposed so an embedding process - or a test - can reach the same store and
    # worker the routes use, rather than a second instance over the same folder.
    app.state.store = store
    app.state.runner = runner

    def require_login(request: Request) -> None:
        if not valid_token(secret, request.cookies.get(COOKIE)):
            raise HTTPException(status_code=401, detail="log in first")

    def job_or_404(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    def ranged(path: Path, request: Request, media_type: str) -> Response:
        """Serve a file with Range support, so the preview can be scrubbed."""
        if not path.exists():
            raise HTTPException(status_code=404, detail="not ready")
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = request.headers.get("range")
        if header and header.startswith("bytes="):
            piece = header[6:].split(",")[0]
            first, _, last = piece.partition("-")
            if first.strip():
                start = min(int(first), size - 1)
            if last.strip():
                end = min(int(last), size - 1)
            status = 206
        end = max(start, end)
        length = end - start + 1

        def stream():
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(262144, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(length),
                   "Cache-Control": "no-store"}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(stream(), status_code=status, media_type=media_type,
                                 headers=headers)

    # -- pages ----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/api/login")
    def login(request: Request, body: dict) -> JSONResponse:
        if not hmac.compare_digest(str(body.get("password", "")), app.state.password):
            time.sleep(1.0)                       # blunt the guessing rate
            raise HTTPException(status_code=401, detail="wrong password")
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, make_token(secret), httponly=True, samesite="lax",
                            max_age=30 * 86400,
                            secure=request.url.scheme == "https")
        return response

    @app.post("/api/logout")
    def logout() -> JSONResponse:
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE)
        return response

    @app.get("/api/me")
    def me(request: Request) -> dict:
        from . import __version__  # noqa: PLC0415
        from .cli import _checkout_revision  # noqa: PLC0415
        return {"authenticated": valid_token(secret, request.cookies.get(COOKIE)),
                "version": f"{__version__} · {_checkout_revision()}"}

    # -- jobs -----------------------------------------------------------
    def defaults_path() -> Path:
        return data_dir / "defaults.json"

    def read_defaults() -> dict:
        try:
            raw = json.loads(defaults_path().read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: str(v) for k, v in raw.items() if k in LOOK_KEYS} \
            if isinstance(raw, dict) else {}

    @app.get("/api/defaults", dependencies=[Depends(require_login)])
    def get_defaults() -> dict:
        saved = read_defaults()
        return {"values": saved,
                "labels": sorted(LOOK_LABELS.get(k, k) for k in saved)}

    @app.post("/api/defaults", dependencies=[Depends(require_login)])
    def set_defaults(body: dict) -> dict:
        """Remember a look, so the next upload already arrives looking right.

        A style you settled on is not something to re-pick on every video.
        """
        if body.get("clear"):
            defaults_path().unlink(missing_ok=True)
            return {"ok": True, "values": {}}
        values = body.get("values")
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="no settings were sent")
        unknown = sorted(set(values) - LOOK_KEYS)
        if unknown:
            raise HTTPException(status_code=400,
                                detail=f"not a setting you can change here: {unknown[0]}")
        base = StyleProfile()
        keep = {key: str(value) for key, value in values.items()}
        try:
            candidate = base.apply_overrides([f"{k}={v}" for k, v in keep.items()])
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        keep = {key: value for key, value in keep.items()
                if _differs(candidate.get(key, None), base.get(key, None))}
        defaults_path().parent.mkdir(parents=True, exist_ok=True)
        defaults_path().write_text(json.dumps(keep, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        return {"ok": True, "values": keep,
                "labels": sorted(LOOK_LABELS.get(k, k) for k in keep)}

    _version_seen: dict = {"at": 0.0, "behind": 0}

    @app.get("/api/version", dependencies=[Depends(require_login)])
    def version(check: int = 0) -> dict:
        """What is running, and whether something newer is waiting.

        Asked by the page itself on every load, so a new version announces
        itself instead of waiting to be looked for.
        """
        import subprocess  # noqa: PLC0415
        from . import __version__  # noqa: PLC0415
        from .cli import _checkout_revision  # noqa: PLC0415
        repo = PACKAGE_ROOT.parent
        info = {"version": __version__, "revision": _checkout_revision(),
                "behind": 0, "can_update": (repo / ".git").exists()}
        if not info["can_update"]:
            return info
        # Asking the network costs a second, so the answer is kept briefly: a
        # page reload should not mean another round trip to GitHub.
        fresh = time.time() - _version_seen["at"] < 120
        if fresh and not check:
            info["behind"] = _version_seen["behind"]
            return info
        try:
            subprocess.run(["git", "-C", str(repo), "fetch", "--quiet"],
                           capture_output=True, timeout=25)
            counted = subprocess.run(
                ["git", "-C", str(repo), "rev-list", "--count", "HEAD..@{u}"],
                capture_output=True, text=True, timeout=10)
            info["behind"] = int((counted.stdout or "0").strip() or 0)
        except (OSError, ValueError, subprocess.SubprocessError):
            info["behind"] = 0
        _version_seen.update(at=time.time(), behind=info["behind"])
        return info

    @app.post("/api/update", dependencies=[Depends(require_login)])
    def update(background: BackgroundTasks) -> dict:
        """Fetch the newest version and restart, without opening a terminal.

        The alternative was telling someone to find a terminal, remember two
        commands and type them in the right folder to get a fix - which is a
        strange thing to ask of a person whose job is making videos.
        """
        import subprocess  # noqa: PLC0415
        repo = PACKAGE_ROOT.parent
        script = repo / ".devcontainer" / "restart.sh"
        if not (repo / ".git").exists():
            raise HTTPException(status_code=409,
                                detail="this copy was not installed from git, so it "
                                       "cannot update itself")
        # --ff-only: never invent a merge on a machine nobody is watching.
        pull = subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"],
                              capture_output=True, text=True)
        if pull.returncode != 0:
            detail = (pull.stderr or pull.stdout or "").strip().splitlines()
            raise HTTPException(status_code=409,
                                detail=f"could not update: {detail[-1] if detail else 'unknown'}")
        moved = "Already up to date" not in (pull.stdout or "")
        if not moved:
            return {"ok": True, "updated": False, "message": "already the newest version"}
        if not script.exists():
            return {"ok": True, "updated": True, "restarting": False,
                    "message": "updated - restart it to pick the new version up"}

        def restart() -> None:
            # After the response has gone, because this kills the process serving it.
            time.sleep(1.0)
            subprocess.Popen(["bash", str(script)], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        background.add_task(restart)
        return {"ok": True, "updated": True, "restarting": True,
                "message": "updated - reload the page in a few seconds"}

    @app.get("/api/templates", dependencies=[Depends(require_login)])
    def templates() -> list[dict]:
        out = []
        for directory in (PACKAGE_ROOT / "templates", PACKAGE_ROOT / "profiles"):
            for path in sorted(directory.glob("*.yml")):
                try:
                    profile = StyleProfile.load(path)
                except Exception:
                    continue
                out.append({"name": path.stem,
                            "description": profile.get("description", "")})
        return out

    # -- the b-roll library ---------------------------------------------
    def broll_or_404(name: str) -> Path:
        """Resolve a library file by name, refusing anything that escapes it."""
        safe = Path(name).name
        path = runner.broll_dir / safe
        # resolve() before comparing: a name is not trustworthy just because it
        # has no slashes in it, and the library is reachable over the network.
        try:
            inside = path.resolve().parent == runner.broll_dir.resolve()
        except OSError:
            inside = False
        if not safe or not inside or not path.is_file() or safe == broll.MANIFEST:
            raise HTTPException(status_code=404, detail=f"no clip called {safe}")
        return path

    @app.get("/api/broll", dependencies=[Depends(require_login)])
    def list_broll() -> list[dict]:
        manifest = broll.read_manifest(runner.broll_dir)
        for asset in broll.BrollLibrary.load(runner.broll_dir).assets:
            if not asset.is_image and not (manifest.get(asset.name) or {}).get("duration"):
                broll.remember_length(runner.broll_dir, asset.name, asset.path)
        manifest = broll.read_manifest(runner.broll_dir)
        library = broll.BrollLibrary.load(runner.broll_dir)
        out = []
        for asset in library.assets:
            entry = manifest.get(asset.name) or {}
            out.append({
                "name": asset.name,
                "kind": "image" if asset.is_image else "video",
                # What you typed, not what the matcher normalised it to - the
                # normalised form strips the vowels and reads like a typo.
                "keywords": list(entry.get("keywords") or asset.keywords),
                "from_filename": not entry.get("keywords"),
                "seconds": entry.get("duration"),
                "hold": entry.get("hold", "cutaway"),
                "size": asset.path.stat().st_size,
            })
        return out

    @app.post("/api/broll/chunk", dependencies=[Depends(require_login)])
    async def add_broll(file: UploadFile, name: str = Form(...),
                        offset: int = Form(0), final: str = Form("false")) -> dict:
        safe = Path(name).name
        if not broll.is_supported(safe):
            raise HTTPException(status_code=400,
                                detail=f"{safe} is not a video or an image")
        runner.broll_dir.mkdir(parents=True, exist_ok=True)
        target = runner.broll_dir / safe
        payload = await file.read()
        if offset + len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="that clip is too large")
        mode = "r+b" if target.exists() and offset else "wb"
        with target.open(mode) as handle:
            handle.seek(offset)
            handle.write(payload)
        done = str(final).lower() == "true"
        if done:
            broll.thumbnail(target, runner.broll_dir / ".thumbs" / f"{safe}.jpg")
            runner.broll_proxy(target)
            broll.remember_length(runner.broll_dir, safe, target)
            broll.remember_length(runner.broll_dir, safe, target)
        return {"ok": True, "complete": done, "size": target.stat().st_size}

    @app.post("/api/broll/{name}", dependencies=[Depends(require_login)])
    def set_broll_keywords(name: str, body: dict) -> dict:
        """The words that make this clip appear, and how long it stays."""
        path = broll_or_404(name)

        if "hold" in body:
            hold = body.get("hold")
            if hold not in ("cutaway", "full"):
                try:
                    hold = round(max(0.2, min(60.0, float(hold))), 2)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400,
                                        detail="how long must be a number, "
                                               "'cutaway' or 'full'") from exc
            manifest = broll.read_manifest(runner.broll_dir)
            entry = dict(manifest.get(path.name) or {})
            entry["hold"] = hold
            manifest[path.name] = entry
            broll.write_manifest(runner.broll_dir, manifest)
            if "keywords" not in body:
                return {"ok": True, "hold": hold}

        raw = body.get("keywords")
        if isinstance(raw, str):
            raw = raw.replace("،", ",").split(",")
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="keywords must be a list")
        keywords = [str(word).strip() for word in raw if str(word).strip()]
        manifest = broll.read_manifest(runner.broll_dir)
        entry = dict(manifest.get(path.name) or {})
        if keywords:
            entry["keywords"] = keywords
            manifest[path.name] = entry
        else:
            # No keywords means fall back to the filename, which is what an asset
            # with no entry already does - so drop the entry, keeping any other
            # setting it carries.
            entry.pop("keywords", None)
            if entry:
                manifest[path.name] = entry
            else:
                manifest.pop(path.name, None)
        broll.write_manifest(runner.broll_dir, manifest)
        return {"ok": True, "keywords": keywords}

    @app.delete("/api/broll/{name}", dependencies=[Depends(require_login)])
    def delete_broll(name: str) -> dict:
        path = broll_or_404(name)
        path.unlink()
        (runner.broll_dir / ".thumbs" / f"{path.name}.jpg").unlink(missing_ok=True)
        manifest = broll.read_manifest(runner.broll_dir)
        if manifest.pop(path.name, None) is not None:
            broll.write_manifest(runner.broll_dir, manifest)
        return {"deleted": True}

    @app.get("/api/broll/{name}/file", dependencies=[Depends(require_login)])
    def broll_file(name: str, request: Request) -> Response:
        """The clip, for laying over you in the preview.

        The small copy when there is one: this is streamed every time a clip
        comes up on screen, and the original can be a hundred megabytes of
        phone footage for a two-second cutaway.
        """
        path = broll_or_404(name)
        if path.suffix.lower() in broll.IMAGE_EXT:
            return ranged(path, request, "image/jpeg")
        return ranged(runner.broll_proxy(path) or path, request, "video/mp4")

    @app.get("/api/broll/{name}/thumb.jpg", dependencies=[Depends(require_login)])
    def broll_thumb(name: str, request: Request) -> Response:
        path = broll_or_404(name)
        thumb = runner.broll_dir / ".thumbs" / f"{path.name}.jpg"
        if not thumb.exists():                       # a library that predates thumbnails
            broll.thumbnail(path, thumb)
        if not thumb.exists():
            raise HTTPException(status_code=404, detail="no picture for this one")
        return ranged(thumb, request, "image/jpeg")

    @app.get("/api/jobs", dependencies=[Depends(require_login)])
    def list_jobs() -> list[dict]:
        # An upload that failed leaves an empty edit behind. Clear the stale ones
        # rather than letting them pile up looking like real work.
        for job in store.list():
            if (job.status == "uploading" and not job.sources
                    and time.time() - job.created > 600):
                store.delete(job.id)
        return [job.to_dict() for job in store.list()]

    @app.post("/api/jobs", dependencies=[Depends(require_login)])
    def create_job(body: dict) -> dict:
        """Open an empty edit. Clips are added one request each, then started."""
        job = store.create(title=str(body.get("title") or "reel"),
                           template=str(body.get("template") or ""),
                           model=str(body.get("model") or "small"))
        # Start from the look you settled on, so a new upload already arrives
        # the way you like it rather than back at the factory settings.
        store.update(job, status="uploading", stage="waiting for clips",
                     overrides=read_defaults())
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/chunk", dependencies=[Depends(require_login)])
    async def add_chunk(job_id: str, file: UploadFile, name: str = Form(...),
                        index: int = Form(...), offset: int = Form(0),
                        final: str = Form("false")) -> dict:
        """A slice of one clip, written at its offset.

        Phone video runs to hundreds of megabytes a take, and a single request
        that large dies somewhere between the browser and here - a proxy limit, a
        timeout, a dropped connection - with nothing useful to show for it.
        Small pieces always get through, and each one that does is progress kept.
        """
        job = job_or_404(job_id)
        safe = Path(name).name
        if Path(safe).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"{safe} is not a video file")
        if offset < 0 or index < 0:
            raise HTTPException(status_code=400, detail="bad chunk position")

        uploads = store.dir(job.id) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / f"{index:02d}-{safe}"

        payload = await file.read()
        if offset + len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="that clip is larger than 4 GB")

        # r+b keeps the earlier chunks; writing at the offset means a retried or
        # out-of-order piece lands in the right place rather than corrupting it.
        mode = "r+b" if target.exists() else "wb"
        with target.open(mode) as handle:
            handle.seek(offset)
            handle.write(payload)

        done = str(final).lower() == "true"
        if done and str(target) not in job.sources:
            store.update(job, sources=[*job.sources, str(target)],
                         stage=f"{len(job.sources) + 1} clip(s) uploaded")
        return {"ok": True, "received": len(payload), "size": target.stat().st_size,
                "complete": done}

    @app.post("/api/jobs/{job_id}/start", dependencies=[Depends(require_login)])
    def start_job(job_id: str) -> dict:
        job = job_or_404(job_id)
        if not job.sources:
            raise HTTPException(status_code=400, detail="no clips were uploaded")
        job.title = Path(job.sources[0]).stem.split("-", 1)[-1] or job.title
        store.update(job, status="queued", stage="queued")
        runner.submit(job.id)
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/stop", dependencies=[Depends(require_login)])
    def stop_job(job_id: str) -> dict:
        """Give up on a job that is going nowhere, so it can be tried again."""
        job = job_or_404(job_id)
        if job.status not in ("working", "queued", "exporting"):
            raise HTTPException(status_code=409, detail="that one is not running")
        return {"ok": True, "killed": runner.stop(job)}

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_login)])
    def get_job(job_id: str) -> dict:
        return job_or_404(job_id).to_dict()

    @app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_login)])
    def delete_job(job_id: str) -> dict:
        return {"deleted": store.delete(job_id)}

    @app.get("/api/jobs/{job_id}/edl", dependencies=[Depends(require_login)])
    def get_edl(job_id: str) -> dict:
        edl = runner.edl_for(job_or_404(job_id))
        if edl is None:
            raise HTTPException(status_code=409, detail="not planned yet")
        return edl.to_dict()

    @app.post("/api/jobs/{job_id}/edl", dependencies=[Depends(require_login)])
    def update_edl(job_id: str, body: dict) -> dict:
        from .captions import apply_text_edit  # noqa: PLC0415
        job = job_or_404(job_id)
        edl = runner.edl_for(job)
        if edl is None:
            raise HTTPException(status_code=409, detail="not planned yet")

        # Segments first: dropping one moves everything that comes after it, so
        # the per-effect flags below must be applied to the retimed edit.
        cuts = body.get("cuts")
        if cuts is not None:
            wanted = {cut.id: bool(item.get("enabled", True))
                      for cut, item in zip(edl.cuts, cuts)}
            if not any(wanted.get(c.id, c.enabled) and c.duration > 0.01 for c in edl.cuts):
                raise HTTPException(status_code=400,
                                    detail="that would remove the whole video")
            edl.retime(wanted)

        for index, item in enumerate(body.get("zooms", [])):
            if index < len(edl.zooms):
                edl.zooms[index].enabled = bool(item.get("enabled", True))
        for index, item in enumerate(body.get("overlays", [])):
            if index < len(edl.overlays):
                edl.overlays[index].enabled = bool(item.get("enabled", True))
        for index, item in enumerate(body.get("transitions", [])):
            if index < len(edl.transitions):
                edl.transitions[index].enabled = bool(item.get("enabled", True))
        for index, item in enumerate(body.get("captions", [])):
            if index < len(edl.captions):
                text = (item.get("text") or "").strip()
                if text and text != edl.captions[index].text:
                    apply_text_edit(edl.captions[index], text)

        runner.keep(job.id)
        if body.get("rerender"):
            store.update(job, status="working", stage="queued for re-render")
            runner.submit(job.id, "rerender")
        return {"ok": True, "queued": bool(body.get("rerender")),
                "summary": edl.summary()}

    @app.post("/api/jobs/{job_id}/segments", dependencies=[Depends(require_login)])
    def update_segments(job_id: str, body: dict) -> dict:
        """Trim the video: keep some segments, drop others, re-decide the rest."""
        job = job_or_404(job_id)
        if job.status in ("working", "queued", "exporting"):
            raise HTTPException(status_code=409, detail="this edit is still busy")
        edl = runner.edl_for(job)
        if edl is None:
            raise HTTPException(status_code=409, detail="not planned yet")

        keep = body.get("keep")
        if not isinstance(keep, dict):
            raise HTTPException(status_code=400, detail="no segments were sent")
        known = {cut.id: cut for cut in edl.cuts}
        unknown = sorted(set(keep) - set(known))
        if unknown:
            raise HTTPException(status_code=400, detail=f"no segment called {unknown[0]}")
        if not any(bool(keep.get(cut_id, cut.enabled)) for cut_id, cut in known.items()):
            raise HTTPException(status_code=400, detail="that would remove the whole video")

        # Stored as source ranges, not as the ids the browser sent, so the trim
        # still means the same thing after the next plan renumbers the segments.
        drops = [[cut.src_start, cut.src_end] for cut_id, cut in known.items()
                 if not bool(keep.get(cut_id, cut.enabled))]
        store.update(job, drops=drops, status="queued", stage="queued for the trim",
                     error="")
        runner.submit(job.id, "replan")
        return {"ok": True, "queued": True, "dropped": len(drops)}

    @app.get("/api/jobs/{job_id}/settings", dependencies=[Depends(require_login)])
    def get_settings(job_id: str) -> dict:
        """The look panel: what can be changed, and where it stands right now."""
        from . import fonts as font_catalog  # noqa: PLC0415
        job = job_or_404(job_id)
        profile = runner.profile_for(job)
        have = font_catalog.installed(runner.fonts_dir)
        fields = []
        for field_def in LOOK_FIELDS:
            entry = dict(field_def)
            entry["value"] = profile.get(entry["key"], None)
            if entry["type"] == "select":
                entry["options"] = [{"value": v, "label": label}
                                    for v, label in entry["options"]]
                # 'none' round-trips through the profile as a real None, which no
                # <option value> can match. Show it as the word again.
                if entry["value"] is None:
                    entry["value"] = "none"
            elif entry["type"] == "font":
                entry["type"] = "select"
                entry["options"] = [
                    {"value": f.family,
                     "label": f"{f.family} — {f.note}" if f.family in have
                              else f"{f.family} (downloads) — {f.note}"}
                    for f in font_catalog.CATALOG
                ]
            fields.append(entry)
        changed = sorted(LOOK_LABELS.get(key, key) for key in (job.overrides or {}))
        return {"fields": fields, "changed": changed}

    @app.post("/api/jobs/{job_id}/settings", dependencies=[Depends(require_login)])
    def update_settings(job_id: str, body: dict) -> dict:
        """Change the look and re-decide the edit.

        Only keys the panel offers are accepted. The profile is a nest of numbers
        that drive ffmpeg expressions, and 'whatever the browser posted' is not
        something to hand to a filtergraph.
        """
        job = job_or_404(job_id)
        if job.status in ("working", "queued", "exporting"):
            raise HTTPException(status_code=409, detail="this edit is still busy")
        if not job.prepared:
            raise HTTPException(status_code=409, detail="nothing has been edited yet")

        values = body.get("values")
        if body.get("reset"):
            overrides: dict = {}
        else:
            if not isinstance(values, dict):
                raise HTTPException(status_code=400, detail="no settings were sent")
            unknown = sorted(set(values) - LOOK_KEYS)
            if unknown:
                raise HTTPException(status_code=400,
                                    detail=f"not a setting you can change here: {unknown[0]}")
            posted = {key: str(value) for key, value in values.items()}
            base = StyleProfile.resolve(job.template or None,
                                        [PACKAGE_ROOT / "templates", PACKAGE_ROOT / "profiles"])
            try:                                   # reject a bad number before it reaches ffmpeg
                candidate = base.apply_overrides([f"{k}={v}" for k, v in posted.items()])
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            overrides = {key: value for key, value in posted.items()
                         if _differs(candidate.get(key, None), base.get(key, None))}

        store.update(job, overrides=overrides, status="queued",
                     stage="queued for the new look", error="")
        runner.submit(job.id, "replan")
        return {"ok": True, "queued": True}

    @app.post("/api/jobs/{job_id}/export", dependencies=[Depends(require_login)])
    def export_job(job_id: str) -> dict:
        job = job_or_404(job_id)
        if runner.edl_for(job) is None:
            raise HTTPException(status_code=409,
                                detail="this edit is no longer loaded - upload it again")
        store.update(job, status="exporting", stage="queued for export")
        runner.submit(job_id, "export")
        return {"ok": True, "queued": True}

    @app.post("/api/jobs/{job_id}/preview-plan", dependencies=[Depends(require_login)])
    def preview_plan(job_id: str, body: dict) -> dict:
        """Try settings and trims without keeping them. Nothing is stored."""
        job = job_or_404(job_id)
        values = body.get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="settings must be an object")
        unknown = sorted(set(values) - LOOK_KEYS)
        if unknown:
            raise HTTPException(status_code=400,
                                detail=f"not a setting you can change here: {unknown[0]}")
        try:
            return runner.plan_preview(job, {k: str(v) for k, v in values.items()},
                                       body.get("drops") or job.drops,
                                       body.get("pauses"), body.get("beats"))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/trim", dependencies=[Depends(require_login)])
    def trim(job_id: str, body: dict) -> dict:
        """Turn selections made while watching into trims of the original footage.

        The browser hands back spans of the *finished* video, because that is what
        it was playing. Stored as source ranges, for the same reason segment ids
        are not: source time is the one frame of reference that survives the next
        plan.
        """
        job = job_or_404(job_id)
        edl = runner.edl_for(job)
        if edl is None:
            raise HTTPException(status_code=409, detail="not planned yet")
        if body.get("reset"):
            return {"ok": True, "drops": []}

        ranges = body.get("ranges")
        if not isinstance(ranges, list):
            raise HTTPException(status_code=400, detail="no selection was sent")
        timeline = edl.timeline
        drops = [list(span) for span in (job.drops or [])]
        for item in ranges:
            try:
                start, end = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise HTTPException(status_code=400, detail="a selection was malformed") from exc
            if end <= start:
                continue
            drops += [[low, high] for low, high in timeline.to_source_spans(start, end)]
        kept = sum(max(0.0, high - low) for low, high in
                   _remaining(edl, drops))
        if kept < 0.4:
            raise HTTPException(status_code=400, detail="that would remove the whole video")
        return {"ok": True, "drops": drops}

    @app.post("/api/jobs/{job_id}/save", dependencies=[Depends(require_login)])
    def save_edit(job_id: str, body: dict) -> dict:
        """Keep what is on screen. Until this, nothing you tried was written down."""
        job = job_or_404(job_id)
        if job.status in ("working", "queued", "exporting"):
            raise HTTPException(status_code=409, detail="this edit is still busy")
        values = body.get("values")
        if values is not None:
            if not isinstance(values, dict):
                raise HTTPException(status_code=400, detail="settings must be an object")
            unknown = sorted(set(values) - LOOK_KEYS)
            if unknown:
                raise HTTPException(status_code=400,
                                    detail=f"not a setting you can change here: {unknown[0]}")
            base = StyleProfile.resolve(job.template or None,
                                        [PACKAGE_ROOT / "templates", PACKAGE_ROOT / "profiles"])
            posted = {key: str(value) for key, value in values.items()}
            try:
                candidate = base.apply_overrides([f"{k}={v}" for k, v in posted.items()])
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            job.overrides = {key: value for key, value in posted.items()
                             if _differs(candidate.get(key, None), base.get(key, None))}

        drops = body.get("drops")
        if drops is not None:
            if not isinstance(drops, list):
                raise HTTPException(status_code=400, detail="drops must be a list")
            job.drops = [[float(a), float(b)] for a, b in drops if float(b) > float(a)]

        pauses = body.get("pauses")
        if pauses is not None:
            if not isinstance(pauses, list):
                raise HTTPException(status_code=400, detail="pauses must be a list")
            job.pauses = [[float(e[0]), float(e[1]),
                           str(e[2]) if len(e) >= 3 else "both"]
                          for e in pauses if abs(float(e[1])) > 0.005]

        beats = body.get("beats")
        if beats is not None:
            if not isinstance(beats, list):
                raise HTTPException(status_code=400, detail="beats must be a list")
            job.beats = [[float(mid), float(seconds)] for mid, seconds in beats]

        store.update(job, status="queued", stage="saving", error="")
        runner.submit(job.id, "replan" if body.get("render") else "replan_only")
        return {"ok": True, "queued": True, "render": bool(body.get("render"))}

    @app.get("/api/jobs/{job_id}/proxy.mp4", dependencies=[Depends(require_login)])
    def proxy(job_id: str, request: Request) -> Response:
        """The untouched footage, small. The browser plays the edit over this."""
        job = job_or_404(job_id)
        path = runner._ensure_proxy(job)
        if path is None:
            raise HTTPException(status_code=409,
                                detail="the footage for this edit is no longer here")
        return ranged(path, request, "video/mp4")

    @app.get("/api/fonts/{family}", dependencies=[Depends(require_login)])
    def font_file(family: str, request: Request) -> Response:
        """The caption font itself, so the live overlay uses the real face."""
        entry = font_catalog_resolve(family)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no font called {family}")
        path = runner.fonts_dir / entry.filename
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{family} is not downloaded")
        return ranged(path, request, "font/ttf")

    @app.get("/api/jobs/{job_id}/preview.mp4", dependencies=[Depends(require_login)])
    def preview(job_id: str, request: Request) -> Response:
        job_or_404(job_id)
        return ranged(store.dir(job_id) / "preview.mp4", request, "video/mp4")

    @app.get("/api/jobs/{job_id}/download", dependencies=[Depends(require_login)])
    def download(job_id: str, request: Request) -> Response:
        job = job_or_404(job_id)
        if not job.output_name:
            raise HTTPException(status_code=409, detail="not exported yet")
        path = store.dir(job_id) / job.output_name
        response = ranged(path, request, "video/mp4")
        response.headers["Content-Disposition"] = f'attachment; filename="{job.output_name}"'
        return response

    return app


def serve(host: str = "0.0.0.0", port: int = 8000, data_dir: str | Path = "data",
          password: str | None = None) -> None:
    import uvicorn  # noqa: PLC0415

    app = create_app(data_dir=data_dir, password=password)
    shown = app.state.password
    print()
    print("  ReelForge is running")
    print(f"    address : http://{host}:{port}/")
    print(f"    password: {shown}")
    if not os.environ.get("REELFORGE_PASSWORD"):
        print("    (generated for this run - set REELFORGE_PASSWORD to keep one)")
    print()
    uvicorn.run(app, host=host, port=port, log_level="warning")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>ReelForge</title>
<style>
:root{--bg:#0d0d13;--card:#16161f;--line:#262633;--fg:#f2f2f7;--dim:#9a9aae;--accent:#ffd24a;--ok:#3ddc97;--bad:#ff6b81}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  padding:16px;padding-bottom:calc(16px + env(safe-area-inset-bottom));max-width:820px;margin:0 auto}
h1{font-size:19px;margin-bottom:4px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:14px}
.dim{color:var(--dim);font-size:13px}
button{background:var(--accent);color:#18181f;border:0;border-radius:11px;padding:13px 18px;
  font-weight:650;font-size:15px;font-family:inherit;cursor:pointer;width:100%;margin-top:10px}
button.ghost{background:#242433;color:var(--fg)}
button:disabled{opacity:.5}
input,select{width:100%;background:#10101a;border:1px solid var(--line);color:var(--fg);
  border-radius:10px;padding:12px;font-size:16px;font-family:inherit;margin-top:8px}
input[type=file]{padding:10px}
video{width:100%;border-radius:12px;background:#000;display:block;margin-bottom:10px}
.row{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row .t{color:var(--dim);font-variant-numeric:tabular-nums;font-size:12px;min-width:56px}
.grow{flex:1;min-width:0}
input[type=checkbox]{width:22px;height:22px;accent-color:var(--accent);flex:none;margin:0}
.job{display:flex;align-items:center;gap:10px;padding:11px 0;border-bottom:1px solid var(--line);cursor:pointer}
.job:last-child{border-bottom:0}
.pill{font-size:11px;padding:3px 9px;border-radius:20px;background:#242433;color:var(--dim);flex:none}
.pill.ready{background:#123026;color:var(--ok)}
.pill.error{background:#2f1620;color:var(--bad)}
.pill.working{background:#2e2a12;color:var(--accent)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}
.stat{background:#1d1d28;border-radius:10px;padding:9px}
.stat b{display:block;font-size:17px}.stat span{font-size:11px;color:var(--dim)}
.log{font:12px/1.5 ui-monospace,Menlo,monospace;color:var(--dim);max-height:130px;overflow:auto;
  background:#10101a;border-radius:10px;padding:10px;margin-top:10px;white-space:pre-wrap}
#msg{margin-top:10px;font-size:13px;color:var(--dim);min-height:18px}
a{color:var(--accent)}
.set{padding:10px 0;border-bottom:1px solid var(--line)}
.set:last-child{border-bottom:0}
.set label{font-size:14px;display:block}
.set .help{font-size:12px;color:var(--dim);margin-top:2px}
.set input,.set select{margin-top:6px}
.set.inline{display:flex;align-items:center;gap:10px}
.set.inline label{flex:1}
.set.inline input[type=checkbox]{margin-top:0}
input[type=color]{height:44px;padding:4px}
.tabs{display:flex;gap:6px;margin-bottom:10px;overflow-x:auto}
.tab{background:#242433;color:var(--dim);border:0;border-radius:9px;padding:8px 13px;
  font:inherit;font-size:13px;cursor:pointer;width:auto;margin:0;flex:none}
.tab.on{background:var(--accent);color:#18181f;font-weight:650}
.two{display:flex;gap:10px}.two>*{flex:1}
.strip{display:flex;gap:2px;height:30px;margin-bottom:12px;border-radius:7px;overflow:hidden}
.strip div{background:var(--accent);min-width:2px;transition:opacity .15s}
.strip div.off{background:#39394a}
.seg{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid var(--line)}
.seg:last-child{border-bottom:0}
.seg.off .txt{opacity:.4;text-decoration:line-through}
.seg .txt{font-size:14px;word-break:break-word}
.asset{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--line)}
.asset:last-child{border-bottom:0}
.asset img{width:58px;height:58px;object-fit:cover;border-radius:8px;background:#10101a;flex:none}
.asset .no{width:58px;height:58px;border-radius:8px;background:#10101a;flex:none}
.asset input[type=text]{margin-top:4px;padding:8px;font-size:15px}
.asset .nm{font-size:12px;color:var(--dim);word-break:break-all}
.stage{position:relative;border-radius:12px;overflow:hidden;background:#000;
  aspect-ratio:9/16;max-height:62vh;margin:0 auto}
.stage video{width:100%;height:100%;object-fit:contain;display:block;
  transform-origin:center center;will-change:transform,filter}
#bv,#bi{position:absolute;object-fit:cover;background:#000;pointer-events:none}
/* The b-roll video sits inside .stage, and `.stage video{display:block}` above
   outranks the browser's own `[hidden]{display:none}` - so `hidden` did nothing,
   and a clip that had finished stayed over the footage to the end of the video.
   Three fixes went past it because they checked the property, not the screen. */
#bv[hidden],#bi[hidden]{display:none!important}
#bv.cover,#bi.cover{inset:0;width:100%;height:100%}
#bv.pip,#bi.pip{right:4%;top:6%;width:42%;height:26%;border-radius:10px;
  box-shadow:0 6px 24px rgba(0,0,0,.5)}
#bv.band,#bi.band{left:0;right:0;top:32%;width:100%;height:32%}
#caps{position:absolute;left:0;right:0;pointer-events:none;text-align:center;
  padding:0 5%;line-height:1.25;white-space:pre-wrap;word-break:break-word}
#caps span{transition:color .08s linear}
.transport{display:flex;align-items:center;gap:10px;margin-top:10px}
.transport button{width:auto;margin:0;padding:10px 16px;flex:none}
.clock{font:12px ui-monospace,Menlo,monospace;color:var(--dim);flex:1;text-align:right}
.track{position:relative;height:52px;margin-top:10px;background:#10101a;border-radius:9px;
  overflow:hidden;touch-action:none;cursor:crosshair;user-select:none}
.track .keep{position:absolute;top:0;bottom:0;background:#2b2b3d}
.track .sel{position:absolute;top:0;bottom:0;background:rgba(255,210,74,.28);
  border-left:2px solid var(--accent);border-right:2px solid var(--accent)}
.track .keeps,.track .joins{position:absolute;inset:0}
.track .grip{position:absolute;top:0;bottom:0;width:28px;margin-left:-14px;
  cursor:ew-resize;touch-action:none;z-index:3}
.track .grip::after{content:'';position:absolute;top:50%;left:12px;width:4px;height:24px;
  margin-top:-12px;border-radius:2px;background:var(--accent)}
.track .head{position:absolute;top:0;bottom:0;width:2px;background:#fff;z-index:4;
  touch-action:none;cursor:grab}
.track .head::before{content:'';position:absolute;top:0;bottom:0;left:-14px;right:-14px}
.track .head::after{content:'';position:absolute;top:-1px;left:-5px;width:12px;height:12px;
  border-radius:50%;background:#fff;box-shadow:0 0 0 2px rgba(0,0,0,.4)}
.track .lbl{position:absolute;bottom:3px;left:6px;right:6px;font:10px ui-monospace,monospace;
  color:var(--dim);pointer-events:none;text-overflow:ellipsis;overflow:hidden;white-space:nowrap}
.track .join{position:absolute;top:0;bottom:0;width:30px;margin-left:-15px;display:none;
  touch-action:none;cursor:ew-resize;z-index:2}
.track.pauses .join{display:block}
.track .join i{position:absolute;top:50%;width:4px;height:26px;margin-top:-13px;
  border-radius:2px;background:#6f6fa8}
.track .join.left i{left:15px}
.track .join.right i{left:11px}
.track .join.on i{background:var(--accent);box-shadow:0 0 0 3px rgba(255,210,74,.2)}
.tools button.on{background:var(--accent);color:#18181f}
#joinInfo{background:#10101a;border-radius:10px;padding:10px;margin-top:8px;font-size:13px}
#joinInfo .row{display:flex;align-items:center;gap:10px;padding:4px 0;border:0}
#joinInfo input[type=range]{flex:1;accent-color:var(--accent);margin:0}
#joinInfo b{font:12px ui-monospace,monospace;color:var(--accent);min-width:104px;text-align:right}
.iconbar{display:flex;gap:6px;overflow-x:auto;margin:12px 0 0;padding-bottom:4px;
  -webkit-overflow-scrolling:touch}
.iconbar button{width:auto;margin:0;flex:none;background:#242433;color:var(--dim);
  border-radius:11px;padding:9px 12px;font-size:11px;line-height:1.25;min-width:64px}
.iconbar button b{display:block;font-size:19px;font-weight:400;margin-bottom:2px}
.iconbar button.on{background:var(--accent);color:#18181f;font-weight:650}
.tools{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
.tools button{width:auto;margin:0;padding:9px 13px;font-size:13px;flex:none}
.bar{display:flex;gap:8px;margin-top:12px}.bar>*{flex:1}
.unsaved{color:var(--accent);font-size:13px;margin-top:8px;min-height:18px}
.bar-outer{height:8px;background:#10101a;border-radius:6px;overflow:hidden;margin:10px 0 6px}
.bar-inner{height:100%;background:var(--accent);width:0;transition:width .4s ease}
.bar-inner.stalled{background:var(--bad)}
.meta{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}
#offline{background:#2f1620;color:var(--bad);border-radius:10px;padding:10px;
  margin-bottom:10px;font-size:13px}
</style></head><body>

<div id="login" hidden>
  <div class="card">
    <h1>ReelForge</h1>
    <div class="dim">Enter the password shown when the server started.</div>
    <input type="password" id="pw" placeholder="password" autocomplete="current-password">
    <button id="loginBtn">Open</button>
    <div id="loginMsg" class="dim"></div>
  </div>
</div>

<div id="app" hidden>
  <div id="offline" hidden></div>
  <div class="card">
    <h1>New edit</h1>
    <div class="dim" style="font-size:11px;margin-bottom:6px">
      <span id="ver"></span>
      <a href="#" id="updateLink" style="margin-left:8px">check for an update</a>
      <span id="updateMsg"></span>
    </div>
    <div id="newVersion" hidden style="background:#2e2a12;color:var(--accent);
      border-radius:10px;padding:10px;margin-bottom:10px;font-size:13px">
      <span id="newVersionText"></span>
      <button id="updateNow" style="margin-top:8px">Update now</button>
    </div>
    <div class="dim">Pick every take of one video. They are joined in the order chosen.</div>
    <input type="file" id="files" accept="video/*" multiple>
    <select id="template"></select>
    <select id="model">
      <option value="small">small — a few minutes, good Arabic</option>
      <option value="medium">medium — slower, better Arabic</option>
      <option value="large-v3">large-v3 — best Arabic, but very slow here</option>
    </select>
    <div class="dim" id="modelNote" style="font-size:12px;margin-top:6px"></div>
    <button id="upload">Upload and edit</button>
    <div id="msg"></div>
  </div>

  <div class="card">
    <h2>Edits</h2>
    <div id="jobs" class="dim">none yet</div>
  </div>

  <div class="card">
    <h2>B-roll library</h2>
    <div class="dim">Your own clips and stills. When you say a word one of them is
      tagged with, it is cut in over you. Give each one the words that should
      bring it up — a clip with no words never appears.</div>
    <input type="file" id="brollFiles" accept="video/*,image/*" multiple>
    <button class="ghost" id="brollUpload">Add to the library</button>
    <div id="brollMsg" class="dim"></div>
    <div id="broll" style="margin-top:10px"></div>
  </div>

  <div id="detail"></div>
</div>

<script>
const $=id=>document.getElementById(id);
let current=null, edl=null, poll=null, drawn='', listed='', misses=0;
const fmt=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;

async function api(path, opts={}){
  let r;
  try{ r = await fetch(path, {credentials:'same-origin', ...opts}); }
  catch(e){ throw new Error('could not reach the server ('+(e.message||'connection lost')+')'); }
  if(r.status===401){ show(false); throw new Error('please log in'); }
  if(!r.ok){
    const text = await r.text().catch(()=>'');
    let detail='';
    try{ detail = JSON.parse(text).detail || ''; }catch(_){ detail = text.slice(0,180); }
    throw new Error(`${r.status} ${detail || r.statusText || 'request failed'}`);
  }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r;
}

const CHUNK = 6 * 1024 * 1024;   // small enough to cross any proxy

function postChunk(url, blob, fields){
  return new Promise((resolve, reject)=>{
    const xhr=new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.withCredentials=true;
    xhr.timeout=180000;
    xhr.onload=()=>{
      if(xhr.status>=200 && xhr.status<300) return resolve();
      let detail=''; try{ detail=JSON.parse(xhr.responseText).detail||''; }catch(_){}
      reject(new Error(`${xhr.status} ${detail||xhr.statusText||'upload failed'}`));
    };
    xhr.onerror=()=>reject(new Error('the connection dropped'));
    xhr.ontimeout=()=>reject(new Error('this piece timed out'));
    const body=new FormData();
    for(const [k,v] of Object.entries(fields)) body.append(k, v);
    body.append('file', blob, fields.name);
    xhr.send(body);
  });
}

async function uploadFile(jobId, file, index, onProgress){
  // Send the clip in pieces. One failed piece is retried on its own, instead of
  // losing a 150 MB upload and starting again.
  for(let offset=0; offset<file.size; offset+=CHUNK){
    const slice=file.slice(offset, Math.min(offset+CHUNK, file.size));
    const last = offset+CHUNK >= file.size;
    let attempt=0;
    for(;;){
      try{
        await postChunk(`/api/jobs/${jobId}/chunk`, slice,
          {name:file.name, index:String(index), offset:String(offset),
           final:last?'true':'false'});
        break;
      }catch(e){
        if(++attempt>=3) throw new Error(`${file.name}: ${e.message}`);
        await new Promise(r=>setTimeout(r, 1000*attempt));
      }
    }
    onProgress(Math.min(offset+CHUNK, file.size)/file.size);
  }
}
function show(authed){ $('login').hidden=authed; $('app').hidden=!authed;
  if(authed){loadTemplates();refresh();loadBroll();} }

// -- the b-roll library ------------------------------------------------------
async function loadBroll(){
  let assets=[];
  try{ assets=await api('/api/broll'); }catch(e){ return; }
  // Silence is the worst answer here: a tagged clip that never appears looks
  // identical to a broken feature. Say whether the open edit ever says the word.
  const spoken = P && P.plan
    ? new Set((P.plan.captions||[]).flatMap(l=>(l.text||'').split(/\s+/)))
    : null;
  const verdict = a => {
    if(!spoken) return '';
    const words=(a.keywords||[]);
    if(!words.length) return ' · no words yet — it will never appear';
    const hit=words.some(w=>[...spoken].some(s=>s.includes(w)||w.includes(s)));
    return hit ? ' · said in this edit' : ' · not said in this edit';
  };
  $('broll').innerHTML = assets.length ? assets.map(a=>`
    <div class="asset">
      <img src="/api/broll/${encodeURIComponent(a.name)}/thumb.jpg" alt=""
        onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'no'}))">
      <span class="grow">
        <span class="nm">${a.name}${a.from_filename?' · words from the filename':''}${verdict(a)}</span>
        <input type="text" dir="auto" data-kw="${a.name}"
          value="${(a.keywords||[]).join(', ').replace(/"/g,'&quot;')}"
          placeholder="words that bring this up, separated by commas">
        <select data-hold="${a.name}">
          <option value="cutaway" ${a.hold==='cutaway'?'selected':''}>
            short cutaway — a couple of seconds</option>
          ${a.kind==='video'?`<option value="full" ${a.hold==='full'?'selected':''}>
            play the whole clip${a.seconds?` — ${a.seconds.toFixed(1)}s`:''}</option>`:''}
          ${[2,3,5,8,12].map(n=>`<option value="${n}" ${Number(a.hold)===n?'selected':''}>
            hold for ${n}s</option>`).join('')}
        </select>
      </span>
      <span class="pill" data-rm="${a.name}" title="remove">✕</span>
    </div>`).join('') : '<span class="dim">empty — nothing will be cut in yet</span>';

  $('broll').querySelectorAll('[data-hold]').forEach(el=>{
    el.onchange=async()=>{
      el.disabled=true;
      try{
        await api('/api/broll/'+encodeURIComponent(el.dataset.hold),{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({hold:el.value})});
        $('brollMsg').textContent='saved';
      }catch(e){ $('brollMsg').textContent='error: '+e.message; }
      el.disabled=false;
      refreshBroll();
    };
  });
  $('broll').querySelectorAll('[data-kw]').forEach(el=>{
    el.onchange=async()=>{
      el.disabled=true;
      try{
        await api('/api/broll/'+encodeURIComponent(el.dataset.kw),{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({keywords:el.value})});
        $('brollMsg').textContent='saved';
        refreshBroll();
      }catch(e){ $('brollMsg').textContent='error: '+e.message; }
      el.disabled=false;
    };
  });
  $('broll').querySelectorAll('[data-rm]').forEach(el=>{
    el.onclick=async()=>{
      await api('/api/broll/'+encodeURIComponent(el.dataset.rm),{method:'DELETE'}).catch(()=>{});
      refreshBroll();
    };
  });
}

// The library is read when the edit is decided, so a clip added afterwards
// changes nothing until it is decided again. Doing that quietly here is the
// difference between the feature working and the feature looking broken.
function refreshBroll(){
  loadBroll();
  if(P){ markDirty(); repaintPlan(); }
}

$('brollUpload').onclick=async()=>{
  const files=[...$('brollFiles').files];
  if(!files.length){ $('brollMsg').textContent='choose a clip or a picture'; return; }
  $('brollUpload').disabled=true;
  try{
    for(let i=0;i<files.length;i++){
      const f=files[i];
      for(let offset=0; offset<f.size; offset+=CHUNK){
        const last = offset+CHUNK >= f.size;
        await postChunk('/api/broll/chunk', f.slice(offset, Math.min(offset+CHUNK, f.size)),
          {name:f.name, offset:String(offset), final:last?'true':'false'});
        $('brollMsg').textContent=
          `adding ${i+1} of ${files.length} — ${Math.round(100*Math.min(offset+CHUNK,f.size)/f.size)}%`;
      }
    }
    $('brollFiles').value='';
    $('brollMsg').textContent='added — now give each one its words';
    refreshBroll();
  }catch(e){ $('brollMsg').textContent='error: '+e.message; }
  finally{ $('brollUpload').disabled=false; }
};

$('loginBtn').onclick=async()=>{
  try{
    await api('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({password:$('pw').value})});
    show(true);
  }catch(e){ $('loginMsg').textContent=e.message; }
};
$('pw').addEventListener('keydown',e=>{ if(e.key==='Enter') $('loginBtn').click(); });

async function loadTemplates(){
  const list=await api('/api/templates');
  $('template').innerHTML='<option value="">default look</option>'+
    list.map(t=>`<option value="${t.name}">${t.name} — ${(t.description||'').slice(0,54)}</option>`).join('');
}

// Said before the choice, not discovered after ten minutes of staring at a bar.
$('model').onchange=()=>{
  const slow={'large-v3':'roughly 10-20 minutes per minute of talking on a machine '
                        +'with no graphics card — it goes quiet for minutes at a time',
              'medium':'roughly 4-8 minutes per minute of talking here'};
  $('modelNote').textContent = slow[$('model').value] || '';
};

$('upload').onclick=async()=>{
  const files=[...$('files').files];
  if(!files.length){ $('msg').textContent='choose at least one video'; return; }
  const total=files.reduce((n,f)=>n+f.size,0);
  $('upload').disabled=true;
  let job=null;
  try{
    job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({template:$('template').value, model:$('model').value,
                           title:files[0].name.replace(/\.[^.]+$/,'')})});
    let done=0;
    for(let i=0;i<files.length;i++){
      const f=files[i];
      await uploadFile(job.id, f, i, frac=>{
        const pct=Math.round(100*(done+frac*f.size)/total);
        $('msg').textContent=`uploading ${i+1} of ${files.length} — ${f.name} — ${pct}%`;
      });
      done+=f.size;
    }
    $('msg').textContent='uploaded — editing has started';
    await api(`/api/jobs/${job.id}/start`,{method:'POST'});
    $('files').value=''; current=job.id; drawn=''; listed=''; refresh();
  }catch(e){
    $('msg').textContent='error: '+e.message;
    // Do not leave a half-made edit sitting in the list.
    if(job) { try{ await api('/api/jobs/'+job.id,{method:'DELETE'}); refresh(); }catch(_){} }
  }
  finally{ $('upload').disabled=false; }
};

const ago=s=>s<60?`${Math.round(s)}s`:`${Math.floor(s/60)}m ${Math.round(s%60)}s`;

function updateProgress(job){
  const bar=$('bar');
  if(!bar) return;
  const pct=Math.max(0, Math.min(100, job.percent||0));
  bar.style.width=pct.toFixed(1)+'%';
  const now=Date.now()/1000;
  const silent=job.heartbeat ? now-job.heartbeat : 0;
  const running=job.started ? now-job.started : 0;
  $('pct').textContent=`${pct.toFixed(0)}% · running ${ago(running)}`;
  // Ninety seconds without a word is the difference between slow and stuck. It
  // is a long time on purpose: transcribing a long take goes quiet for a while.
  // How long quiet is normal depends on the step - the machine says so, because
  // it is the one that knows which step it is on.
  const patience=job.patience||90;
  const stuck=silent>patience;
  bar.classList.toggle('stalled', stuck);
  $('alive').textContent = stuck
    ? `nothing for ${ago(silent)} — this looks stuck now`
    : (silent>20 ? `thinking — quiet for ${ago(silent)}, which is normal here`
                 : (silent>4 ? `last step ${ago(silent)} ago` : 'working'));
}

async function refresh(){
  let jobs=null;
  try{ jobs=await api('/api/jobs'); }
  catch(e){
    // Losing the server used to end the polling for good, so the page sat on
    // whatever it last saw - usually the word "working" - and never moved
    // again, for a job that had long since finished or died. Say so, and keep
    // trying.
    misses++;
    $('offline').hidden = misses < 2;
    $('offline').textContent =
      `Cannot reach the machine (${misses} tries). It may be restarting or asleep — `
      + `this page will pick up again by itself.`;
    clearTimeout(poll);
    poll=setTimeout(refresh, Math.min(15000, 2000*misses));
    return;
  }
  misses=0; $('offline').hidden=true;

  const listSig = jobs.map(j=>j.id+j.status+j.stage).join('|');
  if(listSig!==listed){ listed=listSig; renderJobs(jobs); }
  const job = current ? jobs.find(j=>j.id===current) : null;
  if(job){ draw(job); updateProgress(job); }
  // Keep polling while something is happening. Re-rendering an idle page
  // rebuilds the video element, which restarts whatever you were watching - so
  // once everything is finished, stop.
  const busy = jobs.some(j=>['working','queued','exporting','uploading'].includes(j.status));
  clearTimeout(poll);
  if(busy) poll=setTimeout(refresh, 2000);
}

function renderJobs(jobs){
  $('jobs').innerHTML = jobs.length ? jobs.map(j=>{
    const cls = j.status==='ready'||j.status==='done' ? 'ready' : (j.status==='error'?'error':'working');
    return `<div class="job" data-id="${j.id}">
      <span class="grow"><b>${j.title}</b><br><span class="dim">${j.stage}</span></span>
      <span class="pill ${cls}">${j.status}</span>
      <span class="pill" data-del="${j.id}" title="remove">✕</span></div>`;
  }).join('') : '<span class="dim">none yet</span>';
  $('jobs').querySelectorAll('.job').forEach(el=>el.onclick=()=>{
    current=el.dataset.id; drawn=''; draw();          // clicking always redraws
  });
  $('jobs').querySelectorAll('[data-del]').forEach(el=>el.onclick=async ev=>{
    ev.stopPropagation();
    await api('/api/jobs/'+el.dataset.del,{method:'DELETE'}).catch(()=>{});
    if(current===el.dataset.del){ current=null; drawn=''; $('detail').innerHTML=''; }
    listed=''; refresh();
  });
}

async function draw(job){
  if(!job){ try{ job=await api('/api/jobs/'+current); }catch(e){ return; } }
  // Rebuilding the panel replaces the <video>, which stops playback. Only do it
  // when something actually changed.
  const sig=[job.id,job.status,job.stage,job.output_name,
             JSON.stringify(job.summary||{}),(job.progress||[]).length].join('|');
  if(sig===drawn) return;
  // Rebuilding throws away the video position, the selection on the timeline and
  // anything tried but not saved. A poll arriving mid-edit must not do that, so
  // while there is work on screen the panel stays exactly as it is.
  if(P && P.job.id===job.id && (P.dirty || P.sel || !P.video.paused)) return;
  drawn=sig;
  const s=job.summary||{};
  let html=`<div class="card"><h2>${job.title}</h2>`;
  if(job.status==='error'){
    html+=`<div style="color:var(--bad)">${job.error||job.stage}</div>`;
    // The clips are still here. A render cut off by an idle timeout should cost
    // a button, not another upload.
    if((job.sources||[]).length) html+=`<button id="retry">Try again</button>`;
  }
  if(job.status==='working'||job.status==='queued'||job.status==='exporting'){
    html+=`<div class="dim">${job.stage}…</div>
      <div class="bar-outer"><div class="bar-inner" id="bar"></div></div>
      <div class="meta"><span id="pct"></span><span id="alive"></span></div>
      <button class="ghost" id="stopJob">Stop this</button>
      <div class="log">${(job.progress||[]).join('\\n')}</div>`;
  }
  if(job.status==='ready'||job.status==='done'){
    html+=`<div class="stage"><video id="pv" playsinline preload="metadata"
             src="/api/jobs/${job.id}/proxy.mp4"></video>
             <video id="bv" playsinline muted preload="metadata" hidden></video>
             <img id="bi" alt="" hidden><div id="caps"></div></div>
      <div class="transport">
        <button id="playBtn">Play</button>
        <button class="ghost" id="markIn">Start here</button>
        <button class="ghost" id="markOut">End here</button>
        <button class="ghost" id="speedBtn">1x</button>
        <span class="clock" id="clock">0:00</span>
      </div>
      <div class="track" id="track"></div>
      <div class="tools">
        <button class="ghost" id="cutSel">Cut the selection</button>
        <button class="ghost" id="keepSel">Keep only this</button>
        <button class="ghost" id="clearSel">Clear selection</button>
        <button class="ghost" id="pauseMode">Adjust pauses</button>
        <button class="ghost" id="undoTrims">Undo all edits</button>
      </div>
      <div id="joinInfo" hidden>
        <div class="row"><span class="pauseWhich">edge</span>
          <span class="dim" style="flex:1;font-size:12px">drag its marker on the strip</span>
          <b class="pauseVal">0.00s</b></div>
        <div class="row"><span>Transition</span>
          <input type="range" data-beat min="0" max="0.8" step="0.02">
          <b class="beatVal">0.18s</b></div>
      </div>
      <div id="lookHost"></div>
      <div class="stats">
        <div class="stat"><b id="statOut">${s.output_duration??'-'}s</b><span>from ${s.source_duration??'-'}s</span></div>
        <div class="stat"><b id="statCut">${s.removed??'-'}s</b><span>cut away</span></div>
        <div class="stat"><b id="statZoom">${s.zooms??0}</b><span>zooms</span></div>
      </div>
      <div class="unsaved" id="dirty"></div>
      <div class="bar">
        <button id="saveEdit" disabled>Save</button>
        <button class="ghost" id="discardEdit" disabled>Discard</button>
      </div>
      <button class="ghost" id="rerender">Render a real preview</button>
      <button id="export">${job.status==='done'?'Export again':'Approve &amp; export'}</button>`;
    if(job.output_name) html+=`<button class="ghost" onclick="location.href='/api/jobs/${job.id}/download'">Download ${job.output_name}</button>`;
  }
  html+='</div>';
  $('detail').innerHTML=html;

  // Kept out of the signature check above so the bar can move without the
  // whole panel being rebuilt under whatever you are watching.
  updateProgress(job);

  const stop=$('stopJob');
  if(stop) stop.onclick=async()=>{
    stop.disabled=true; stop.textContent='stopping…';
    try{ await api('/api/jobs/'+job.id+'/stop',{method:'POST'});
         drawn=''; listed=''; refresh(); }
    catch(e){ stop.textContent='error: '+e.message; stop.disabled=false; }
  };

  const retry=$('retry');
  if(retry) retry.onclick=async()=>{
    retry.disabled=true; retry.textContent='starting…';
    try{ await api('/api/jobs/'+job.id+'/start',{method:'POST'}); drawn=''; listed=''; refresh(); }
    catch(e){ retry.textContent='error: '+e.message; retry.disabled=false; }
  };

  if(job.status==='ready'||job.status==='done'){
    try{ edl=await api('/api/jobs/'+job.id+'/edl'); }catch(e){ edl=null; }
    if(edl){ mountPlayer(job); renderControls(job); }
    renderLook(job);
    const rr=$('rerender'), ex=$('export');
    if(rr) rr.onclick=()=>send(job,true);
    if(ex) ex.onclick=async()=>{ ex.disabled=true; ex.textContent='queued…';
      try{ await send(job,false); await api('/api/jobs/'+job.id+'/export',{method:'POST'});
           drawn=''; listed=''; refresh(); }
      catch(e){ ex.textContent='error: '+e.message; ex.disabled=false; } };
  }
}

// ---------------------------------------------------------------- the player
//
// The browser plays the untouched footage and draws the edit over it: segments
// it should skip, captions it should show, zooms it should apply. Nothing here
// is rendered, so a caption style or a trim is something you watch change while
// the video keeps playing, instead of something you queue and wait for.
//
// What you see is close, not identical: this is the browser laying out text,
// while the export is libass. Use it to judge timing, wording and framing, and
// the rendered preview to check the Arabic reads correctly.

let P = null;        // the live editing session for the open job

const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const stamp=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;

function mountPlayer(job){
  const video=$('pv'), track=$('track'), caps=$('caps');
  if(!video||!track) return;

  const copy=list=>(list||[]).map(d=>[d[0],d[1]]);
  P={job, video, track, caps, plan:edl, look:null, values:{},
     drops:copy(job.drops), pauses:copy(job.pauses), beats:copy(job.beats),
     saved:{drops:copy(job.drops), pauses:copy(job.pauses), beats:copy(job.beats)},
     sel:null, activeJoin:null, pauseMode:false, dirty:false, raf:0, pending:0,
     grabAt:0};

  P.duration = Math.max(...(P.plan.cuts||[]).map(c=>c.src_end), 1);
  buildTrack(); wireTransport(); wireTrack(); tick();
  // A clip shorter than the moment it was given would otherwise sit on its last
  // frame until the window ran out, which reads as it having got stuck.
  $('bv').addEventListener('ended', ()=>{ $('bv').hidden=true; $('bv').style.display='none'; });
  // A clip that will not play falls back to its still, so the preview shows
  // that b-roll happens here even when the browser cannot decode it.
  $('bv').addEventListener('error', ()=>{
    const name=(P.shown||'').split('|')[1];
    if(!name) return;
    $('bv').hidden=true; $('bv').style.display='none';
    $('bi').src='/api/broll/'+encodeURIComponent(name)+'/thumb.jpg';
    $('bi').className=$('bv').className||'cover';
    $('bi').style.display=''; $('bi').hidden=false;
  });
  video.addEventListener('loadedmetadata', layoutTrack);
}

function kept(){ return (P.plan.cuts||[]).filter(c=>c.enabled); }

// -- where are we, in the finished video? ------------------------------------
function outAt(src){
  for(const c of kept()) if(src>=c.src_start && src<=c.src_end)
    return c.out_start + (src - c.src_start);
  return null;
}
function nextKeptAfter(src){
  let best=null;
  for(const c of kept()) if(c.src_start>src && (!best || c.src_start<best.src_start)) best=c;
  return best;
}

function tick(){
  cancelAnimationFrame(P.raf);
  const step=()=>{
    if(!P || !document.body.contains(P.video)) return;
    // The next frame is asked for FIRST, and the work is wrapped. This loop is
    // what hides the b-roll, moves the captions and advances the playhead, so
    // one thrown error used to stop all of it for good: the footage kept playing
    // underneath, because that is the browser's own doing, while the b-roll sat
    // on screen to the end of the video and nothing answered. A bad frame should
    // cost a frame.
    P.raf=requestAnimationFrame(step);
    try{
      const v=P.video, t=v.currentTime;
      let out=outAt(t);
      if(out===null){
        // In footage that is cut. Jump to the next piece that survives, so
        // playback is the edit rather than the raw take.
        const nxt=nextKeptAfter(t);
        if(nxt){ v.currentTime=nxt.src_start; out=nxt.out_start; }
        else if(!v.paused){ v.pause(); const first=kept()[0]; if(first) v.currentTime=first.src_start; }
      }
      paint(out===null?0:out, t);
    }catch(err){
      // Said once, not sixty times a second, and never silently: a preview that
      // quietly stops matching the edit is worse than one that admits it.
      if(!P.complained){ P.complained=true;
        $('dirty').textContent='the preview hit a problem: '+(err&&err.message||err);
        console.error(err); }
    }
  };
  P.raf=requestAnimationFrame(step);
}

function paint(out, src){
  drawCaption(out);
  drawZoom(out);
  drawOverlay(out);
  if(P.els) P.els.head.style.left=(100*src/P.duration).toFixed(3)+'%';
  const total=kept().reduce((n,c)=>n+(c.src_end-c.src_start),0);
  $('clock').textContent=`${stamp(out)} / ${stamp(total)}`;
}

// -- captions ---------------------------------------------------------------
function drawCaption(out){
  const look=P.look||{}, caps=P.caps;
  const line=(P.plan.captions||[]).find(l=>out>=l.start-0.02 && out<=l.end+0.02);
  if(!line){ caps.innerHTML=''; return; }
  const h=P.video.clientHeight||P.caps.parentElement.clientHeight||600;
  const scale=h/(P.plan.output?.height||1920);
  const style=(look.style||'karaoke');
  const one=style==='word';
  const size=(look.font_size||92)*scale*(one?(look.word_size_boost||1.55):1);
  const primary=look.primary||'#FFFFFF', hot=look.highlight||'#FFD700';

  caps.style.bottom=((1-(look.y_pct??0.72))*h)+'px';
  caps.style.fontFamily=`"${look.font||'Cairo'}", system-ui, sans-serif`;
  caps.style.fontSize=size.toFixed(1)+'px';
  caps.style.fontWeight=(look.bold===false)?'600':'800';
  caps.style.color=primary;
  const edge=Math.max(1,(look.outline||7)*scale);
  caps.style.textShadow=[`0 0 ${edge}px ${look.outline_color||'#101010'}`,
    `${edge*0.5}px ${edge*0.5}px ${edge}px rgba(0,0,0,.85)`].join(',');

  const words=line.words&&line.words.length?line.words
    :[{text:line.text,start:line.start,end:line.end}];
  const active=words.findIndex(w=>out>=w.start&&out<=w.end);
  const shown=one?(active>=0?[words[active]]:[words[0]]):words;
  caps.innerHTML=shown.map(w=>{
    const on=(w===words[active]);
    const colour=(style==='plain')?primary:(on?hot:primary);
    const box=(style==='box'&&on)
      ?`background:${look.box_color||'#FFD700'};color:${look.box_text||'#101010'};`
      +`padding:.04em .16em;border-radius:.12em;`:'';
    return `<span style="color:${colour};${box}">${escapeHtml(w.text)}</span>`;
  }).join(' ');
  if(style==='pop'&&active===0) caps.animate(
    [{transform:'scale(1.10)'},{transform:'scale(1)'}],{duration:150,easing:'ease-out'});
}
function escapeHtml(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// -- b-roll ------------------------------------------------------------------
//
// The renderer lays these over you with ffmpeg. Without drawing them here too,
// tagging a clip and watching nothing happen looks exactly like the matching
// being broken - which is how it looked, and why this exists.
// Hidden, not torn down: removing the source and reloading would make coming
// back to the same clip fetch the whole thing again.
function hideOverlay(){
  if(!P) return;
  clearTimeout(P.overlayTimer);
  // Three ways, on purpose: the attribute, an inline display, and dropping the
  // class that positions it over the footage. Any one of them is enough; all
  // three means no future stylesheet rule can quietly bring it back.
  for(const el of [$('bv'), $('bi')]){
    if(!el) continue;
    if(el.tagName==='VIDEO'){ try{ el.pause(); }catch(_){} }
    el.hidden=true; el.style.display='none'; el.className='';
  }
  P.shown=null;
}

function drawOverlay(out){
  const live=(P.plan.overlays||[]).find(o=>o.enabled && out>=o.out_start && out<=o.out_end);
  const video=$('bv'), image=$('bi');

  if(!live){
    if(P.shown!==null) hideOverlay();
    return;
  }

  const name=live.name||'';
  const still=/\.(jpe?g|png|webp)$/i.test(name);
  const el=still?image:video;
  // Keyed on the clip as well as the overlay: ids are handed out fresh on
  // every plan, so the same id can come back pointing at a different clip.
  const key=live.id+'|'+name;

  if(P.shown!==key){
    P.shown=key;
    (still?video:image).hidden=true;
    if(!still) video.pause();
    el.src=(still?'/api/broll/':'/api/broll/')+encodeURIComponent(name)+'/file';
    el.className=live.mode||'cover';
    el.style.opacity=live.opacity ?? 1;
    el.style.display='';
    el.hidden=false;
    if(!still){
      // Seek and play once, when the clip is ready. Doing it every frame is
      // what took the server down: an unloaded video answers a seek with a
      // fresh range request, so sixty a second went out until nothing was
      // left to serve the page itself.
      const begin=()=>{
        try{ video.currentTime=live.asset_start||0; }catch(_){}
        try{ const started=video.play(); if(started) started.catch(()=>{}); }catch(_){}
      };
      if(video.readyState>=1) begin();
      else video.addEventListener('loadedmetadata', begin, {once:true});
    }
    // Belt and braces. The loop above normally takes it away at the right
    // moment; this takes it away even if the loop is not running, because a
    // b-roll that will not leave is the worst way for that to show up.
    clearTimeout(P.overlayTimer);
    const left=(live.out_end-out)/(P.video.playbackRate||1);
    P.overlayTimer=setTimeout(hideOverlay, Math.max(120, left*1000)+150);
  }
}

// -- zooms and transitions ---------------------------------------------------
function drawZoom(out){
  let factor=1, filter='';
  for(const z of (P.plan.zooms||[])){
    if(!z.enabled || out<z.out_start || out>z.out_end) continue;
    const p=(out-z.out_start)/Math.max(0.001,z.out_end-z.out_start);
    const eased=p*p*(3-2*p);                       // the renderer's smooth ease
    factor=z.start_factor+(z.end_factor-z.start_factor)*eased;
  }
  for(const t of (P.plan.transitions||[])){
    if(!t.enabled) continue;
    const p=(out-t.out_time)/Math.max(0.001,t.duration);
    if(p<0||p>1) continue;
    const fade=1-p;
    if(t.kind==='punch') factor*=1+0.06*t.strength*fade;
    else if(t.kind==='flash') filter=`brightness(${1+0.5*t.strength*fade})`;
    else if(t.kind==='blur') filter=`blur(${(4*t.strength*fade).toFixed(2)}px)`;
  }
  P.video.style.transform=`scale(${factor.toFixed(4)})`;
  P.video.style.filter=filter;
}

// -- the track ---------------------------------------------------------------
//
// Built once, then moved by styles alone. The first version rebuilt the whole
// strip on every pointermove, which is exactly what makes a drag feel like it is
// catching on something: the element under your finger is destroyed and remade
// sixty times a second. Nothing here touches innerHTML while you are dragging.

function buildTrack(){
  const t=P.track;
  t.innerHTML='<div class="keeps"></div><div class="joins"></div>'
    +'<div class="sel" hidden></div>'
    +'<div class="grip" data-grip="0" hidden></div>'
    +'<div class="grip" data-grip="1" hidden></div>'
    +'<div class="head"></div><div class="lbl"></div>';
  P.els={keeps:t.querySelector('.keeps'), joins:t.querySelector('.joins'),
         sel:t.querySelector('.sel'), head:t.querySelector('.head'),
         lbl:t.querySelector('.lbl'),
         grips:[...t.querySelectorAll('[data-grip]')]};
  layoutTrack();
}

const pct=v=>(100*v/P.duration).toFixed(4)+'%';

function layoutTrack(){
  const live=kept();
  // Only rebuild the blocks when their number changes; otherwise move them.
  if(P.els.keeps.children.length!==live.length)
    P.els.keeps.innerHTML=live.map(()=>'<div class="keep"></div>').join('');
  live.forEach((c,i)=>{
    const el=P.els.keeps.children[i];
    el.style.left=pct(c.src_start);
    el.style.width=pct(c.src_end-c.src_start);
  });

  const joins=P.pauseMode?junctions():[];
  if(P.els.joins.children.length!==joins.length*2)
    P.els.joins.innerHTML=joins.map(()=>
      '<div class="join left"><i></i></div><div class="join right"><i></i></div>').join('');
  joins.forEach((j,i)=>{
    const left=P.els.joins.children[i*2], right=P.els.joins.children[i*2+1];
    left.style.left=pct(j.leftAt);   left.dataset.join=i; left.dataset.side='before';
    right.style.left=pct(j.rightAt); right.dataset.join=i; right.dataset.side='after';
    left.classList.toggle('on', P.activeJoin===i);
    right.classList.toggle('on', P.activeJoin===i);
  });

  const s=P.sel;
  P.els.sel.hidden=!s;
  P.els.grips.forEach((g,i)=>{ g.hidden=!s; if(s) g.style.left=pct(s[i]); });
  if(s){ P.els.sel.style.left=pct(s[0]); P.els.sel.style.width=pct(s[1]-s[0]); }
  P.els.lbl.textContent = s
    ? `${stamp(s[0])} → ${stamp(s[1])} (${(s[1]-s[0]).toFixed(1)}s selected)`
    : (P.pauseMode ? 'each gap has two markers — drag either edge; only that piece moves'
                   : 'tap to jump · drag across to select');
}

// Every gap between two surviving pieces: the silence the cut took out.
function junctions(){
  const live=kept(), out=[];
  for(let i=0;i<live.length-1;i++){
    const before=live[i], after=live[i+1];
    out.push({index:i, middle:(before.src_end+after.src_start)/2,
              gap:Math.max(0, after.src_start-before.src_end),
              leftAt:before.src_end, rightAt:after.src_start});
  }
  return out;
}

// One entry per edge. The end of this shot and the start of the next are two
// different decisions, and moving one has no business moving the other.
function pauseOf(i, side){
  const middle=junctions()[i].middle;
  const stored=P.pauses.find(e=>Math.abs(e[0]-middle)<=0.35 && (e[2]||'both')===side);
  return stored?stored[1]:0;
}
function setPause(i, side, seconds){
  const middle=junctions()[i].middle;
  const at=P.pauses.findIndex(e=>Math.abs(e[0]-middle)<=0.35 && (e[2]||'both')===side);
  if(at>=0) P.pauses[at]=[middle,seconds,side];
  else P.pauses.push([middle,seconds,side]);
}

function atX(clientX){
  const box=P.track.getBoundingClientRect();
  return clamp((clientX-box.left)/Math.max(1,box.width),0,1)*P.duration;
}

// One handler for the whole strip. Pointer events cover mouse, pen and touch
// with the same code, and capture means a finger that slides off the strip -
// or off the screen - still finishes the drag it started.
function wireTrack(){
  const t=P.track;
  let mode=null, anchor=0, which=0, side='before', frame=0, pending=null;

  const apply=()=>{
    frame=0;
    // Compared against null, not truthiness: the very start of the video is 0,
    // and a drag to the first frame is a real drag, not an absent one.
    if(pending===null) return;
    const at=pending; pending=null;
    if(mode==='scrub'){
      seekTo(at);
    } else if(mode==='select'){
      P.sel=[Math.min(anchor,at),Math.max(anchor,at)];
    } else if(mode==='grip'){
      P.sel = which===0 ? [Math.min(at,P.sel[1]-0.05),P.sel[1]]
                        : [P.sel[0],Math.max(at,P.sel[0]+0.05)];
      seekTo(P.sel[which],{quiet:true});
    } else if(mode==='join'){
      const j=junctions()[which];
      // The marker follows your finger: the edge you took hold of goes where you
      // put it. Which way that lengthens the pause depends on which edge it is.
      const travel=at-P.grabAt;
      const delta=clamp(anchor+(side==='before'?travel:-travel), -3.0, j.gap);
      setPause(which, side, delta);
      showJoin(which, side, delta, j.gap);
    }
    layoutTrack();
  };
  const queue=at=>{ pending=at; if(!frame) frame=requestAnimationFrame(apply); };

  t.addEventListener('pointerdown', ev=>{
    const join=ev.target.closest('.join'), grip=ev.target.closest('[data-grip]');
    const at=atX(ev.clientX);
    if(ev.target.closest('.head')){
      mode='scrub';
      P.wasPlaying=!P.video.paused;
      if(P.wasPlaying) P.video.pause();
      seekTo(at);
    }
    else if(join){ mode='join'; which=Number(join.dataset.join); side=join.dataset.side;
              P.grabAt=at; anchor=pauseOf(which, side);
              P.activeJoin=which; layoutTrack(); }
    else if(grip){ mode='grip'; which=Number(grip.dataset.grip); }
    else { mode='maybe'; anchor=at; }
    // preventDefault first: if capture throws, a touch that was meant to drag
    // must still not turn into the page scrolling away under the finger.
    ev.preventDefault();
    try{ t.setPointerCapture(ev.pointerId); }catch(_){}
  });

  t.addEventListener('pointermove', ev=>{
    if(!mode) return;
    const at=atX(ev.clientX);
    // A tap only becomes a drag once it has travelled far enough that it cannot
    // be a shaky finger. Below that, it stays a tap.
    if(mode==='maybe'){
      if(Math.abs(at-anchor) < P.duration*0.004) return;
      mode='select';
    }
    queue(at);
  });

  const finish=ev=>{
    if(!mode) return;
    if(mode==='maybe') seekTo(atX(ev.clientX));
    // Flush the move still waiting on a frame. Releasing cancels that frame, so
    // without this the last thing you did before letting go is thrown away -
    // and a quick flick, which is over inside one frame, does nothing at all.
    if(pending!==null) apply();
    if(mode==='join'||mode==='grip'||mode==='select'){ markDirty(); }
    if(mode==='scrub' && P.wasPlaying){ P.video.play(); P.wasPlaying=false; }
    if(mode==='join') repaintPlan();
    cancelAnimationFrame(frame); frame=0; pending=null;
    mode=null;
    try{ t.releasePointerCapture(ev.pointerId); }catch(_){}
  };
  t.addEventListener('pointerup', finish);
  // A cancelled pointer (a phone deciding it was a scroll, a call arriving)
  // must end the drag too, or the strip stays stuck to the finger.
  t.addEventListener('pointercancel', finish);
}

function showJoin(i, side, delta, gap){
  const box=$('joinInfo');
  if(!box) return;
  const beat=beatOf(i);
  box.hidden=false;
  box.querySelector('.pauseWhich').textContent=
    side==='before' ? 'end of this piece' : 'start of the next piece';
  box.querySelector('.pauseVal').textContent=
    (delta>=0?'+':'')+delta.toFixed(2)+'s'+(gap?` (up to +${gap.toFixed(2)}s)`:'');
  box.querySelector('input[data-beat]').value=beat;
  box.querySelector('.beatVal').textContent=beat.toFixed(2)+'s';
}
function beatOf(i){
  const middle=junctions()[i].middle;
  const stored=P.beats.find(([mid])=>Math.abs(mid-middle)<=0.35);
  if(stored) return stored[1];
  const look=P.plan.transitions||[];
  return look.length?look[0].duration:0.18;
}
function setBeat(i, seconds){
  const middle=junctions()[i].middle;
  const at=P.beats.findIndex(([mid])=>Math.abs(mid-middle)<=0.35);
  if(at>=0) P.beats[at]=[middle,seconds]; else P.beats.push([middle,seconds]);
}

function seekTo(src, opts={}){
  P.video.currentTime=clamp(src,0,P.duration);
  if(!opts.quiet && P.video.paused) paint(outAt(src)??0, src);
}

function wireTransport(){
  const v=P.video;
  $('playBtn').onclick=()=>{
    if(v.paused){ if(outAt(v.currentTime)===null){ const c=kept()[0]; if(c) v.currentTime=c.src_start; }
      v.play(); } else v.pause();
  };
  v.addEventListener('pause',()=>$('playBtn').textContent='Play');
  v.addEventListener('play',()=>$('playBtn').textContent='Pause');
  $('markIn').onclick=()=>{ const t=v.currentTime;
    P.sel=[t, Math.max(t+0.2, P.sel?P.sel[1]:t+1)]; layoutTrack(); markDirty(); };
  $('markOut').onclick=()=>{ const t=v.currentTime;
    P.sel=[Math.min(P.sel?P.sel[0]:Math.max(0,t-1), t-0.2), t]; layoutTrack(); markDirty(); };
  $('clearSel').onclick=()=>{ P.sel=null; layoutTrack(); };
  const SPEEDS=[1,1.5,2,0.5];
  $('speedBtn').onclick=()=>{
    P.speed=SPEEDS[(SPEEDS.indexOf(P.speed||1)+1)%SPEEDS.length];
    v.playbackRate=P.speed;
    $('speedBtn').textContent=P.speed+'x';
  };
  $('cutSel').onclick=()=>{ if(P.sel) applyTrim([P.sel]); };
  $('keepSel').onclick=()=>{ if(P.sel) applyTrim([[0,P.sel[0]],[P.sel[1],P.duration]]); };
  $('undoTrims').onclick=()=>{ P.drops=[]; P.pauses=[]; P.beats=[]; P.sel=null;
    P.activeJoin=null; markDirty(); repaintPlan(); };
  $('pauseMode').onclick=()=>{
    P.pauseMode=!P.pauseMode; P.sel=null; P.activeJoin=null;
    $('pauseMode').classList.toggle('on', P.pauseMode);
    $('joinInfo').hidden=true;
    P.track.classList.toggle('pauses', P.pauseMode);
    layoutTrack();
  };
  const beat=$('joinInfo')?.querySelector('input[data-beat]');
  if(beat) beat.oninput=()=>{
    if(P.activeJoin===null) return;
    setBeat(P.activeJoin, Number(beat.value));
    $('joinInfo').querySelector('.beatVal').textContent=Number(beat.value).toFixed(2)+'s';
    markDirty(); repaintPlan();
  };
  $('saveEdit').onclick=saveEdit;
  $('discardEdit').onclick=()=>{ P.values={}; P.drops=P.saved.drops.map(d=>[...d]);
    P.pauses=P.saved.pauses.map(d=>[...d]); P.beats=P.saved.beats.map(d=>[...d]);
    P.sel=null; P.activeJoin=null; markDirty(false); repaintPlan(); renderLook(P.job); };
}

// A selection is made against the finished video, so hand back output time and
// let the server work out which footage that was.
function applyTrim(ranges){
  const out=ranges.map(([a,b])=>[outAt(a)??nearestOut(a),outAt(b)??nearestOut(b)])
                  .filter(([a,b])=>b>a);
  if(!out.length){ $('dirty').textContent='that selection is already cut'; return; }
  api(`/api/jobs/${P.job.id}/trim`,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ranges:out})})
    .then(r=>{ P.drops=r.drops; P.sel=null; markDirty(); repaintPlan(); })
    .catch(e=>{ $('dirty').textContent='error: '+e.message; });
}
function nearestOut(src){
  let best=0;
  for(const c of kept()){
    if(c.src_end<=src) best=c.out_end;
    else if(c.src_start>=src) return c.out_start;
  }
  return best;
}

// -- asking the server what this would look like -----------------------------
function repaintPlan(){
  clearTimeout(P.pending);
  P.pending=setTimeout(async()=>{
    try{
      const r=await api(`/api/jobs/${P.job.id}/preview-plan`,{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({values:P.values, drops:P.drops,
                             pauses:P.pauses, beats:P.beats})});
      P.plan=r.edl; P.look=r.look; edl=r.edl;
      $('statOut').textContent=(r.summary.output_duration??'-')+'s';
      $('statCut').textContent=(r.summary.removed??'-')+'s';
      $('statZoom').textContent=r.summary.zooms??0;
      layoutTrack();
    }catch(e){ $('dirty').textContent='error: '+e.message; }
  }, 260);
}

function markDirty(on=true){
  P.dirty=on;
  $('saveEdit').disabled=!on; $('discardEdit').disabled=!on;
  $('dirty').textContent=on
    ? 'not saved yet — what you are watching is a try-out'
    : '';
}

async function saveEdit(){
  $('saveEdit').disabled=true; $('dirty').textContent='saving…';
  try{
    await api(`/api/jobs/${P.job.id}/save`,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({values:P.values, drops:P.drops,
                           pauses:P.pauses, beats:P.beats})});
    P.saved={drops:P.drops.map(d=>[...d]), pauses:P.pauses.map(d=>[...d]),
             beats:P.beats.map(d=>[...d])};
    markDirty(false);
    drawn=''; listed=''; refresh();
  }catch(e){ $('dirty').textContent='error: '+e.message; $('saveEdit').disabled=false; }
}

function renderControls(job){
  const cap=(edl.captions||[]).map((c,i)=>
    `<div class="row"><span class="t">${fmt(c.start)}</span>
     <span class="grow"><input type="text" dir="auto" data-cap="${i}" value="${(c.text||'').replace(/"/g,'&quot;')}"></span></div>`).join('');
  const zoom=(edl.zooms||[]).map((z,i)=>
    `<div class="row"><input type="checkbox" data-zoom="${i}" ${z.enabled?'checked':''}>
     <span class="t">${fmt(z.out_start)}</span><span class="grow dim">${z.kind.replace('_',' ')} → ${z.end_factor.toFixed(2)}x</span></div>`).join('');
  const tr=(edl.transitions||[]).map((t,i)=>
    `<div class="row"><input type="checkbox" data-tr="${i}" ${t.enabled?'checked':''}>
     <span class="t">${fmt(t.out_time)}</span><span class="grow dim">${t.kind}</span></div>`).join('');
  $('detail').insertAdjacentHTML('beforeend',
    `<div class="card"><h2>Captions</h2>${cap||'<span class="dim">none</span>'}</div>
     <div class="card"><h2>Zoom moves</h2>${zoom||'<span class="dim">none</span>'}</div>
     <div class="card"><h2>Transitions</h2>${tr||'<span class="dim">none</span>'}</div>`);
}

// The look panel. Everything here is a real profile key, so a choice made on a
// phone is the same change as `--set captions.style=word` on the command line.
let look=null;

function control(f){
  const id=`s_${f.key.replace('.','_')}`;
  const help=f.help?`<div class="help">${f.help}</div>`:'';
  if(f.type==='bool')
    return `<div class="set inline"><label for="${id}">${f.label}${help}</label>
      <input type="checkbox" id="${id}" data-set="${f.key}" ${f.value?'checked':''}></div>`;
  if(f.type==='select')
    return `<div class="set"><label for="${id}">${f.label}</label>${help}
      <select id="${id}" data-set="${f.key}">${f.options.map(o=>
        `<option value="${o.value}" ${String(o.value)===String(f.value)?'selected':''}>${o.label}</option>`
      ).join('')}</select></div>`;
  if(f.type==='color')
    return `<div class="set"><label for="${id}">${f.label}</label>${help}
      <input type="color" id="${id}" data-set="${f.key}" value="${f.value||'#ffffff'}"></div>`;
  return `<div class="set"><label for="${id}">${f.label}</label>${help}
    <input type="number" id="${id}" data-set="${f.key}" value="${f.value}"
      min="${f.min}" max="${f.max}" step="${f.step}"></div>`;
}

// An icon per group, opening its controls right under the video. The panel used
// to be one long list below everything else, which meant scrolling past the
// whole edit to change a font and scrolling back to see what it did.
const GROUP_ICONS={Captions:'💬', Motion:'🎬', 'B-roll':'🎞️', Pacing:'⏱️'};

async function renderLook(job){
  let look;
  try{ look=await api('/api/jobs/'+job.id+'/settings'); }catch(e){ return; }
  const host=$('lookHost');
  if(!host) return;
  const groups=[...new Set(look.fields.map(f=>f.group))];

  host.innerHTML=
    `<div class="iconbar">${groups.map(g=>
        `<button data-group="${g}"><b>${GROUP_ICONS[g]||'⚙️'}</b>${g}</button>`).join('')}
      <button data-group="__defaults"><b>⭐</b>Default</button></div>
     <div id="lookPane"></div>`;

  if(P){
    // Whatever the panel shows is the look the player should already be using,
    // so the first frame drawn matches the controls without a round trip.
    P.look={}; look.fields.forEach(f=>{
      if(f.key.startsWith('captions.')) P.look[f.key.slice(9)]=f.value;
    });
    P.fields=look.fields;
    loadFont(P.look.font);
  }

  const openGroup=name=>{
    host.querySelectorAll('[data-group]').forEach(b=>
      b.classList.toggle('on', b.dataset.group===name && P.openGroup===name));
    const pane=$('lookPane');
    if(P.openGroup!==name){ pane.innerHTML=''; P.openGroup=null; return; }
    pane.innerHTML = name==='__defaults' ? defaultsPane()
      : look.fields.filter(f=>f.group===name).map(control).join('')
        + `<div class="dim" style="font-size:12px;margin-top:8px">
             Changes show on the video as you make them. Nothing is kept until
             you press Save.</div>`;
    if(name==='__defaults') wireDefaults(); else wireControls();
  };

  host.querySelectorAll('[data-group]').forEach(el=>el.onclick=()=>{
    P.openGroup = P.openGroup===el.dataset.group ? null : el.dataset.group;
    openGroup(el.dataset.group);
  });
  if(P.openGroup) openGroup(P.openGroup);
}

function readPanel(){
  const values={};
  document.querySelectorAll('[data-set]').forEach(el=>{
    values[el.dataset.set]=el.type==='checkbox'?(el.checked?'true':'false'):el.value;
  });
  // A control that is not on screen still counts: the panel shows one group at
  // a time, and closing it must not quietly revert the others.
  return Object.assign({}, P.values, values);
}

function wireControls(){
  document.querySelectorAll('[data-set]').forEach(el=>{
    const live=()=>{
      if(!P) return;
      P.values=readPanel();
      // Anything purely about appearance is applied in the browser on the next
      // frame. Anything that changes a decision - how many words to a line, how
      // often to zoom - has to be re-decided, which the server does without
      // rendering, in about a second.
      const key=el.dataset.set;
      if(key.startsWith('captions.') && !LIVE_REPLAN.has(key)){
        P.look[key.slice(9)]=el.type==='checkbox'?el.checked:coerce(el.value);
        if(key==='captions.font') loadFont(el.value);
      } else {
        repaintPlan();
      }
      markDirty();
    };
    el.addEventListener('input', live);
    el.addEventListener('change', live);
  });
}

function defaultsPane(){
  return `<div class="dim" style="font-size:13px">
      Keep the look you are using now for every new upload, so a style you
      settled on is not something to pick again on each video.</div>
    <button id="saveDefaults">Save this look as my default</button>
    <button class="ghost" id="resetLook">Put this video back to the template</button>
    <button class="ghost" id="clearDefaults">Forget my default</button>
    <div id="defaultsMsg" class="dim" style="margin-top:8px"></div>`;
}

function wireDefaults(){
  api('/api/defaults').then(d=>{
    $('defaultsMsg').textContent = d.labels.length
      ? 'your default sets: '+d.labels.join(', ')
      : 'no default saved yet — new uploads use the template as-is';
  }).catch(()=>{});

  $('saveDefaults').onclick=async()=>{
    const values=Object.assign({}, P.values);
    // Everything the panel can set, not just what was touched this session.
    (P.fields||[]).forEach(f=>{ if(!(f.key in values)) values[f.key]=String(f.value); });
    try{
      const r=await api('/api/defaults',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({values})});
      $('defaultsMsg').textContent = r.labels.length
        ? 'saved — new uploads will use: '+r.labels.join(', ')
        : 'saved — that is the factory look, so nothing to remember';
    }catch(e){ $('defaultsMsg').textContent='error: '+e.message; }
  };
  $('resetLook').onclick=async()=>{
    // This one is about the open video, not the default: it throws away the
    // look settings saved against this edit and goes back to its template.
    try{
      P.values={};
      await api('/api/jobs/'+P.job.id+'/settings',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({reset:true})});
      drawn=''; listed=''; refresh();
    }catch(e){ $('defaultsMsg').textContent='error: '+e.message; }
  };
  $('clearDefaults').onclick=async()=>{
    try{
      await api('/api/defaults',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({clear:true})});
      $('defaultsMsg').textContent='cleared — new uploads use the template as-is';
    }catch(e){ $('defaultsMsg').textContent='error: '+e.message; }
  };
}

// Settings the browser cannot fake: they change what the editor decides, not
// how it looks, so the plan has to be made again.
const LIVE_REPLAN=new Set(['captions.max_words','captions.enabled']);
const coerce=v=>{ const n=Number(v); return Number.isFinite(n)&&v.trim!==undefined&&v!==''?n:v; };

const FONTS=new Set();
function loadFont(family){
  if(!family||FONTS.has(family)) return;
  FONTS.add(family);
  // The same file libass will use, so the shape of the letters is not a guess.
  const style=document.createElement('style');
  style.textContent=`@font-face{font-family:"${family}";`
    +`src:url("/api/fonts/${encodeURIComponent(family)}") format("truetype");`
    +`font-display:swap}`;
  document.head.appendChild(style);
}

async function send(job, rerender){
  const body={rerender, captions:[], zooms:[], overlays:(edl.overlays||[]), transitions:[]};
  document.querySelectorAll('[data-cap]').forEach(el=>body.captions[el.dataset.cap]={text:el.value});
  document.querySelectorAll('[data-zoom]').forEach(el=>body.zooms[el.dataset.zoom]={enabled:el.checked});
  document.querySelectorAll('[data-tr]').forEach(el=>body.transitions[el.dataset.tr]={enabled:el.checked});
  const r=await api('/api/jobs/'+job.id+'/edl',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(rerender){ drawn=''; listed=''; refresh(); }
  return r;
}

async function runUpdate(where){
  where.textContent='updating…';
  try{
    const r=await api('/api/update',{method:'POST'});
    where.textContent=r.message;
    // The server is restarting under us; wait for it to answer again, then
    // reload, so the page you end up on is the new one.
    if(r.restarting) setTimeout(async function wait(){
      try{ await api('/api/me'); location.reload(); }
      catch(e){ setTimeout(wait, 1500); }
    }, 4000);
  }catch(e){ where.textContent=e.message; }
}

$('updateLink').onclick=async ev=>{
  ev.preventDefault();
  $('updateMsg').textContent=' · checking…';
  try{
    const v=await api('/api/version?check=1');
    if(!v.behind){ $('updateMsg').textContent=' · this is the newest version'; return; }
    announce(v);
    $('updateMsg').textContent='';
  }catch(e){ $('updateMsg').textContent=' · '+e.message; }
};

function announce(v){
  if(!v.behind) return;
  $('newVersion').hidden=false;
  $('newVersionText').textContent =
    `A newer version is ready — ${v.behind} change${v.behind>1?'s':''} since this one.`;
}
$('updateNow').onclick=()=>runUpdate($('newVersionText'));

// Checked on every load, so a new version says so rather than waiting to be
// found. A machine that has been left running for days is the case that matters:
// nothing restarts it, so nothing else would ever mention it.
function checkVersion(){
  api('/api/version').then(announce).catch(()=>{});
}

api('/api/me').then(d=>{
  show(d.authenticated);
  if(d.version) $('ver').textContent=d.version;
  if(d.authenticated) checkVersion();
}).catch(()=>show(false));
</script></body></html>
"""
