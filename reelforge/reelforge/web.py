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

from .edl import EDL
from .pipeline import PACKAGE_ROOT, AutoEditor
from .profile import StyleProfile

# Imported at module level, not inside create_app: `from __future__ import
# annotations` makes every annotation a string, and FastAPI resolves those
# against module globals - a name imported inside a function is invisible to it
# and silently becomes a query parameter.
try:
    from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
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


def apply_drops(edl: EDL, drops: list) -> None:
    """Switch off the segments covering trimmed-away stretches of the source.

    Matched by overlap rather than by segment id, so a trim holds even when a new
    plan splits the footage differently - which it does the moment the pacing
    settings change.
    """
    if not drops:
        return
    ranges = [(float(start), float(end)) for start, end in drops if float(end) > float(start)]
    wanted = {}
    for cut in edl.cuts:
        middle = (cut.src_start + cut.src_end) / 2.0
        wanted[cut.id] = not any(start <= middle <= end for start, end in ranges)
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

    def __init__(self, store: JobStore, fonts_dir: Path):
        self.store = store
        self.fonts_dir = fonts_dir
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.editors: dict[str, AutoEditor] = {}
        self.edls: dict[str, EDL] = {}
        self._recent: list[str] = []
        # Restoring happens on whichever thread asked for the edit, which is the
        # request thread as often as the worker.
        self._lock = threading.Lock()
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

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
                elif task == "export":
                    self._export(job)
            except Exception as exc:                      # a failed job must not kill the worker
                self.store.update(job, status="error", stage="failed",
                                  error=f"{type(exc).__name__}: {exc}"[:400])

    def _process(self, job: Job) -> None:
        directory = self.store.dir(job.id)

        def note(message: str) -> None:
            job.progress.append(message)
            self.store.update(job, stage=message)

        self.store.update(job, status="working", stage="starting", error="")
        profile = self.profile_for(job)
        self._ensure_font(profile, note)
        editor = AutoEditor(profile, project_dir=directory / "project",
                            fonts_dir=self.fonts_dir, on_status=note)
        clips = [Path(p) for p in job.sources]
        if len(clips) > 1:
            from .join import join_clips  # noqa: PLC0415
            source = join_clips(clips, editor.work_dir / "joined.mp4",
                                max_height=int(profile.get("output.height")
                                               * float(profile.get("output.zoom_headroom"))),
                                on_status=note)
        else:
            source = clips[0]

        self.store.update(job, prepared=str(source))
        self._plan_into(job, source, editor, note)

    def profile_for(self, job: Job) -> StyleProfile:
        """Template, then the model, then whatever look settings were chosen."""
        profile = StyleProfile.resolve(job.template or None,
                                       [PACKAGE_ROOT / "templates", PACKAGE_ROOT / "profiles"])
        overrides = [f"asr.model={job.model}"]
        backend = os.environ.get("REELFORGE_ASR_BACKEND")
        if backend:
            overrides.append(f"asr.backend={backend}")
        overrides += [f"{key}={value}" for key, value in (job.overrides or {}).items()]
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

    def _plan_into(self, job: Job, source: Path, editor: AutoEditor, note) -> None:
        """Decide the edit for `source` and render a preview into the job folder."""
        result = editor.plan(source)
        apply_drops(result.edl, job.drops)
        self._remember(job.id, editor, result.edl)
        for warning in result.warnings:
            note(warning)
        self.keep(job.id)
        note("rendering preview")
        editor.render(result.edl, self.store.dir(job.id) / "preview.mp4", preview=True)
        self.store.update(job, status="ready", stage="ready to review",
                          run_id=result.run_id, summary=result.edl.summary())

    def _replan(self, job: Job) -> None:
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
            self.store.update(job, stage=message)

        fixes = self._typed_fixes(job)
        self.store.update(job, status="working", stage="applying the new look", error="")
        profile = self.profile_for(job)
        self._ensure_font(profile, note)
        editor = AutoEditor(profile, project_dir=self.store.dir(job.id) / "project",
                            fonts_dir=self.fonts_dir, on_status=note)
        self._plan_into(job, source, editor, note)
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
        self.store.update(job, status="working", stage="re-rendering the preview")
        editor.render(edl, self.store.dir(job.id) / "preview.mp4", preview=True)
        self.store.update(job, status="ready", stage="ready to review",
                          summary=edl.summary())

    def _export(self, job: Job) -> None:
        edl = self.edl_for(job)
        if edl is None:
            raise RuntimeError("this edit is no longer loaded - upload it again")
        editor = self.editors[job.id]
        self.store.update(job, status="exporting", stage="rendering the final video")
        name = f"{Path(job.title).stem or 'reel'}-reel.mp4"
        editor.render(edl, self.store.dir(job.id) / name, preview=False)
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
    runner = Runner(store, PACKAGE_ROOT / "assets" / "fonts")

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
        store.update(job, status="uploading", stage="waiting for clips")
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
  <div class="card">
    <h1>New edit</h1>
    <div class="dim" id="ver" style="font-size:11px;margin-bottom:6px"></div>
    <div class="dim">Pick every take of one video. They are joined in the order chosen.</div>
    <input type="file" id="files" accept="video/*" multiple>
    <select id="template"></select>
    <select id="model">
      <option value="small">small - faster, good Arabic</option>
      <option value="large-v3">large-v3 - slower, best Arabic</option>
      <option value="medium">medium - in between</option>
    </select>
    <button id="upload">Upload and edit</button>
    <div id="msg"></div>
  </div>

  <div class="card">
    <h2>Edits</h2>
    <div id="jobs" class="dim">none yet</div>
  </div>

  <div id="detail"></div>
</div>

<script>
const $=id=>document.getElementById(id);
let current=null, edl=null, poll=null, drawn='', listed='';
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
function show(authed){ $('login').hidden=authed; $('app').hidden=!authed; if(authed){loadTemplates();refresh();} }

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

async function refresh(){
  let jobs=[];
  try{ jobs=await api('/api/jobs'); }catch(e){ return; }
  const listSig = jobs.map(j=>j.id+j.status+j.stage).join('|');
  if(listSig!==listed){ listed=listSig; renderJobs(jobs); }
  const job = current ? jobs.find(j=>j.id===current) : null;
  if(job) draw(job);
  // Only keep polling while something is actually happening. Re-rendering an
  // idle page rebuilds the video element, which restarts whatever you were
  // watching - so once everything is finished, stop.
  const busy = jobs.some(j=>['working','queued','exporting','uploading'].includes(j.status));
  clearTimeout(poll);
  if(busy) poll=setTimeout(refresh, 2500);
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
    html+=`<div class="dim">${job.stage}…</div><div class="log">${(job.progress||[]).join('\\n')}</div>`;
  }
  if(job.status==='ready'||job.status==='done'){
    html+=`<video controls playsinline preload="metadata" src="/api/jobs/${job.id}/preview.mp4?v=${Date.now()}"></video>
      <div class="stats">
        <div class="stat"><b>${s.output_duration??'-'}s</b><span>from ${s.source_duration??'-'}s</span></div>
        <div class="stat"><b>${s.removed??'-'}s</b><span>dead air cut</span></div>
        <div class="stat"><b>${s.zooms??0}</b><span>zooms</span></div>
      </div>
      <button class="ghost" id="rerender">Re-render preview</button>
      <button id="export">${job.status==='done'?'Export again':'Approve &amp; export'}</button>`;
    if(job.output_name) html+=`<button class="ghost" onclick="location.href='/api/jobs/${job.id}/download'">Download ${job.output_name}</button>`;
  }
  html+='</div>';
  $('detail').innerHTML=html;

  const retry=$('retry');
  if(retry) retry.onclick=async()=>{
    retry.disabled=true; retry.textContent='starting…';
    try{ await api('/api/jobs/'+job.id+'/start',{method:'POST'}); drawn=''; listed=''; refresh(); }
    catch(e){ retry.textContent='error: '+e.message; retry.disabled=false; }
  };

  if(job.status==='ready'||job.status==='done'){
    try{ edl=await api('/api/jobs/'+job.id+'/edl'); }catch(e){ edl=null; }
    if(edl){ renderSegments(job); renderControls(job); }
    renderLook(job);
    const rr=$('rerender'), ex=$('export');
    if(rr) rr.onclick=()=>send(job,true);
    if(ex) ex.onclick=async()=>{ ex.disabled=true; ex.textContent='queued…';
      try{ await send(job,false); await api('/api/jobs/'+job.id+'/export',{method:'POST'});
           drawn=''; listed=''; refresh(); }
      catch(e){ ex.textContent='error: '+e.message; ex.disabled=false; } };
  }
}

// The timeline. Each segment is a stretch of speech that survived the silence
// cutting; the bar above is the same list drawn to scale, so the shape of the
// video is visible before any of it is read.
function renderSegments(job){
  const cuts=edl.cuts||[];
  if(!cuts.length) return;
  const total=cuts.reduce((n,c)=>n+(c.src_end-c.src_start),0)||1;
  const bar=cuts.map((c,i)=>
    `<div class="${c.enabled?'':'off'}" data-bar="${i}"
      style="flex:${((c.src_end-c.src_start)/total*100).toFixed(3)}"
      title="${fmt(c.src_start)}"></div>`).join('');
  const rows=cuts.map((c,i)=>{
    const secs=(c.src_end-c.src_start).toFixed(1);
    const text=(c.text||'').replace(/</g,'&lt;') || '<span class="dim">no speech here</span>';
    return `<div class="seg ${c.enabled?'':'off'}" data-seg="${i}">
      <input type="checkbox" data-cut="${c.id}" ${c.enabled?'checked':''}>
      <span class="grow"><span class="t">${fmt(c.src_start)} · ${secs}s</span>
        <div class="txt" dir="auto">${text}</div></span></div>`;
  }).join('');
  const dropped=cuts.filter(c=>!c.enabled).length;
  $('detail').insertAdjacentHTML('beforeend',
    `<div class="card"><h2>Timeline</h2>
      <div class="strip">${bar}</div>
      <div class="dim" style="font-size:12px;margin-bottom:6px">
        Untick a segment to cut it. Times are in the original footage, so they
        keep meaning the same thing however else you change the edit${dropped
          ? ` — ${dropped} cut so far; tick one back to bring it and its words back`
          : ''}.</div>
      ${rows}
      <button id="applyTrim">Apply the trim</button>
      <div id="trimMsg" class="dim"></div>
    </div>`);

  document.querySelectorAll('[data-cut]').forEach(el=>el.onchange=()=>{
    const row=el.closest('.seg'), index=row.dataset.seg;
    row.classList.toggle('off', !el.checked);
    const bar=document.querySelector(`[data-bar="${index}"]`);
    if(bar) bar.classList.toggle('off', !el.checked);
  });

  $('applyTrim').onclick=async()=>{
    const keep={};
    document.querySelectorAll('[data-cut]').forEach(el=>keep[el.dataset.cut]=el.checked);
    $('applyTrim').disabled=true; $('trimMsg').textContent='queued…';
    try{
      await api('/api/jobs/'+job.id+'/segments',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({keep})});
      drawn=''; listed=''; refresh();
    }catch(e){
      $('trimMsg').textContent='error: '+e.message;
      $('applyTrim').disabled=false;
    }
  };
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

async function renderLook(job){
  try{ look=await api('/api/jobs/'+job.id+'/settings'); }catch(e){ return; }
  const groups=[...new Set(look.fields.map(f=>f.group))];
  const tabs=groups.map((g,i)=>`<button class="tab ${i?'':'on'}" data-tab="${g}">${g}</button>`).join('');
  const panes=groups.map((g,i)=>
    `<div data-pane="${g}" ${i?'hidden':''}>${look.fields.filter(f=>f.group===g).map(control).join('')}</div>`
  ).join('');
  const changed=look.changed.length
    ? `<div class="dim" style="margin-top:10px">changed from the template: ${look.changed.join(', ')}</div>` : '';
  $('detail').insertAdjacentHTML('beforeend',
    `<div class="card"><h2>Look</h2>
      <div class="tabs">${tabs}</div>${panes}
      <div class="dim" style="margin-top:10px;font-size:12px">
        Applying re-decides the edit with the transcript already taken, so it is
        the preview render you wait for, not the listening. Caption wording you
        typed is kept; the zoom and transition ticks reset, because a new setting
        gives you different moves.</div>
      <button id="applyLook">Apply and re-render</button>
      <button class="ghost" id="resetLook">Back to the template</button>
      <div id="lookMsg" class="dim"></div>
    </div>`);

  document.querySelectorAll('[data-tab]').forEach(el=>el.onclick=()=>{
    document.querySelectorAll('[data-tab]').forEach(t=>t.classList.toggle('on', t===el));
    document.querySelectorAll('[data-pane]').forEach(pane=>{
      pane.hidden = pane.dataset.pane !== el.dataset.tab;
    });
  });
  $('applyLook').onclick=()=>applyLook(job, false);
  $('resetLook').onclick=()=>applyLook(job, true);
}

async function applyLook(job, reset){
  const values={};
  if(!reset) document.querySelectorAll('[data-set]').forEach(el=>{
    values[el.dataset.set] = el.type==='checkbox' ? (el.checked?'true':'false') : el.value;
  });
  $('applyLook').disabled=$('resetLook').disabled=true;
  $('lookMsg').textContent='queued…';
  try{
    await api('/api/jobs/'+job.id+'/settings',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({values, reset})});
    drawn=''; listed=''; refresh();
  }catch(e){
    $('lookMsg').textContent='error: '+e.message;
    $('applyLook').disabled=$('resetLook').disabled=false;
  }
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

api('/api/me').then(d=>{
  show(d.authenticated);
  if(d.version) $('ver').textContent=d.version;
}).catch(()=>show(false));
</script></body></html>
"""
