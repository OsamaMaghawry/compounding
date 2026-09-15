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
        self.queue: queue.Queue[str] = queue.Queue()
        self.editors: dict[str, AutoEditor] = {}
        self.edls: dict[str, EDL] = {}
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def submit(self, job_id: str) -> None:
        self.queue.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.store.get(job_id)
            if job is None:
                continue
            try:
                self._process(job)
            except Exception as exc:                      # a failed job must not kill the worker
                self.store.update(job, status="error", stage="failed",
                                  error=f"{type(exc).__name__}: {exc}"[:400])

    def _process(self, job: Job) -> None:
        directory = self.store.dir(job.id)

        def note(message: str) -> None:
            job.progress.append(message)
            self.store.update(job, stage=message)

        self.store.update(job, status="working", stage="starting", error="")

        profile = StyleProfile.resolve(job.template or None,
                                       [PACKAGE_ROOT / "templates", PACKAGE_ROOT / "profiles"])
        overrides = [f"asr.model={job.model}"]
        # Lets a deployment pin the speech backend - and lets the tests run
        # without downloading a model.
        backend = os.environ.get("REELFORGE_ASR_BACKEND")
        if backend:
            overrides.append(f"asr.backend={backend}")
        profile = profile.apply_overrides(overrides)
        editor = AutoEditor(profile, project_dir=directory / "project",
                            fonts_dir=self.fonts_dir, on_status=note)
        self.editors[job.id] = editor

        clips = [Path(p) for p in job.sources]
        if len(clips) > 1:
            from .join import join_clips  # noqa: PLC0415
            source = join_clips(clips, editor.work_dir / "joined.mp4",
                                max_height=int(profile.get("output.height")
                                               * float(profile.get("output.zoom_headroom"))),
                                on_status=note)
        else:
            source = clips[0]

        result = editor.plan(source)
        self.edls[job.id] = result.edl
        note("rendering preview")
        editor.render(result.edl, directory / "preview.mp4", preview=True)

        self.store.update(job, status="ready", stage="ready to review",
                          run_id=result.run_id, summary=result.edl.summary())

    def rerender(self, job: Job) -> None:
        editor, edl = self.editors.get(job.id), self.edls.get(job.id)
        if not editor or not edl:
            raise RuntimeError("this job is no longer loaded - re-upload to edit it")
        editor.render(edl, self.store.dir(job.id) / "preview.mp4", preview=True)
        self.store.update(job, summary=edl.summary())

    def export(self, job: Job) -> Path:
        editor, edl = self.editors.get(job.id), self.edls.get(job.id)
        if not editor or not edl:
            raise RuntimeError("this job is no longer loaded - re-upload to edit it")
        self.store.update(job, status="exporting", stage="rendering the final video")
        name = f"{Path(job.title).stem or 'reel'}-reel.mp4"
        output = self.store.dir(job.id) / name
        editor.render(edl, output, preview=False)
        if job.run_id:
            try:
                editor.accept(job.run_id, edl)
            except Exception:
                pass                                       # learning must never block a download
        self.store.update(job, status="done", stage="finished", output_name=name)
        return output


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
        return {"authenticated": valid_token(secret, request.cookies.get(COOKIE))}

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
        return [job.to_dict() for job in store.list()]

    @app.post("/api/jobs", dependencies=[Depends(require_login)])
    async def create_job(files: list[UploadFile], template: str = Form(""),
                         model: str = Form("small")) -> dict:
        usable = [f for f in files if Path(f.filename or "").suffix.lower() in VIDEO_SUFFIXES]
        if not usable:
            raise HTTPException(status_code=400, detail="no video files in that upload")

        title = Path(usable[0].filename or "reel").stem
        job = store.create(title=title, template=template, model=model)
        uploads = store.dir(job.id) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)

        total = 0
        saved: list[str] = []
        for index, upload in enumerate(usable):
            name = f"{index:02d}-{Path(upload.filename or 'clip').name}"
            target = uploads / name
            with target.open("wb") as handle:
                while chunk := await upload.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        store.delete(job.id)
                        raise HTTPException(status_code=413, detail="upload too large")
                    handle.write(chunk)
            saved.append(str(target))

        store.update(job, sources=saved)
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
        job_or_404(job_id)
        edl = runner.edls.get(job_id)
        if edl is None:
            raise HTTPException(status_code=409, detail="not planned yet")
        return edl.to_dict()

    @app.post("/api/jobs/{job_id}/edl", dependencies=[Depends(require_login)])
    def update_edl(job_id: str, body: dict) -> dict:
        from .captions import apply_text_edit  # noqa: PLC0415
        job = job_or_404(job_id)
        edl = runner.edls.get(job_id)
        if edl is None:
            raise HTTPException(status_code=409, detail="not planned yet")

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

        if body.get("rerender"):
            runner.rerender(job)
        return {"ok": True, "summary": edl.summary()}

    @app.post("/api/jobs/{job_id}/export", dependencies=[Depends(require_login)])
    def export_job(job_id: str) -> dict:
        job = job_or_404(job_id)
        output = runner.export(job)
        return {"ok": True, "download": f"/api/jobs/{job_id}/download",
                "name": output.name}

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
let current=null, edl=null, poll=null;
const fmt=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;

async function api(path, opts={}){
  const r = await fetch(path, {credentials:'same-origin', ...opts});
  if(r.status===401){ show(false); throw new Error('please log in'); }
  if(!r.ok){ throw new Error((await r.json().catch(()=>({detail:r.statusText}))).detail); }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r;
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
  const files=$('files').files;
  if(!files.length){ $('msg').textContent='choose at least one video'; return; }
  const body=new FormData();
  for(const f of files) body.append('files', f);
  body.append('template', $('template').value);
  body.append('model', $('model').value);
  $('upload').disabled=true; $('msg').textContent='uploading…';
  try{
    const job=await api('/api/jobs',{method:'POST',body});
    $('msg').textContent='uploaded — editing has started';
    $('files').value=''; current=job.id; refresh();
  }catch(e){ $('msg').textContent='error: '+e.message; }
  finally{ $('upload').disabled=false; }
};

async function refresh(){
  let jobs=[];
  try{ jobs=await api('/api/jobs'); }catch(e){ return; }
  $('jobs').innerHTML = jobs.length ? jobs.map(j=>{
    const cls = j.status==='ready'||j.status==='done' ? 'ready' : (j.status==='error'?'error':'working');
    return `<div class="job" data-id="${j.id}">
      <span class="grow"><b>${j.title}</b><br><span class="dim">${j.stage}</span></span>
      <span class="pill ${cls}">${j.status}</span></div>`;
  }).join('') : '<span class="dim">none yet</span>';
  $('jobs').querySelectorAll('.job').forEach(el=>el.onclick=()=>{current=el.dataset.id;draw();});
  if(current) draw(jobs.find(j=>j.id===current));
  const busy = jobs.some(j=>j.status==='working'||j.status==='queued'||j.status==='exporting');
  clearTimeout(poll); poll=setTimeout(refresh, busy?2500:15000);
}

async function draw(job){
  if(!job){ try{ job=await api('/api/jobs/'+current); }catch(e){ return; } }
  const s=job.summary||{};
  let html=`<div class="card"><h2>${job.title}</h2>`;
  if(job.status==='error'){ html+=`<div style="color:var(--bad)">${job.error||job.stage}</div>`; }
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

  if(job.status==='ready'||job.status==='done'){
    try{ edl=await api('/api/jobs/'+job.id+'/edl'); }catch(e){ edl=null; }
    if(edl) renderControls(job);
    const rr=$('rerender'), ex=$('export');
    if(rr) rr.onclick=()=>send(job,true);
    if(ex) ex.onclick=async()=>{ ex.disabled=true; ex.textContent='rendering…';
      try{ await send(job,false); await api('/api/jobs/'+job.id+'/export',{method:'POST'}); refresh(); }
      catch(e){ ex.textContent='error: '+e.message; ex.disabled=false; } };
  }
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

async function send(job, rerender){
  const body={rerender, captions:[], zooms:[], overlays:(edl.overlays||[]), transitions:[]};
  document.querySelectorAll('[data-cap]').forEach(el=>body.captions[el.dataset.cap]={text:el.value});
  document.querySelectorAll('[data-zoom]').forEach(el=>body.zooms[el.dataset.zoom]={enabled:el.checked});
  document.querySelectorAll('[data-tr]').forEach(el=>body.transitions[el.dataset.tr]={enabled:el.checked});
  const r=await api('/api/jobs/'+job.id+'/edl',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(rerender) draw();
  return r;
}

api('/api/me').then(d=>show(d.authenticated)).catch(()=>show(false));
</script></body></html>
"""
