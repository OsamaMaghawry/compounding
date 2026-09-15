"""A local review page - the surface that turns your corrections into training data.

Runs on 127.0.0.1 with the Python standard library only. Nothing is uploaded and
nothing is served beyond the files this run produced.
"""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .captions import apply_text_edit
from .edl import EDL
from .pipeline import AutoEditor

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReelForge review</title>
<style>
:root{--bg:#0d0d13;--card:#16161f;--line:#262633;--fg:#f2f2f7;--dim:#9a9aae;--accent:#ffd24a;--ok:#3ddc97}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.wrap{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:20px;padding:20px;max-width:1400px;margin:0 auto}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
video{width:100%;border-radius:14px;background:#000;display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:14px}
h1{font-size:17px;margin-bottom:2px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:10px}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px;word-break:break-all}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}
.stat{background:#1d1d28;border-radius:10px;padding:8px}
.stat b{display:block;font-size:17px}.stat span{font-size:11px;color:var(--dim)}
.row{display:flex;align-items:center;gap:10px;padding:8px;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.row .t{color:var(--dim);font-variant-numeric:tabular-nums;font-size:12px;min-width:92px}
.row .meta{font-size:12px;color:var(--dim)}
.grow{flex:1;min-width:0}
input[type=text]{width:100%;background:#10101a;border:1px solid var(--line);color:var(--fg);
  border-radius:8px;padding:8px 10px;font-size:15px;font-family:inherit}
input[type=text]:focus{outline:none;border-color:var(--accent)}
input[dir=rtl]{text-align:right}
input[type=checkbox]{width:17px;height:17px;accent-color:var(--accent);flex:none}
button{background:var(--accent);color:#18181f;border:0;border-radius:10px;padding:10px 16px;
  font-weight:650;font-size:14px;cursor:pointer;font-family:inherit}
button.ghost{background:#242433;color:var(--fg)}
button:disabled{opacity:.5;cursor:default}
.actions{display:flex;gap:10px;flex-wrap:wrap}
#status{margin-top:10px;font-size:13px;color:var(--dim);min-height:20px}
.badge{font-size:11px;background:#242433;border-radius:6px;padding:2px 7px;color:var(--dim)}
.learned{background:#12251c;border-color:#1f4534;color:var(--ok)}
</style></head><body>
<div class="wrap">
  <div>
    <div class="card">
      <h1>ReelForge review</h1>
      <div class="sub" id="ver"></div>
      <div class="sub" id="src"></div>
      <video id="player" controls playsinline></video>
    </div>
    <div class="card">
      <h2>Result</h2>
      <div class="stats" id="stats"></div>
      <div class="actions">
        <button id="rerender" class="ghost">Re-render preview</button>
        <button id="accept">Approve &amp; export</button>
      </div>
      <div id="status">Toggle anything you don't like, fix any caption, then approve.
        Your changes train the next run.</div>
    </div>
  </div>
  <div>
    <div class="card"><h2>Captions</h2><div id="captions"></div></div>
    <div class="card"><h2>Zoom moves</h2><div id="zooms"></div></div>
    <div class="card"><h2>B-roll layers</h2><div id="overlays"></div></div>
    <div class="card"><h2>Transitions</h2><div id="transitions"></div></div>
  </div>
</div>
<script>
let edl=null, runId=null;
const $=id=>document.getElementById(id);
const fmt=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}.${String(Math.floor(s*100%100)).padStart(2,'0')}`;

async function load(){
  const r=await fetch('/api/state'); const d=await r.json();
  edl=d.edl; runId=d.run_id;
  $('src').textContent=d.source;
  $('ver').textContent='version '+(d.version||'?');
  $('player').src='/preview.mp4?v='+Date.now();
  renderStats(d.summary); renderCaptions(); renderZooms(); renderOverlays(); renderTransitions();
}
function renderStats(s){
  $('stats').innerHTML=[
    ['<b>'+s.output_duration+'s</b><span>from '+s.source_duration+'s</span>'],
    ['<b>'+s.removed+'s</b><span>dead air cut</span>'],
    ['<b>'+s.cuts+'</b><span>segments</span>'],
    ['<b>'+s.zooms+'</b><span>zoom moves</span>'],
    ['<b>'+s.overlays+'</b><span>b-roll</span>'],
    ['<b>'+s.words+'</b><span>words</span>'],
  ].map(x=>'<div class="stat">'+x+'</div>').join('');
}
function renderCaptions(){
  $('captions').innerHTML = edl.captions.length? edl.captions.map((c,i)=>
    `<div class="row"><span class="t">${fmt(c.start)}</span>
     <span class="grow"><input type="text" dir="auto" data-i="${i}" value="${c.text.replace(/"/g,'&quot;')}"></span></div>`
  ).join('') : '<div class="meta">No captions in this edit.</div>';
  $('captions').querySelectorAll('input').forEach(el=>{
    el.onchange=()=>{edl.captions[el.dataset.i].text=el.value;};
  });
}
function renderZooms(){
  $('zooms').innerHTML = edl.zooms.length? edl.zooms.map((z,i)=>
    `<div class="row"><input type="checkbox" data-i="${i}" ${z.enabled?'checked':''}>
     <span class="t">${fmt(z.out_start)}</span>
     <span class="grow meta">${z.kind.replace('_',' ')} &rarr; ${z.end_factor.toFixed(2)}x</span>
     <span class="badge">score ${z.score.toFixed(2)}</span></div>`
  ).join('') : '<div class="meta">No zoom moves.</div>';
  $('zooms').querySelectorAll('input').forEach(el=>{
    el.onchange=()=>{edl.zooms[el.dataset.i].enabled=el.checked;};
  });
}
function renderOverlays(){
  $('overlays').innerHTML = edl.overlays.length? edl.overlays.map((o,i)=>
    `<div class="row"><input type="checkbox" data-i="${i}" ${o.enabled?'checked':''}>
     <span class="t">${fmt(o.out_start)}</span>
     <span class="grow meta">${o.asset.split('/').pop()}</span>
     <span class="badge">${o.keyword}</span></div>`
  ).join('') : '<div class="meta">No b-roll matched. Add clips to your library folder.</div>';
  $('overlays').querySelectorAll('input').forEach(el=>{
    el.onchange=()=>{edl.overlays[el.dataset.i].enabled=el.checked;};
  });
}
function renderTransitions(){
  $('transitions').innerHTML = edl.transitions.length? edl.transitions.map((t,i)=>
    `<div class="row"><input type="checkbox" data-i="${i}" ${t.enabled?'checked':''}>
     <span class="t">${fmt(t.out_time)}</span>
     <span class="grow meta">${t.kind}</span>
     <span class="badge">${Math.round(t.duration*1000)}ms</span></div>`
  ).join('') : '<div class="meta">No transitions.</div>';
  $('transitions').querySelectorAll('input').forEach(el=>{
    el.onchange=()=>{edl.transitions[el.dataset.i].enabled=el.checked;};
  });
}
async function post(url){
  $('status').textContent='Working...'; $('rerender').disabled=$('accept').disabled=true;
  try{
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({run_id:runId,edl:edl})});
    const d=await r.json();
    if(d.error){$('status').textContent='Error: '+d.error; return null;}
    return d;
  }catch(e){$('status').textContent='Error: '+e.message; return null;}
  finally{$('rerender').disabled=$('accept').disabled=false;}
}
$('rerender').onclick=async()=>{
  const d=await post('/api/rerender'); if(!d)return;
  $('player').src='/preview.mp4?v='+Date.now();
  renderStats(d.summary);
  $('status').textContent='Preview updated in '+d.seconds+'s.';
};
$('accept').onclick=async()=>{
  const d=await post('/api/accept'); if(!d)return;
  const l=d.learned||{};
  $('status').innerHTML='Exported <b>'+d.output+'</b>.<br>Learned: '
    +(l.zooms_kept||0)+' zooms kept, '+(l.zooms_dropped||0)+' dropped'
    +(l.vocab_added?', '+l.vocab_added+' new vocabulary fixes':'')
    +(l.model?'. Zoom model retrained on '+l.model.samples+' examples (accuracy '+l.model.accuracy+').':'.');
  document.querySelector('#status').parentElement.classList.add('learned');
};
load();
</script></body></html>
"""


class ReviewServer:
    """Serves the review page and applies whatever you change there."""

    def __init__(self, editor: AutoEditor, plan_result, *, output: Path,
                 preview: Path, host: str = "127.0.0.1", port: int = 8733):
        self.editor = editor
        self.edl = plan_result.edl
        self.run_id = plan_result.run_id
        self.output = Path(output)
        self.preview = Path(preview)
        self.host = host
        self.port = port
        self.done = threading.Event()
        self.result: dict = {}

    # -- edit application -------------------------------------------------
    def apply_client_edl(self, payload: dict) -> EDL:
        """Merge the browser's toggles and caption text into the real EDL."""
        for index, zoom in enumerate(payload.get("zooms", [])):
            if index < len(self.edl.zooms):
                self.edl.zooms[index].enabled = bool(zoom.get("enabled", True))
        for index, overlay in enumerate(payload.get("overlays", [])):
            if index < len(self.edl.overlays):
                self.edl.overlays[index].enabled = bool(overlay.get("enabled", True))
        for index, transition in enumerate(payload.get("transitions", [])):
            if index < len(self.edl.transitions):
                self.edl.transitions[index].enabled = bool(transition.get("enabled", True))
        for index, caption in enumerate(payload.get("captions", [])):
            if index < len(self.edl.captions):
                text = (caption.get("text") or "").strip()
                if text and text != self.edl.captions[index].text:
                    apply_text_edit(self.edl.captions[index], text)
        return self.edl

    def serve(self, *, open_browser: bool = True) -> dict:
        server = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://{self.host}:{self.port}/"
        print(f"  review at {url}  (Ctrl+C when you are done)")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            while not self.done.wait(0.4):
                pass
        except KeyboardInterrupt:
            print("\n  review closed")
        finally:
            server.shutdown()
        return self.result


def _make_handler(app: ReviewServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # keep the console quiet
            pass

        # -- helpers -----------------------------------------------------
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_video(self, path: Path) -> None:
            if not path.exists():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            status = 200
            header = self.headers.get("Range")
            if header:
                match = RANGE_RE.match(header)
                if match:
                    if match.group(1):
                        start = int(match.group(1))
                    if match.group(2):
                        end = min(int(match.group(2)), size - 1)
                    status = 206
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))
            length = end - start + 1

            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(262144, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        # -- routes ------------------------------------------------------
        def do_GET(self):  # noqa: N802 - stdlib naming
            route = urlparse(self.path).path
            if route == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/state":
                from .cli import _checkout_revision  # noqa: PLC0415
                self._send_json({
                    "run_id": app.run_id,
                    "version": _checkout_revision(),
                    "source": app.edl.source,
                    "summary": app.edl.summary(),
                    "edl": app.edl.to_dict(),
                })
            elif route == "/preview.mp4":
                self._send_video(app.preview)
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            route = urlparse(self.path).path
            payload = self._read_json()
            edl_payload = payload.get("edl") or {}

            if route == "/api/rerender":
                import time  # noqa: PLC0415
                started = time.time()
                try:
                    app.apply_client_edl(edl_payload)
                    app.editor.render(app.edl, app.preview, preview=True)
                except Exception as exc:  # surface the real reason in the page
                    self._send_json({"error": str(exc)[:400]}, status=200)
                    return
                self._send_json({"ok": True, "summary": app.edl.summary(),
                                 "seconds": round(time.time() - started, 1)})

            elif route == "/api/accept":
                try:
                    app.apply_client_edl(edl_payload)
                    app.editor.render(app.edl, app.output, preview=False)
                    learned = app.editor.accept(app.run_id, app.edl)
                except Exception as exc:
                    self._send_json({"error": str(exc)[:400]}, status=200)
                    return
                app.result = {"output": str(app.output), "learned": learned}
                self._send_json({"ok": True, "output": app.output.name, "learned": learned})
                app.done.set()
            else:
                self.send_error(404)

    return Handler
