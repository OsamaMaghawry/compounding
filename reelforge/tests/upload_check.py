"""End-to-end check of uploading many clips in a real browser.

Not part of the test suite: it needs playwright and a Chromium. Run it by hand
after changing the upload path:

    python tests/upload_check.py

It answers the question that started this: thirty short takes, chosen at once,
all arrive and the edit starts. Then it breaks one of them on purpose and checks
that the twenty-nine that arrived are not thrown away with it.
"""
import os, subprocess, sys, tempfile, threading, time
from pathlib import Path

os.environ["REELFORGE_ASR_BACKEND"] = "stub"
tmp = Path(tempfile.mkdtemp(prefix="rf-up-"))

CLIPS = int(os.environ.get("CLIPS", "30"))
clips = []
for i in range(CLIPS):
    path = tmp / f"IMG_{4000 + i}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=608x1080:rate=25:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
        "-filter_complex", "[1:a]volume='if(between(t,0.4,2.5)+between(t,3.5,5.6),0.8,0.0)'"
                           ":eval=frame[a]",
        "-map", "0:v", "-map", "[a]", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True)
    clips.append(str(path))
print(f"made {len(clips)} clips, {sum(Path(c).stat().st_size for c in clips) / 1e6:.0f} MB")

import uvicorn
from reelforge.web import create_app
app = create_app(data_dir=tmp / "data", password="pw", secret="s")
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8778, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(60):
    time.sleep(0.5)
    if getattr(server, "started", False):
        break

from playwright.sync_api import sync_playwright


def poll(read, until, tries=240, gap=0.5):
    """Wait from here, not from inside the page.

    A predicate handed to wait_for_function that returns a promise can be read
    as satisfied before its fetch has answered, which reads as a value going
    backwards. Reading the answer here removes the question.
    """
    value = None
    for _ in range(tries):
        value = read()
        if until(value):
            return value
        time.sleep(gap)
    raise AssertionError(f"never settled, last was {value!r}")


errors = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(
        executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    page = browser.new_page(viewport={"width": 420, "height": 900})
    page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))

    page.goto("http://127.0.0.1:8778/")
    page.fill("#pw", "pw"); page.click("#loginBtn")
    page.wait_for_selector("#app:not([hidden])", timeout=15000)

    # -- all of them, chosen at once -------------------------------------
    page.set_input_files("#files", clips)
    print("chosen line     :", repr(page.inner_text("#chosen")))
    assert f"{CLIPS} clip" in page.inner_text("#chosen")

    started = time.time()
    page.click("#upload")
    page.wait_for_function("document.getElementById('msg').textContent.includes('editing has started')",
                           timeout=600000)
    print(f"uploaded {CLIPS} clips in {time.time() - started:.0f}s")

    job = page.evaluate("fetch('/api/jobs').then(r=>r.json()).then(j=>j[0])")
    print("clips on server :", len(job["clips"]))
    print("order kept      :", job["clips"][:3], "…", job["clips"][-2:])
    assert len(job["clips"]) == CLIPS, job["clips"]
    assert job["clips"] == [f"IMG_{4000 + i}.mp4" for i in range(CLIPS)]
    assert job["status"] in ("queued", "working"), job["status"]

    # -- one bad file must not take the good ones with it ----------------
    bad = tmp / "notes.txt"
    bad.write_text("this is not a video", encoding="utf-8")
    page.set_input_files("#files", [clips[0], str(bad), clips[1]])
    page.click("#upload")
    page.wait_for_selector("#retryClips", timeout=120000)
    print("partial message :", repr(page.inner_text("#msg")[:120].replace("\n", " ")))
    assert "2 clip(s) arrived" in page.inner_text("#msg")
    assert "not a video" in page.inner_text("#msg")

    kept = page.evaluate("fetch('/api/jobs').then(r=>r.json()).then(j=>j.length)")
    print("jobs on server  :", kept, "(the part-uploaded one is still there)")
    assert kept == 2, kept

    page.click("#startAnyway")
    two = poll(lambda: page.evaluate("""fetch('/api/jobs').then(r=>r.json())
        .then(j => j.map(x => [x.clips.length, x.status]))"""),
        lambda v: all(st != "uploading" for _, st in v) and len(v) == 2)
    print("both edits      :", two)
    assert sorted(n for n, _ in two) == [2, CLIPS], two

    # -- a clip big enough to go in several pieces, sent at once ---------
    import hashlib
    big = tmp / "LONG_TAKE.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=25",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=25",
        "-map", "0:v", "-map", "1:a", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "8M",
        "-c:a", "aac", str(big)], check=True, capture_output=True)
    pieces = -(-big.stat().st_size // (6 * 1024 * 1024))
    print(f"big take        : {big.stat().st_size/1e6:.0f} MB, {pieces} pieces")
    assert pieces >= 3, "make it bigger or this proves nothing about sending several at once"

    page.set_input_files("#files", [str(big)])
    page.click("#upload")
    page.wait_for_function(
        "document.getElementById('msg').textContent.includes('editing has started')",
        timeout=300000)
    landed = poll(lambda: page.evaluate(
        "fetch('/api/jobs').then(r=>r.json()).then(j=>j.filter(x=>x.clips.length===1).length)"),
        lambda n: n >= 1)
    on_disk = sorted((tmp / "data" / "jobs").rglob("*LONG_TAKE.mp4"))
    assert on_disk, "the big take never landed"
    same = hashlib.sha256(on_disk[0].read_bytes()).hexdigest() == \
           hashlib.sha256(big.read_bytes()).hexdigest()
    print(f"arrived intact  : {same} ({on_disk[0].stat().st_size/1e6:.0f} MB)")
    assert same, "the pieces did not reassemble into the same file"

    # -- an upload cut off halfway can be picked up again ----------------
    cut = page.evaluate("""() => fetch('/api/jobs', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({template:'', model:'small', title:'cut off'})})
        .then(r=>r.json()).then(j=>j.id)""")
    page.evaluate("""(id) => {
        const body=new FormData();
        body.append('name','IMG_9000.mp4'); body.append('index','0');
        body.append('offset','0'); body.append('final','true');
        return fetch('/api/jobs/'+id+'/proxy.mp4');
    }""", cut)
    # Send one clip through the API the way the page would, then walk away.
    page.set_input_files("#files", [clips[0]])
    page.evaluate("""([id]) => {
        const f = document.getElementById('files').files[0];
        const body = new FormData();
        body.append('name', f.name); body.append('index', '0');
        body.append('offset', '0'); body.append('final', 'true');
        body.append('file', f, f.name);
        return fetch('/api/jobs/'+id+'/chunk', {method:'POST', body}).then(r=>r.status);
    }""", [cut])
    page.evaluate("(id)=>{ current=id; drawn=''; listed=''; refresh(); }", cut)
    page.wait_for_selector("#addMore", timeout=30000)
    print("resume panel    :", repr(page.inner_text("#detail")[:90].replace("\n", " ")))
    assert "1 clip(s) arrived" in page.inner_text("#detail")

    page.set_input_files("#moreFiles", [clips[1], clips[2]])
    page.click("#addMore")
    # Poll from Python rather than inside the page: a predicate that returns a
    # promise can be read as "done" before its fetch has answered.
    got = poll(lambda: page.evaluate(
        "(id)=>fetch('/api/jobs/'+id).then(r=>r.json()).then(j=>j.clips)", cut),
        lambda v: len(v) == 3)
    print("after adding    :", got)
    assert got == ["IMG_4000.mp4", "IMG_4001.mp4", "IMG_4002.mp4"], got
    page.wait_for_selector("#startWith", timeout=30000)
    page.click("#startWith")
    poll(lambda: page.evaluate(
        "(id)=>fetch('/api/jobs/'+id).then(r=>r.json()).then(j=>j.status)", cut),
        lambda st: st != "uploading")
    print("resumed edit    : started with", len(got), "clips")

    browser.close()

print("\npage errors:", errors if errors else "none")
server.should_exit = True
print("OK")
