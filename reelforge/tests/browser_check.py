"""End-to-end check of the editor in a real browser.

Not part of the test suite: it needs playwright and a Chromium, and takes a
minute. Run it by hand after changing the page:

    pip install playwright && python tests/browser_check.py

One caveat worth knowing before you read a failure here as a bug. Playwright's
Chromium is built without the proprietary codecs, so it cannot decode H.264 and
the <video> element stays empty - `codec support` below says so plainly. The
drawing code is therefore driven directly instead of by playback. Everything
else - the timeline, selections, trimming, saving - is exercised for real.

Its mouse also does not synthesise pointer events, so the drag is dispatched as
the PointerEvents a real browser would send.
"""
import os, subprocess, sys, tempfile, threading, time
from pathlib import Path

os.environ["REELFORGE_ASR_BACKEND"] = "stub"
tmp = Path(tempfile.mkdtemp(prefix="rf-ui-"))
clip = tmp / "take.mp4"
subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y",
    "-f","lavfi","-i","testsrc2=size=608x1080:rate=25:duration=6",
    "-f","lavfi","-i","sine=frequency=220:duration=6",
    "-filter_complex","[1:a]volume='if(between(t,0.4,2.5)+between(t,3.5,5.6),0.8,0.0)':eval=frame[a]",
    "-map","0:v","-map","[a]","-pix_fmt","yuv420p","-c:a","aac",str(clip)],
    check=True, capture_output=True)

import uvicorn
from reelforge.web import create_app
app = create_app(data_dir=tmp/"data", password="pw", secret="s")
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8777, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(60):
    time.sleep(0.5)
    if getattr(server, "started", False): break

from playwright.sync_api import sync_playwright
errors, logs = [], []
with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    page = browser.new_page(viewport={"width": 420, "height": 900})
    page.on("console", lambda m: (errors if m.type == "error" else logs).append(m.text))
    page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))

    page.goto("http://127.0.0.1:8777/")
    page.fill("#pw", "pw"); page.click("#loginBtn")
    page.wait_for_selector("#app:not([hidden])", timeout=15000)

    page.set_input_files("#files", str(clip))
    page.click("#upload")
    page.wait_for_selector(".job", timeout=30000)
    print("uploaded; waiting for the edit…")
    page.wait_for_selector(".pill.ready", timeout=240000)
    page.click(".job")
    page.wait_for_selector("#track .keep", timeout=30000)

    print("segments drawn   :", page.locator("#track .keep").count())
    print("cuts in plan     :", page.evaluate("P && P.plan.cuts ? P.plan.cuts.length : 'no P'"))
    print("enabled cuts     :", page.evaluate("P ? P.plan.cuts.filter(c=>c.enabled).length : 0"))
    print("P.duration       :", page.evaluate("P ? P.duration : null"))
    print("track html       :", page.evaluate("document.getElementById('track').innerHTML")[:160])
    print("proxy fetch      :", page.evaluate(
        "fetch('/api/jobs/'+P.job.id+'/proxy.mp4').then(r=>r.status)"))
    probe = page.evaluate('''(() => {
        const v = document.createElement('video');
        return {h264: v.canPlayType('video/mp4; codecs="avc1.42E01E"') || '(no)',
                aac: v.canPlayType('audio/mp4; codecs="mp4a.40.2"') || '(no)',
                webm: v.canPlayType('video/webm; codecs="vp8"') || '(no)'};
    })()''')
    print("codec support    :", probe)
    print("mounts so far    :", page.evaluate("window.__mounts||0"))
    print("P.track attached :", page.evaluate("P ? document.body.contains(P.track) : null"))
    print("video error      :", page.evaluate(
        "(()=>{const v=document.getElementById('pv');"
        "return {ready:v.readyState, net:v.networkState, err:v.error&&v.error.message};})()"))

    # This Chromium has no H.264, so the element cannot decode the proxy. Drive
    # the drawing code directly instead - that is the part worth checking.
    shot = page.evaluate("""(() => {
        const line = P.plan.captions[1] || P.plan.captions[0];
        const at = (line.start + line.end) / 2;
        paint(at, 0);
        const zoom = P.plan.zooms.find(z => z.enabled);
        if (zoom) paint((zoom.out_start + zoom.out_end) / 2, 0);
        return {caption: document.getElementById('caps').innerText,
                spans: document.querySelectorAll('#caps span').length,
                colours: [...document.querySelectorAll('#caps span')].map(s => s.style.color),
                font: getComputedStyle(document.getElementById('caps')).fontFamily,
                size: getComputedStyle(document.getElementById('caps')).fontSize,
                transform: document.getElementById('pv').style.transform,
                clock: document.getElementById('clock').innerText};
    })()""")
    for key, value in shot.items():
        print(f"  {key:10}:", value)

    # The settings are icons under the video; one click opens that group there.
    page.wait_for_selector(".iconbar [data-group='Captions']", timeout=15000)
    print("setting groups   :", page.locator(".iconbar button").count())
    print("pane starts shut :", page.inner_text("#lookPane") == "")
    page.click(".iconbar [data-group='Captions']")
    page.wait_for_selector("[data-set='captions.style']", timeout=15000)
    print("font is reachable:", page.locator("[data-set='captions.font']").count() == 1)

    # Weight is a real control now: pick light, and the live caption goes light.
    page.select_option("[data-set='captions.weight']", "300")
    page.wait_for_timeout(300)
    weight = page.evaluate("""() => { const l=P.plan.captions[0]; paint(l.start+0.05, 0);
        return getComputedStyle(document.getElementById('caps')).fontWeight; }""")
    print("live weight      :", weight)
    assert weight == "300", f"weight did not apply live: {weight}"
    page.select_option("[data-set='captions.style']", "box")
    page.wait_for_timeout(600)
    print("dirty banner     :", repr(page.inner_text("#dirty")[:60]))
    print("save enabled     :", page.is_enabled("#saveEdit"))

    # Clicking the same icon closes it again, so the video comes back into view.
    page.click(".iconbar [data-group='Captions']")
    print("pane shuts again :", page.inner_text("#lookPane") == "")

    # Pause mode: markers appear on the joins and can be dragged.
    page.click("#pauseMode")
    page.wait_for_timeout(300)
    joins = page.locator("#track .join")
    print("pause markers    :", joins.count())
    if joins.count():
        spot = joins.first.bounding_box()
        drag = page.evaluate("""(box) => {
            const el = document.querySelector('#track .join');
            const send = (type, dx) => el.dispatchEvent(new PointerEvent(type, {
                bubbles: true, cancelable: true, pointerId: 3, isPrimary: true,
                clientX: box.x + box.width / 2 + dx, clientY: box.y + box.height / 2}));
            send('pointerdown', 0); send('pointermove', 26); send('pointerup', 26);
            return new Promise(done => requestAnimationFrame(() => done(
                P.pauses.map(([m, d]) => [Math.round(m * 100) / 100,
                                          Math.round(d * 100) / 100]))));
        }""", spot)
        print("pause after drag :", drag)
    page.click("#pauseMode")

    # Put a clip in the library and tag it with something actually said, so
    # there is an overlay to walk through.
    page.set_input_files("#brollFiles", str(clip))
    page.click("#brollUpload")
    page.wait_for_selector("[data-kw]", timeout=120000)
    spoken = page.evaluate("""() => {
        const line = (P.plan.captions || []).find(l => l.start > 1.5);
        return line ? line.text.split(/\s+/).find(w => w.length > 3) : null;
    }""")
    print("tagging with     :", spoken)
    if spoken:
        page.fill("[data-kw]", spoken)
        page.dispatch_event("[data-kw]", "change")
        page.wait_for_function("P && (P.plan.overlays||[]).length > 0", timeout=60000)
        print("overlays now     :", page.evaluate("P.plan.overlays.length"))
        print("library verdict  :", page.inner_text(".asset .nm")[-24:])

    # B-roll: how many requests does one appearance cost? The first version
    # re-seeked every frame, so an unloaded clip sent a range request per frame
    # until the server had no thread left to serve the page itself.
    page.evaluate("""() => { window.__hits = 0;
        const real = window.fetch;
        window.__reqs = [];
    }""")
    hits = []
    page.on("request", lambda r: hits.append(r.url) if "/broll/" in r.url else None)
    overlay = page.evaluate("""() => {
        if (!P.plan.overlays || !P.plan.overlays.length) return null;
        const o = P.plan.overlays[0];
        // Walk the whole overlay as the player would, frame by frame.
        for (let t = o.out_start; t <= o.out_end; t += 1 / 60) paint(t, 0);
        return {id: o.id, name: o.name, span: +(o.out_end - o.out_start).toFixed(2)};
    }""")
    page.wait_for_timeout(1200)
    print("overlay walked   :", overlay)
    if overlay:
        # The property said hidden three times while the clip stayed on screen,
        # because a stylesheet rule outranked the browser's own [hidden]. So this
        # asks the screen: computed display and a real bounding box.
        gone = page.evaluate("""() => {
            const o = P.plan.overlays[0], bv = document.getElementById('bv');
            paint(o.out_end + 1.0, 0);
            const r = bv.getBoundingClientRect();
            return {display: getComputedStyle(bv).display,
                    onScreen: r.width > 0 && r.height > 0};
        }""")
        print("after its window :", gone)
        assert gone["display"] == "none" and not gone["onScreen"], \
            f"b-roll still on screen after its window: {gone}"
    print("broll requests   :", len(hits), "(one appearance)")
    if overlay:
        frames = int(overlay["span"] * 60)
        print("frames drawn     :", frames)
        assert len(hits) <= 4, f"{len(hits)} requests for {frames} frames — flooding"

    # A word fixed in a caption box shows on the video as it is typed, enables
    # Save, survives a settings change, and is what gets saved.
    page.wait_for_selector("[data-cap='0']", timeout=15000)
    original = page.get_attribute("[data-cap='0']", "value")
    words = original.split()
    words[0] = "مُصَحَّح"
    fixed = " ".join(words)
    page.fill("[data-cap='0']", fixed)
    page.dispatch_event("[data-cap='0']", "input")
    shown = page.evaluate("""(want) => {
        const line = P.plan.captions[0];
        paint(line.start + 0.05, 0);
        return {planText: line.text === want,
                onVideo: document.getElementById('caps').innerText.includes(want.split(' ')[0]),
                saveEnabled: !document.getElementById('saveEdit').disabled};
    }""", fixed)
    print("caption edit live:", shown)
    assert shown["planText"] and shown["onVideo"] and shown["saveEnabled"], shown

    # A settings change re-plans on the server; the wording must come back.
    page.click(".iconbar [data-group='Captions']")
    page.wait_for_selector("[data-set='captions.max_words']", timeout=10000)
    page.fill("[data-set='captions.max_words']", "3")
    page.dispatch_event("[data-set='captions.max_words']", "change")
    page.wait_for_timeout(2500)
    kept = page.evaluate("""(w) => (P.plan.captions||[]).some(l => l.text.includes(w))""",
                         "مُصَحَّح")
    print("survives replan  :", kept)
    assert kept, "the typed word was lost when settings changed"
    page.click(".iconbar [data-group='Captions']")

    job_id = page.evaluate("P.job.id")
    page.click("#saveEdit")
    # Saving re-decides the edit on the machine; wait for that, not the panel,
    # which is rebuilt in the meantime.
    page.wait_for_function("""(id) => fetch('/api/jobs/'+id).then(r=>r.json())
        .then(j => j.status === 'ready')""", arg=job_id, timeout=90000)
    page.wait_for_timeout(800)
    saved = page.evaluate("""([id, w]) => fetch('/api/jobs/'+id+'/edl').then(r=>r.json())
        .then(e => e.captions.some(l => l.text.includes(w)))""", [job_id, "مُصَحَّح"])
    print("saved on server  :", saved)
    assert saved, "Save did not keep the typed word"

    # Saving a default look, so the next upload arrives already set up.
    page.click(".iconbar [data-group='__defaults']")
    page.wait_for_selector("#saveDefaults", timeout=10000)
    page.click("#saveDefaults")
    page.wait_for_timeout(1200)
    print("defaults saved   :", repr(page.inner_text("#defaultsMsg")[:70]))
    page.click(".iconbar [data-group='__defaults']")

    # Drag a selection across the timeline.
    page.wait_for_timeout(1500)                    # let any pending redraw settle
    box = page.locator("#track").bounding_box()
    fired = page.evaluate("""(() => {
        const t = document.getElementById('track');
        const box = t.getBoundingClientRect();
        const seen = [];
        ['pointerdown','pointermove','pointerup'].forEach(k =>
            t.addEventListener(k, () => seen.push(k), {once: true}));
        const at = frac => new PointerEvent(arguments, {});
        const send = (type, frac) => t.dispatchEvent(new PointerEvent(type, {
            bubbles: true, cancelable: true, pointerId: 1, isPrimary: true,
            clientX: box.left + box.width * frac, clientY: box.top + box.height / 2}));
        send('pointerdown', 0.30);
        send('pointermove', 0.45);
        send('pointermove', 0.60);
        send('pointerup', 0.60);
        return new Promise(done => requestAnimationFrame(
            () => done({seen, sel: P.sel})));
    })()""")
    print("events seen      :", fired)
    page.wait_for_timeout(400)
    print("selection label  :", repr(page.inner_text("#track .lbl")[:70]))
    print("grips drawn      :", page.locator("#track .grip").count())

    before = page.inner_text("#statOut")
    page.click("#cutSel")
    page.wait_for_timeout(1800)
    print("length before/after cut:", before, "->", page.inner_text("#statOut"))
    print("segments after cut     :", page.locator("#track .keep").count())

    page.click("#saveEdit")
    page.wait_for_timeout(3000)
    print("after save, banner:", repr(page.inner_text("#dirty")[:60]))
    browser.close()

print("\nconsole errors:", errors if errors else "none")
server.should_exit = True
