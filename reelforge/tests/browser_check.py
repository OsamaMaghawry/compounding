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
