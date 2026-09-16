# Running it online

The CLI ties you to one computer. `reelforge serve` gives you the same pipeline as a
web app: upload takes from a phone, watch progress, review, export, download.

```bash
pip install -e ".[web]"
reelforge serve
```

It prints an address and a password. That is the whole thing locally. The rest of this
page is about putting it somewhere you can reach from anywhere.

## The Look panel

Under the preview there is a **Look** card, in three tabs.

| Tab | What it changes |
|---|---|
| **Captions** | style, font, words per line, size, height on screen, pause between lines, colours |
| **Motion** | punch-ins per minute and strength; the transition at a cut, its length and strength |
| **Pacing** | whether dead air is cut, the shortest silence worth cutting, the breath left behind |

**Caption styles.** *Karaoke* colours the word you are saying. *Box* puts it in a filled
box - the CapCut look. *Pop* pulses the whole line on each word. *One word* shows a
single large word at a time; set **words per line** to 1 with it. *Plain* marks nothing.

**Apply and re-render** re-decides the edit and renders a new preview. It does not
listen to your voice again - the transcript is already taken and cached against the
clip - so trying four caption styles costs four preview renders, not four
transcriptions. **Back to the template** undoes everything at once.

Two things to know:

- Wording you typed into a caption is carried over. Fix a name once and it survives
  every look you try afterwards.
- The zoom and transition tick boxes reset, because a different rate gives you
  different moves - there is nothing for the old ticks to attach to. Choose the look
  first, then untick the individual moves you do not want, then export.

Picking a font that has not been downloaded fetches it during the run; the log line
says which. Each control is a real profile key, so anything the panel does is the same
as `--set captions.style=word` on the command line, and a look you settle on can be
written into a template.

## Editing while it plays

The player is not a rendered file. It plays your untouched footage and draws the
edit over it - skipping what is cut, showing the captions, applying the zooms - so
a change is something you watch happen rather than something you queue.

Press play and start changing things. Caption style, font, colours, position: the
video keeps playing and the captions change under you. Words per line, zoom rate,
transitions, pacing: those are decisions rather than appearance, so the edit is
re-decided on the server - about a second, and still no render.

**Nothing is written down until you press Save.** Try six caption styles and keep
none; the banner tells you when you are looking at something unsaved, and
**Discard** puts it back.

### The shape of the letters

Under **Captions**, beyond size, the letters themselves can be changed.

**Weight** runs from light to black. This is a real weight, not a bold switch:
the font is cut to a genuine face at that weight the first time you pick it, and
the renderer is handed that face. It has to be done that way because the
renderer ignores a variable font's weight axis - asked for light or black it draws
the same bold either way, which was measured before it was believed. Families
that ship one file per weight have that file fetched instead. A weight a family
does not have falls back to the nearest it does, and the log says so.

**Outline** and **shadow** are the dark edge and drop that keep text readable
over footage.

Then three controls change the letters' proportions:
**width** and **height** as percentages, and **space between letters** in pixels.
Width below 100 condenses the font; height above 100 gives the tall, narrow look.
They are the same three settings the renderer has, so the live view and the
export agree.

Arabic letters join, so spacing pulls them apart and breaks the joins - use width
and height for a condensed look, and keep spacing for Latin text.

### The settings, under the video

A row of icons sits directly under the timeline: **Captions**, **Motion**,
**B-roll**, **Pacing**, **Default**. Tap one and its controls open right there;
tap it again and they close. That is the whole panel - no scrolling past the edit
to reach a font and scrolling back to see what it did.

**Default** keeps the look you are using now for every new upload, so a style you
settled on is not something to pick again on each video. Only what actually
differs from the template is remembered, so saving a default does not freeze
every other setting at today's value.

### The timeline

The strip under the video is your footage end to end. Lit areas are what survives;
dark areas are already cut.

- **Tap** it to jump there.
- **Drag the white playhead** to scrub. Playback pauses while you drag and picks
  up again when you let go.
- **Drag across** it to select a stretch. Two handles appear; drag either to
  adjust, and the video scrubs as you drag so you set the point by seeing the
  frame.
- **Start here** / **End here** set an edge from wherever the playhead is.
- **Cut the selection** removes it. **Keep only this** throws away everything else.
- The **1x** button cycles the playback speed, for getting through a long take.
- **Undo all edits** puts everything back.

### Pauses and transitions, by dragging

**Adjust pauses** puts **two** markers on every join between surviving pieces: one
on the end of the piece before it, one on the start of the piece after. Drag either
and only that piece moves. The end of one shot and the head of the next are two
separate decisions, and trimming the tail of a take has no business shortening the
one that follows it.

Dragging an edge outward gives back silence the cut removed; inward takes more
away, eating the padding and then the shot itself. Neither can pass the other -
footage cannot play twice - and neither can shave a piece down to nothing.

Tapping a marker also opens a **Transition** slider for that one join, so a single
cut can land harder than the rest while the others keep the length set in Motion.

Both are stored against your original footage, like trims, so they survive a
restyle, a re-plan and a restart.

### Dragging, on a phone

Everything on the strip is driven by pointer events, so a mouse, a pen and a
finger take the same path through the code. A drag keeps following your finger
after it slides off the strip, or off the screen; a gesture the phone cancels -
deciding it was a scroll, or a call arriving - ends the drag rather than leaving
it stuck. Updates are batched to one per frame and nothing is rebuilt while you
drag, which is what made the first version feel like it was catching: the element
under your finger was being destroyed and remade sixty times a second.

Selections are made against the finished video, which is what you were watching,
and stored as positions in the **original footage**. That is what lets a trim keep
meaning the same thing after you change the caption style, make the pacing more
aggressive, or restart the machine.

A selection rarely lines up with the segments the silence cutter made, so those
are split at the edges of what you selected and only the pieces inside it are
dropped. Splitting changes nothing you can see: two halves play exactly as the
whole did.

What moves with a trim: every zoom, transition and caption after it, because they
are positioned in finished-video time and would otherwise land on the wrong
sentence. What goes: a zoom that lived inside what you cut, since its moment no
longer exists. That is not recorded as you rejecting the zoom - cutting a weak
take says nothing about the zoom that happened to sit in it.

### Fixing a word

The caption list under the player is editable. Type in a box and the word changes
on the video as you type; Save keeps it, Discard puts it back. The zoom and
transition tick boxes work the same way.

What is saved is *what changed*, never a line number: a word that became another
word, a line that now reads differently, a move at a moment in the footage that
was switched off. That is what lets a fix survive the edit being decided again -
a settings change regroups the lines, and a fix pinned to line 4 would land on the
wrong sentence the moment there were five. A word you fix in one line is fixed
wherever else it was misheard, which is also what the editor learns from.

### What the live view is not

It is the browser laying out text; the export is libass. They are close, not
identical, and Arabic shaping is exactly where they can differ. Use the live view
to judge timing, wording, framing and pacing - then **Render a real preview** to
check the Arabic reads correctly before exporting.

## The b-roll library

Your own clips and stills, kept in `<data>/broll` - beside your edits, not inside
the package, which is replaced on every update.

Add them from the main page, then give each one **the words that should bring it
up**. Say one of those words and the clip is cut in over you. A clip with no words
never appears, so tagging is the whole job; with nothing typed it falls back to the
filename, which means naming a file `ارباح.mp4` is already enough.

Keywords are stored as you typed them in `library.json`. The matcher normalises
them when it reads them - that is why what you see in the box is your spelling
rather than the stripped-down form it matches on.

The **B-roll** tab in the Look panel controls how often clips appear, how sure the
word match has to be, and whether a clip covers the screen, sits in a corner, or
runs as a band across the middle.

Each clip has a **how long** setting, and says underneath exactly what it will do -
*the whole clip, 12.3s* or *4s, from 2s in* - so the setting is never a guess.

- **short cutaway** - sized to the sentence that triggered it, a couple of seconds,
  which is what a cutaway wants. Never longer than the clip has.
- **whole clip, to its end** - runs until the clip finishes, however long that is,
  then hands back to you. No limit.
- **a number of seconds** - type it.

**start the clip at** skips into it, for a clip whose first seconds are not the
point. A still has no length of its own, so it gets one.

A clip's length is measured when it is added, and if it was added before that was
done, when the library next loads. This matters: an unmeasured clip on *whole clip*
used to fall back to a one-second cutaway without saying so.

A clip told to play out is used **once** in an edit. It is a deliberate insert
rather than decoration, and repeating it is how a long clip ends up covering most
of a video - which looks like b-roll that will not go away. Cutaways may still
appear more than once. Whatever the settings, b-roll never covers more than about
three fifths of the finished video: the person talking is the video.

Each clip in the list says whether the edit you have open actually says one of its
words - **said in this edit**, **not said in this edit**, or **no words yet**. A
tagged clip that never appears and a broken feature look identical otherwise.

Which clips get cut in is decided when the edit is decided, so adding or tagging
one re-decides the open edit on the spot. Without that, a clip added to an edit
that was already planned would change nothing until something else happened to
re-plan it - which is exactly what it looked like when it was broken.

B-roll also plays in the live preview, laid over you the way the render will do
it, rather than only appearing after an export. When its moment is over it leaves
the screen - that sounds too obvious to write down, but for a while it did not:
a stylesheet rule for the main video outranked the browser's own way of hiding
things, so the clip was told to go and stayed. The checks now ask the screen
whether it is gone, not the element whether it was told to. What the preview streams is a
small copy of each clip, made once when you add it: the original is whatever came
off a phone, and this goes down the wire every time the clip comes up on screen.
A clip the browser cannot decode falls back to its still picture, so you still see
that b-roll happens there.

## Watching it work

A job shows a bar, a percentage, how long it has been running, and when it last
did something. That last one is the useful one: "working" on its own cannot tell
a job that is slow from a job that has died, and both look identical while you
wait.

How long quiet is normal depends on the step, and the machine says which it is
on. Whisper without a graphics card goes quiet for **minutes** between chunks -
that is the model thinking, not a hang - so transcribing is given fifteen minutes
of silence before anything is said about it, and a render three. Judging both by
the same stopwatch is how a working job gets called dead.

While it is quiet but within that, it says **thinking — quiet for 2m, which is
normal here** rather than crying wolf.

## Choosing a model, and what it costs

This matters more than anything else for how long a run takes:

| Model | Arabic | On a cloud machine with no graphics card |
|---|---|---|
| **small** | good | a few minutes |
| medium | better | roughly 4-8 minutes per minute of talking |
| large-v3 | best | roughly 10-20 minutes per minute of talking |

`large-v3` is not broken when it takes twenty minutes on a minute of speech - that
is what it costs without a GPU. Start with **small**; move up only when its Arabic
is not good enough for a particular video. The page says this next to the choice
now, rather than leaving you to discover it by watching a bar.

**Stop this** gives up on a job and kills whatever ffmpeg it had running, so a
job that is going nowhere can be abandoned and tried again instead of waited out.

If the page cannot reach the machine it says so and keeps trying, backing off as
it goes. It used to give up silently on the first failed request and sit there
showing whatever it last saw - usually the word "working" - for a job that had
long since finished or died.

## Does closing the page stop the work?

No. The work runs on the machine, off a queue. The page only asks how it is
going. Refresh it, close the tab, shut the laptop, run out of battery - the job
carries on and is where you left it when you come back.

The one exception is **uploading**: those bytes come from the browser, so closing
the tab mid-upload does stop that. Once the clips are up and the edit has started,
the browser has no part in it.

## Leaving and coming back

Everything saves itself as you go, to whatever `--data` points at - `/workspaces/data`
in a Codespace, which sits outside the repository and survives the machine stopping.
There is no save button and nothing to remember.

A Codespace stops after thirty idle minutes. When you open it again, the edit is
listed, the preview plays, the caption and effect lists are there, and Export works -
the edit is read back from disk the moment anything asks for it. What you changed
before you left is what comes back, not what the editor originally proposed.

If the machine stopped *during* a render, that edit is marked failed, because it was.
Your clips are still on it, so it shows a **Try again** button rather than asking you
to upload them a second time.

Two things do not survive, by design:

- The transcript cache and the speech model are re-used, not re-downloaded, so a
  restart costs nothing there - but a **deleted** edit is gone, including its clips.
- Nothing is recoverable if you delete the Codespace itself. `/workspaces/data` lives
  on that machine. Download anything you care about, or copy the folder out.

## The quickest route: GitHub Codespaces

If you have a GitHub account, you already have a server. A Codespace is a Linux
machine in the cloud with a browser terminal and an HTTPS address that only you can
open. No card, no new account, no Docker.

1. Open the repository on GitHub
2. **Code** → **Codespaces** → **Create codespace on
   `claude/ai-video-editor-reels-52docy`**
3. Wait a few minutes the first time - ffmpeg, the app and the fonts install themselves
4. Open the **PORTS** tab at the bottom, click the globe icon next to port 8000
5. Log in with the password printed in the terminal

That address is a normal link. Bookmark it; it stays the same for that Codespace, so
from any computer you sign in to GitHub, start the Codespace and open the bookmark.
Forwarded ports are **private to your account** by default - nobody else can reach it.

## Getting updates

The page checks on every load and says so at the top when a newer version is
ready: **A newer version is ready — N changes since this one**, with an **Update
now** button. That is the whole thing. Press it, wait a few seconds, and the page
reloads into the new version.

It also updates on its own whenever the machine starts - and if new code arrives
while the app is already running, it restarts into it rather than pulling code
nobody loads.

One catch worth knowing, because it bites exactly once: a copy that predates this
cannot update itself into it. There was no Update button in it to press. If the
version on the page is older than 0.5.1, it needs one terminal visit:

```bash
cd /workspaces/compounding && git pull && bash .devcontainer/restart.sh
```

After that the button exists and the terminal is done with.

Codespaces sleep after thirty idle minutes, and a sleeping machine cannot serve a
link. So the address is not a thing you can hand to someone and expect to answer -
opening the Codespace is what wakes it. Two bookmarks make that about as short as
it gets:

1. `https://github.com/codespaces` - click yours to wake it
2. `https://<your-codespace>-8000.app.github.dev` - the app itself

For an address that answers without waking anything, the machine has to stay on,
which Codespaces deliberately do not. That means a small always-on server of your
own with a Cloudflare Tunnel in front - the app runs the same way there, but it is
a machine you rent and pay for by the month. Worth it only once you are editing
most days.

To see the password again:

```bash
cat /workspaces/.reelforge-keys
```

Worth knowing:

- Codespaces **stop after 30 minutes idle** and your files are kept. Restarting is quick.
- The free allowance is generous for occasional editing but it is measured in
  core-hours, so a 4-core machine uses it twice as fast as a 2-core one. Stop the
  Codespace when you are finished rather than leaving it running.
- Uploads, exports and the speech model live in `/workspaces`, outside the repository,
  so nothing you record is ever committed.
- For a permanent address of your own, run a **Cloudflare Tunnel** inside the Codespace
  pointing at `localhost:8000`. Codespaces' own link is enough to start with.

## Read this before you pay for anything

**Record at 1080p, not 4K.** Takes shot on one phone in one mode are now stuck
end to end without being re-encoded at all, which takes about a second instead of
minutes - but only when they match. Change camera mode between takes, or shoot 4K
and force everything to be scaled down, and that shortcut is gone.
 The output is 1080x1920 whatever you feed it, so 4K buys
you nothing and costs you four times the pixels:

| | 4K | 1080p |
|---|---|---|
| A 25-second clip | ~150 MB | ~37 MB |
| Upload on hotel wifi | painful | fine |
| Joining and encoding | minutes | seconds |

On an iPhone: Settings → Camera → Record Video → **1080p HD at 30 fps**. This single
change matters more than which server you rent.

**Consider not renting anything.** If your laptop is usually on and with you, running
`reelforge serve` on it and reaching it through a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
or [Tailscale](https://tailscale.com) costs nothing, needs no upload, and is faster than
any cheap VPS - your laptop has more cores. Rent a server when you want it working while
your laptop is shut.

## What a server needs

| | Why |
|---|---|
| **4+ CPU cores** | Joining and rendering are CPU-bound. Fewer cores is proportionally slower. |
| **8 GB RAM** | `small` fits comfortably; `large-v3` wants more. |
| **40 GB+ disk** | Uploads, previews and exports add up. Delete old jobs. |
| **ffmpeg with libass** | Already in the Docker image. |

A GPU only speeds up transcription, not rendering. It is worth it if you use
`large-v3` constantly; otherwise `small` on CPU is the better trade.

Rough expectation for a 60-second 1080p video on 4 CPU cores: **a couple of minutes**
end to end, most of it rendering. The same on 2 cores is roughly double. Check current
pricing yourself - it moves - but a 4-core box is usually in the tens of dollars a month.

## Deploying with Docker

```bash
git clone <this repo> && cd reelforge
# edit docker-compose.yml and change BOTH passwords
docker compose up -d --build
```

Then open `http://your-server:8000`. Two volumes are mounted:

- `./data` - uploads, previews, exports, and everything the tool has learned from you
- `./models` - the speech model, downloaded once and kept

Both survive rebuilds. Back up `data` and you have your whole studio.

## Put HTTPS in front of it

The password protects the app, but over plain HTTP that password crosses the network in
the clear. Do not skip this.

The easiest option, with no open ports and no certificate to manage, is a **Cloudflare
Tunnel**: install `cloudflared` on the server, point it at `localhost:8000`, and you get
an HTTPS address. A reverse proxy such as Caddy with a real domain works equally well.

Other things worth doing:

- Set `REELFORGE_PASSWORD` to something long. A generated one is printed at startup if
  you do not, and it changes on every restart.
- Set `REELFORGE_SECRET` to any long random string, so logins survive a restart.
- Do not expose port 8000 directly to the internet.

## What it does and does not do

One job runs at a time, on purpose - two renders on the same cores are slower than one
after the other. Uploading while a job runs is fine; it queues.

Jobs are kept on disk, so a restart does not lose your uploads. A job that was mid-render
when the process died is marked interrupted rather than left looking healthy.

There is **one password and one user**. This is a tool for you, not a service for a team.
If several people need it, run separate instances rather than sharing a login.

## Environment variables

| | |
|---|---|
| `REELFORGE_PASSWORD` | The login password. Generated and printed if unset. |
| `REELFORGE_SECRET` | Signs session cookies. Set it, or logins drop on restart. |
| `REELFORGE_ASR_BACKEND` | Pin the speech backend (`faster-whisper`, `stub`). |
| `HF_HOME` | Where the speech model is cached. |
