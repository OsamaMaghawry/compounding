# Running it online

The CLI ties you to one computer. `reelforge serve` gives you the same pipeline as a
web app: upload takes from a phone, watch progress, review, export, download.

```bash
pip install -e ".[web]"
reelforge serve
```

It prints an address and a password. That is the whole thing locally. The rest of this
page is about putting it somewhere you can reach from anywhere.

## Read this before you pay for anything

**Record at 1080p, not 4K.** The output is 1080x1920 whatever you feed it, so 4K buys
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
