#!/usr/bin/env bash
# Runs every time the Codespace starts, so the app is already up and the
# forwarded port is a link you can just open - no commands to type.
#
# The one rule here: never leave the machine without a working link. There is no
# terminal in this setup, so a link that goes nowhere cannot be fixed from the
# browser, and nobody is watching the machine to notice. Everything below is in
# service of that - the update is checked before it is kept, and the app is
# watched after it starts.
set -uo pipefail

STATE="${REELFORGE_STATE:-/workspaces}"
PIDFILE="$STATE/.reelforge.pid"
LOG="$STATE/reelforge.log"
PORT="${REELFORGE_PORT:-8000}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo/reelforge"
mkdir -p "$STATE/data" "$STATE/models" 2>/dev/null || true

export REELFORGE_STATE REELFORGE_PORT="$PORT"

answering() {
  curl -fsS -m 3 -o /dev/null "http://127.0.0.1:$PORT/api/health" 2>/dev/null
}

wait_until_answering() {          # long enough to cover the retries below
  for _ in $(seq 1 "${REELFORGE_WAIT:-100}"); do
    answering && return 0
    sleep 1
  done
  return 1
}

install_deps() {
  # Three attempts, each one less than the last. The speech library is the big,
  # fragile one; if it will not install, an app that runs and says so beats no
  # app at all, because there is no terminal here to find out from.
  pip install -e ".[web,asr]" --quiet 2>/dev/null && return 0
  echo "the speech library would not install - installing the rest…"
  pip install -e ".[web]" --quiet 2>/dev/null && return 0
  pip install -e . --quiet --no-deps 2>/dev/null || true
}

# The one-time setup that runs when the Codespace is created can fail - an apt
# mirror having a bad morning is enough - and it used to stop at the first error,
# which meant ffmpeg failing took the install of ReelForge itself down with it.
# The machine then came up with nothing to run and the link went nowhere. So
# check here too: this runs on every start, and the check costs nothing.
ensure_installed() {
  # Asked from elsewhere on purpose. This script works inside the project
  # folder, and from there Python finds the package sitting on disk whether it
  # was ever installed or not - so asking here would always say yes.
  if command -v reelforge >/dev/null 2>&1 \
     && (cd / && python -c "import reelforge" 2>/dev/null); then
    return 0
  fi
  echo "ReelForge is not installed on this machine yet - installing it…"
  install_deps
  command -v reelforge >/dev/null 2>&1 \
    || echo "it could not be installed; the log will say why."
}

ensure_ffmpeg() {
  command -v ffmpeg >/dev/null 2>&1 && return 0
  echo "ffmpeg is missing - installing it…"
  sudo apt-get update -qq 2>/dev/null || true
  sudo apt-get install -y -qq --no-install-recommends ffmpeg 2>/dev/null || true
  command -v ffmpeg >/dev/null 2>&1 \
    || echo "ffmpeg is still missing - the app will run, but edits will fail."
}

# Pick up new code every time the machine starts, so the usual way to get an
# update is simply to open it - no terminal, nothing to type, nothing to
# remember. --ff-only so this can never invent a merge on its own; if the pull
# cannot fast-forward, the old version keeps running and says so.
moved=""
was=""
if [ -z "${REELFORGE_NO_UPDATE:-}" ] && git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
  echo "checking for a newer version…"
  was="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
  if git -C "$repo" pull --ff-only --quiet 2>/dev/null; then
    now="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
    [ "$was" != "$now" ] && moved="yes"
    # With dependencies: an update that brings a new one would otherwise leave
    # the code expecting a library the machine never got, failing quietly where
    # that library was needed. pip is quick when nothing has changed.
    install_deps
  else
    echo "could not update automatically - carrying on with what is here."
  fi
fi

running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

if running && answering; then
  if [ -n "$moved" ]; then
    # New code arrived and the old one is still serving it. Pulling without
    # restarting is the same as not pulling at all, and telling someone to go
    # and type a restart command is the thing this is here to avoid.
    echo "a newer version arrived - restarting into it…"
    REELFORGE_NO_UPDATE=1 REELFORGE_CAME_FROM="$was" \
      exec bash "$repo/.devcontainer/restart.sh"
  fi
  echo "ReelForge is already running (pid $(cat "$PIDFILE"))."
  echo "It is up to date. To restart anyway:  bash .devcontainer/restart.sh"
  exit 0
fi

# A process that is up but not answering is worse than a dead one: it holds the
# port and the pid file, so nothing else starts. Clear it out.
if running; then
  echo "something is running but not answering - replacing it…"
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 2
  kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
fi
rm -f "$PIDFILE"

# A password that survives restarts. Set REELFORGE_PASSWORD as a Codespaces
# secret to choose your own; otherwise one is generated once and kept.
KEYS="$STATE/.reelforge-keys"
if [ ! -f "$KEYS" ]; then
  {
    echo "REELFORGE_PASSWORD=$(python -c 'import secrets;print(secrets.token_urlsafe(9))')"
    echo "REELFORGE_SECRET=$(python -c 'import secrets;print(secrets.token_hex(16))')"
  } > "$KEYS"
  chmod 600 "$KEYS"
fi
# shellcheck disable=SC1090
set -a; . "$KEYS"; set +a

ensure_installed
ensure_ffmpeg

start_supervisor() {
  nohup bash "$repo/.devcontainer/serve.sh" >>"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
}

start_supervisor

if ! wait_until_answering; then
  # It did not come up. If this start was the first one on new code, that code
  # is the obvious suspect - so go back to the version that was working and
  # start that instead. A machine with no terminal cannot be left on a version
  # that will not run.
  previous="${REELFORGE_CAME_FROM:-$was}"
  if [ -n "$previous" ] && [ -n "${moved:-}${REELFORGE_CAME_FROM:-}" ]; then
    echo "the new version will not start - going back to the one that worked…"
    {
      echo "[$(date -Is)] the new version did not answer; rolling back to $previous"
    } >>"$LOG"
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    sleep 1
    git -C "$repo" reset --hard --quiet "$previous" 2>/dev/null || true
    install_deps
    start_supervisor
    wait_until_answering || true
  fi
fi

# Codespaces publishes the forwarded-port hostname, so the exact link can be
# printed rather than described. Terminals make it clickable.
URL=""
if [ -n "${CODESPACE_NAME:-}" ] && [ -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]; then
  URL="https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
fi

echo
echo "=================================================================="
if answering; then
  echo "  ReelForge is running."
else
  echo "  ReelForge did not start. The last of the log:"
  echo
  tail -n 15 "$LOG" 2>/dev/null | sed 's/^/    /'
fi
echo
if [ -n "$URL" ]; then
  echo "  Open this in your browser (ctrl/cmd-click it):"
  echo
  echo "     $URL"
else
  echo "  Open the PORTS tab below, then click the globe next to port $PORT."
fi
echo
echo "  Password: ${REELFORGE_PASSWORD}"
echo "=================================================================="
echo
