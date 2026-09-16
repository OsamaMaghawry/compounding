#!/usr/bin/env bash
# Runs every time the Codespace starts, so the app is already up and the
# forwarded port is a link you can just open - no commands to type.
set -euo pipefail

PIDFILE=/workspaces/.reelforge.pid
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo/reelforge"
mkdir -p /workspaces/data /workspaces/models

# Pick up new code every time the machine starts, so the usual way to get an
# update is simply to open it - no terminal, nothing to type, nothing to
# remember. --ff-only so this can never invent a merge on its own; if the pull
# cannot fast-forward, the old version keeps running and says so.
moved=""
if [ -z "${REELFORGE_NO_UPDATE:-}" ] && git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
  echo "checking for a newer version…"
  was="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
  if git -C "$repo" pull --ff-only --quiet 2>/dev/null; then
    now="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
    [ "$was" != "$now" ] && moved="yes"
    # With dependencies: an update that brings a new one would otherwise leave
    # the code expecting a library the machine never got, failing quietly where
    # that library was needed. pip is quick when nothing has changed.
    pip install -e ".[web,asr]" --quiet 2>/dev/null \
      || pip install -e . --quiet --no-deps 2>/dev/null || true
  else
    echo "could not update automatically - carrying on with what is here."
  fi
fi

running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

if running; then
  if [ -n "$moved" ]; then
    # New code arrived and the old one is still serving it. Pulling without
    # restarting is the same as not pulling at all, and telling someone to go
    # and type a restart command is the thing this is here to avoid.
    echo "a newer version arrived - restarting into it…"
    REELFORGE_NO_UPDATE=1 exec bash "$repo/.devcontainer/restart.sh"
  fi
  echo "ReelForge is already running (pid $(cat $PIDFILE))."
  echo "It is up to date. To restart anyway:  bash .devcontainer/restart.sh"
  exit 0
fi

# A password that survives restarts. Set REELFORGE_PASSWORD as a Codespaces
# secret to choose your own; otherwise one is generated once and kept.
KEYS=/workspaces/.reelforge-keys
if [ ! -f "$KEYS" ]; then
  {
    echo "REELFORGE_PASSWORD=$(python -c 'import secrets;print(secrets.token_urlsafe(9))')"
    echo "REELFORGE_SECRET=$(python -c 'import secrets;print(secrets.token_hex(16))')"
  } > "$KEYS"
  chmod 600 "$KEYS"
fi
# shellcheck disable=SC1090
set -a; . "$KEYS"; set +a

nohup reelforge serve --host 0.0.0.0 --port 8000 --data /workspaces/data \
  > /workspaces/reelforge.log 2>&1 &
echo $! > "$PIDFILE"

sleep 2

# Codespaces publishes the forwarded-port hostname, so the exact link can be
# printed rather than described. Terminals make it clickable.
URL=""
if [ -n "${CODESPACE_NAME:-}" ] && [ -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]; then
  URL="https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
fi

echo
echo "=================================================================="
echo "  ReelForge is running."
echo
if [ -n "$URL" ]; then
  echo "  Open this in your browser (ctrl/cmd-click it):"
  echo
  echo "     $URL"
else
  echo "  Open the PORTS tab below, then click the globe next to port 8000."
fi
echo
echo "  Password: ${REELFORGE_PASSWORD}"
echo "=================================================================="
echo
