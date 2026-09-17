#!/usr/bin/env bash
# Keep ReelForge running.
#
# The whole point of this setup is that there is no terminal - there is a link,
# and the link either works or it does not. So a crash must not be the end of it:
# nobody is watching the machine, and a browser cannot restart a process. If the
# app stops for any reason - a bug, or the system killing it for using too much
# memory - it is started again.
set -uo pipefail

STATE="${REELFORGE_STATE:-/workspaces}"
LOG="$STATE/reelforge.log"
PORT="${REELFORGE_PORT:-8000}"
DATA="${REELFORGE_DATA:-$STATE/data}"

child=""
stop() {
  [ -n "$child" ] && kill "$child" 2>/dev/null
  wait "$child" 2>/dev/null
  exit 0
}
trap stop TERM INT

# A crash loop is worse than being down: it burns the machine and fills the disk
# with the same error. Back off, and after enough failures in a row stop and
# leave the reason at the end of the log.
fails=0
while true; do
  reelforge serve --host 0.0.0.0 --port "$PORT" --data "$DATA" >>"$LOG" 2>&1 &
  child=$!
  started=$(date +%s)
  wait "$child"
  code=$?
  child=""
  lived=$(( $(date +%s) - started ))

  # Up for a while then gone is a crash worth recovering from, every time.
  # Dying immediately, over and over, is a broken version, not bad luck.
  if [ "$lived" -ge 60 ]; then
    fails=0
  else
    fails=$(( fails + 1 ))
  fi
  if [ "$fails" -ge 5 ]; then
    echo "[$(date -Is)] ReelForge has failed to start 5 times in a row (exit $code)." >>"$LOG"
    echo "[$(date -Is)] Giving up rather than looping. The reason is above." >>"$LOG"
    exit 1
  fi
  # Wait longer each time. The first failure is often the last one still letting
  # go of the port, which is worth a moment rather than five instant retries and
  # a conclusion that the version is broken.
  pause=$(( 3 * (1 << (fails - 1)) ))
  echo "[$(date -Is)] ReelForge stopped after ${lived}s (exit $code) - trying again in ${pause}s" >>"$LOG"
  sleep "$pause"
done
