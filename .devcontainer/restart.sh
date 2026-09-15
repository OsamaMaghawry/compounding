#!/usr/bin/env bash
# Restart ReelForge after a `git pull`.
#
# Stops by recorded PID rather than by matching the command line: `pkill -f
# "reelforge serve"` also matches any shell whose own command line contains that
# text, which kills the terminal you typed it in. Stopping and starting as two
# separate commands is also unreliable - the process takes a moment to exit, so
# the start sees it still alive and quietly does nothing.
set -uo pipefail

PIDFILE=/workspaces/.reelforge.pid
here="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping ReelForge (pid $pid)…"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "it did not stop politely - forcing it"
      kill -9 "$pid" 2>/dev/null || true
      sleep 1
    fi
  fi
  rm -f "$PIDFILE"
fi

exec bash "$here/start.sh"
