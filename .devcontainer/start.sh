#!/usr/bin/env bash
# Runs every time the Codespace starts, so the app is already up and the
# forwarded port is a link you can just open - no commands to type.
set -euo pipefail

cd "$(dirname "$0")/../reelforge"
mkdir -p /workspaces/data /workspaces/models

if pgrep -f "reelforge serve" >/dev/null 2>&1; then
  echo "ReelForge is already running."
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

sleep 2
echo
echo "=================================================================="
echo "  ReelForge is running."
echo
echo "  1. Open the PORTS tab at the bottom of this window"
echo "  2. Click the globe icon next to port 8000"
echo
echo "  Password: ${REELFORGE_PASSWORD}"
echo "=================================================================="
echo
