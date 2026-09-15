#!/usr/bin/env bash
# Runs once when the Codespace is created. Everything ReelForge needs, installed
# in the cloud machine rather than on whatever laptop you happen to be using.
set -euo pipefail

echo "==> installing ffmpeg"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends ffmpeg

cd "$(dirname "$0")/../reelforge"

echo "==> installing ReelForge (web app + speech recognition)"
python -m pip install --upgrade pip --quiet
python -m pip install -e ".[web,asr]" --quiet

echo "==> downloading Arabic caption fonts"
python -m reelforge setup || true

# Uploads, exports and the speech model live outside the repo so they are not
# committed by accident and survive a rebuild of the container.
mkdir -p /workspaces/data /workspaces/models

echo
python -m reelforge doctor || true
echo
echo "=================================================================="
echo " Ready. Start it with:"
echo
echo "   cd reelforge && reelforge serve --data /workspaces/data"
echo
echo " Then open the PORTS tab, find port 8000, and click the globe icon."
echo "=================================================================="
