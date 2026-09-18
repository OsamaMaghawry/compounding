#!/usr/bin/env bash
# Runs once when the Codespace is created. Everything ReelForge needs, installed
# in the cloud machine rather than on whatever laptop you happen to be using.
# Not -e: one step failing must not take the rest with it. Installing ffmpeg
# needs a package mirror, and when that had a bad morning the install of
# ReelForge itself never ran - so the machine came up with nothing to serve and
# the link went nowhere, which is a lot of consequence for a slow mirror.
set -uo pipefail

echo "==> installing ffmpeg"
sudo apt-get update -qq || sudo apt-get update -qq || true
sudo apt-get install -y -qq --no-install-recommends ffmpeg \
  || sudo apt-get install -y -qq --no-install-recommends ffmpeg \
  || echo "!! ffmpeg would not install. Editing needs it; starting will try again."

cd "$(dirname "$0")/../reelforge"

echo "==> installing ReelForge (web app + speech recognition)"
python -m pip install --upgrade pip --quiet || true
python -m pip install -e ".[web,asr]" --quiet \
  || python -m pip install -e ".[web,asr]" \
  || echo "!! ReelForge would not install. Starting will try again."

echo "==> downloading Arabic caption fonts"
python -m reelforge setup || true

# Uploads, exports and the speech model live outside the repo so they are not
# committed by accident and survive a rebuild of the container.
mkdir -p /workspaces/data /workspaces/models

echo
python -m reelforge doctor || true
echo
echo "=================================================================="
echo " Ready. It starts itself - the link and the password are printed when"
echo " the machine starts, and again in the PORTS tab next to port 8000."
echo "=================================================================="
