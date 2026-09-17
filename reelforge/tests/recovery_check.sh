#!/usr/bin/env bash
# Does the link survive the three ways it has gone away?
#
# Not part of the test suite: it clones the repository, binds a port and takes a
# minute. Run it by hand after changing anything under .devcontainer:
#
#     bash reelforge/tests/recovery_check.sh
#
# There is no terminal in the Codespace setup - there is a link, and the link
# either works or it does not. So all three of these must fix themselves:
#
#   1. the machine starts      - the app comes up on its own
#   2. the app is killed       - something starts it again
#   3. an update will not run  - the version that did run comes back
#   4. it was never installed  - starting installs it
#
# The third is the dangerous one. An update that cannot start leaves a dead link
# on a machine with no way to type a command, and nobody watching to notice.
set -uo pipefail
work="$(mktemp -d)"
export REELFORGE_STATE="$work/state"
export REELFORGE_PORT=8811
mkdir -p "$REELFORGE_STATE"

here="$(cd "$(dirname "$0")/../.." && pwd)"
git clone --quiet "$here" "$work/repo"
cd "$work/repo"
cp "$here/.devcontainer/"*.sh "$work/repo/.devcontainer/"
cp "$here/reelforge/reelforge/web.py" "$work/repo/reelforge/reelforge/web.py"
chmod +x "$work/repo/.devcontainer/"*.sh
git -c user.email=t@t -c user.name=t commit --quiet -am "the scripts under test"

# The clone's source ahead of anything installed, so the code under test is the
# code that runs - otherwise a "broken version" is broken only on paper.
export PYTHONPATH="$work/repo/reelforge"
export PIP_NO_INDEX=1                       # keep the test off the network
export REELFORGE_NO_UPDATE=1

say(){ printf '\n>>> %s\n' "$1"; }
ask(){ curl -fsS -m 3 "http://127.0.0.1:$REELFORGE_PORT/api/health" 2>/dev/null; }
cleanup(){ kill "$(cat "$REELFORGE_STATE/.reelforge.pid" 2>/dev/null)" 2>/dev/null; sleep 1; rm -rf "$work"; }
trap cleanup EXIT

python -c "import reelforge; print('running from:', reelforge.__file__)"

say "1. a cold start"
bash "$work/repo/.devcontainer/start.sh" >"$work/start1.log" 2>&1
echo "health: $(ask)"
[ -n "$(ask)" ] || { echo "FAIL: did not come up"; tail -20 "$work/start1.log"; exit 1; }
echo "PASS"

say "2. the app is killed, as the system would kill it for using too much memory"
sup="$(cat "$REELFORGE_STATE/.reelforge.pid")"
child="$(pgrep -P "$sup" | head -1)"
kill -9 "$child"
sleep 1
echo "straight after the kill: '$(ask)'"
for _ in $(seq 1 30); do [ -n "$(ask)" ] && break; sleep 1; done
[ -n "$(ask)" ] && echo "PASS: back up on its own" || { echo "FAIL"; tail -20 "$REELFORGE_STATE/reelforge.log"; exit 1; }

say "3. an update to a version that cannot start"
good="$(git rev-parse HEAD)"
printf '\nthis is not python(\n' >> reelforge/reelforge/__init__.py
git -c user.email=t@t -c user.name=t commit --quiet -am "a broken version"
broken="$(git rev-parse HEAD)"
echo "good ${good:0:8} -> broken ${broken:0:8}"
if python -c "import reelforge" 2>/dev/null; then
  echo "FAIL: the broken code still imports, so this proves nothing"; exit 1
fi
echo "confirmed: the broken version will not even import"
REELFORGE_CAME_FROM="$good" bash "$work/repo/.devcontainer/restart.sh" >"$work/start2.log" 2>&1
now="$(git rev-parse HEAD)"
echo "health: $(ask)"
echo "now on: ${now:0:8}"
if [ "$now" = "$good" ] && [ -n "$(ask)" ]; then
  echo "PASS: went back to the version that worked, and is serving"
else
  echo "FAIL: left on ${now:0:8}, health '$(ask)'"
  tail -20 "$work/start2.log"; tail -10 "$REELFORGE_STATE/reelforge.log"
fi
echo "--- what the log says about the rollback ---"
grep -i "rolling back\|stopped after\|failed to start" "$REELFORGE_STATE/reelforge.log" | tail -4

say "4. a machine where the one-time setup fell over, so nothing was installed"
kill "$(cat "$REELFORGE_STATE/.reelforge.pid" 2>/dev/null)" 2>/dev/null
rm -f "$REELFORGE_STATE/.reelforge.pid"
sleep 1
# A python of its own, and a PATH without this machine's own copy on it.
python -m venv "$work/venv" >/dev/null
export PATH="$work/venv/bin:/usr/bin:/bin"
unset PYTHONPATH
unset PIP_NO_INDEX          # this one genuinely has to reach the network
echo "reelforge before: $(command -v reelforge || echo 'not installed')"
if command -v reelforge >/dev/null 2>&1; then
  echo "SKIP: a copy is still on the PATH, so this would prove nothing"
else
  bash "$work/repo/.devcontainer/start.sh" >"$work/start3.log" 2>&1
  echo "reelforge after : $(command -v reelforge || echo 'not installed')"
  echo "health: $(ask)"
  [ -n "$(ask)" ] && echo "PASS: it installed itself and came up" \
                  || { echo "FAIL"; tail -20 "$work/start3.log"; }
fi
