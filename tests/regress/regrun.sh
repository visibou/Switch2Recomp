#!/bin/bash
# SwitchRecomp - Copyright 2026 MZ Labs
# Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
# runs every test program with one runtime and saves its last frame as TAG.ppm in the test's folder
# usage: tests/regress/regrun.sh path/to/runtime_linux_x11 TAG [tests folder, default tests/regress/programs]
# each test gets a fresh Xvfb, a shared one kept dying after a few runs
RT=$(realpath "$1"); TAG=$2; ROOT=${3:-$(dirname "$0")/programs}; disp=60
[ -x "$RT" ] || { echo "no runtime at $1"; exit 1; }
for d in "$ROOT"/*/; do
  n=$(ls "$d"*_code.so 2>/dev/null | head -1); [ -z "$n" ] && continue
  b=$(basename "${n%_code.so}"); disp=$((disp+1))
  Xvfb :$disp -screen 0 1920x1080x24 >/dev/null 2>&1 & xp=$!; sleep 2
  ( cd "$d"; cp "$RT" "$b"; rm -rf shader_cache; DISPLAY=:$disp timeout -s KILL 40 ./"$b" --seconds=3 --shot="$TAG" >"$TAG.out" 2>&1 )
  kill $xp 2>/dev/null; wait $xp 2>/dev/null
  printf '%-24s %s\n' "$(basename "$d")" "$(grep -o 'frames=.*' "$d$TAG.out" | tail -1)"
done
