#!/bin/sh
# Recompile MKW's code with GCC (never TCC) into games/MKW, reusing the runtime that
# tools/build_rt.sh built. Only chunks whose C changed are compiled again (MKW_build keeps the objects);
# a first build takes ~33 min. switchrecomp.py's -o is a name (anything but letters, digits, '-', '_' and spaces
# is dropped), the folder is --outdir.
set -e
cd "$(dirname "$0")/.."
. tools/env.sh   # GCC on the PATH, $PY, $EXTERNAL
ROOT="$(pwd -W)"
GAME="${1:?usage: tools/build_game.sh <your dump>/base}"
taskkill //IM MKW.exe //F > /dev/null 2>&1 || true
"$PY" switchrecomp.py "$GAME" -o MKW --outdir "$ROOT/games/MKW" \
  --cc x86_64-w64-mingw32-gcc -O 2 --jobs 12 --runtime "$ROOT/bin/runtime_win64_win32.exe" 2>&1 |
  grep -v "C generated\|compiled [0-9]*/"
