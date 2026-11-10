#!/bin/sh
# Builds the no-Eden runtime and puts it in games/MKWTest only, never in the installed game. Refuses while a game runs; prints "differ" if the copy did not land.
#   sh tools/build_test.sh [--no-copy]
set -e
cd "$(dirname "$0")/.."
. tools/env.sh   # GCC on the PATH, $PY, $EXTERNAL
export EDEN_FLAVOR=none
"$PY" tools/build_nvn.py || python tools/build_nvn.py
"$PY" switchrecomp.py --build-runtime --target win64 --backend win32 --cc x86_64-w64-mingw32-gcc > tools/.build_rt.log 2>&1 || { tail -20 tools/.build_rt.log; exit 1; }
if [ "$1" = "--no-copy" ]; then echo "built, not copied"; exit 0; fi
if tasklist | grep -qi MKW; then echo "a game is running: not copied"; exit 1; fi
cp bin/runtime_win64_win32.exe games/MKWTest/MKW.exe
cmp bin/runtime_win64_win32.exe games/MKWTest/MKW.exe && echo "runtime in games/MKWTest" || { echo differ; exit 1; }
