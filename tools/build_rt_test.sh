#!/bin/sh
# Rebuild the win32 runtime with GCC and put it in a TEST folder only (default games/MKWTest), never the user's install.
# Unlike build_rt.sh it kills nothing: it refuses while a game runs from that folder.
#   sh tools/build_rt_test.sh [games/MKWTest]        SWR_KEEP_SYMBOLS=1 keeps symbols for tools/host_prof_syms.py
set -e
cd "$(dirname "$0")/.."
DEST="${1:-games/MKWTest}"
case "$DEST" in *games/MKW|*games/MKW/) echo "refusing: $DEST is the user's install (use build_rt.sh, with the user's OK)"; exit 1;; esac
. tools/env.sh   # GCC on the PATH, $PY, $EXTERNAL
python tools/build_nvn.py > tools/.build_nvn.log 2>&1 || { tail -30 tools/.build_nvn.log; exit 1; }   # the NVN layer (runtime/nvn) is a library of its own
"$PY" switchrecomp.py --build-runtime --target win64 --backend win32 --cc x86_64-w64-mingw32-gcc > tools/.build_rt.log 2>&1 || { tail -20 tools/.build_rt.log; exit 1; }
if ! cp bin/runtime_win64_win32.exe "$DEST/MKW.exe"; then echo "copy failed (a test game still running from $DEST?)"; exit 1; fi
echo "runtime built into $DEST"
