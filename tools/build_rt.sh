#!/bin/sh
# rebuild the win32 runtime with the gcc folder and put it in the game folder; stops on the first failure
set -e
cd "$(dirname "$0")/.."
. tools/env.sh   # GCC on the PATH, $PY, $EXTERNAL
"$PY" switchrecomp.py --build-runtime --target win64 --backend win32 --cc x86_64-w64-mingw32-gcc > tools/.build_rt.log 2>&1 || { tail -20 tools/.build_rt.log; exit 1; }
taskkill //IM MKW.exe //F > /dev/null 2>&1 || true
cp bin/runtime_win64_win32.exe games/MKW/MKW.exe
echo "runtime built and installed"
