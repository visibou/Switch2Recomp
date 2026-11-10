#!/bin/sh
# Build the Eden GPU backend (runtime/eden) into build/eden with the gcc folder. Needs ../external (eden-ns2, deps,
# tools: see runtime/eden/README.md). Afterwards tools/build_rt.sh links it into the runtime (SWR_GPU=eden).
set -e
cd "$(dirname "$0")/.."
. tools/env.sh   # GCC on the PATH, $PY, $EXTERNAL
EXT="$EXTERNAL"
FLAVOR="${EDEN_FLAVOR:-eden-ns2}"   # eden-ns2 (default) or eglantine (experimental)
B="build/eden_$FLAVOR"
if [ ! -f $B/build.ninja ]; then
  cmake -S runtime/eden -B $B -G Ninja -DCMAKE_MAKE_PROGRAM="$EXT/tools/ninja.exe" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc -DEXT="$EXT" -DEDEN_FLAVOR="$FLAVOR" > tools/.build_eden.log 2>&1 \
    || { tail -20 tools/.build_eden.log; exit 1; }
fi
"$EXT/tools/ninja.exe" -C $B swr_eden video_core shader_recompiler gpu_logging eden_common eden_ext sirit_lib \
  >> tools/.build_eden.log 2>&1 || { grep -m5 -A10 "FAILED\|error:" tools/.build_eden.log; exit 1; }
echo "eden backend built ($FLAVOR, $B)"
