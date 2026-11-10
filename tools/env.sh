# Sourced by the tools' shell scripts: ROOT (the repository), GCC on the PATH, PY (the Python that builds games), EXTERNAL.
# Same search as tools/s2r_paths.py: inside the repository first (gcc/, python/, external/), then beside it, then the PATH.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for d in "$ROOT/gcc/Library/bin" "$ROOT/../gcc/Library/bin"; do
    if [ -x "$d/x86_64-w64-mingw32-gcc.exe" ] || [ -x "$d/x86_64-w64-mingw32-gcc" ]; then export PATH="$(cd "$d" && pwd):$PATH"; break; fi
done
PY=python
for p in "$ROOT/python/python.exe" "$ROOT/../SwitchRecomp/python/python.exe"; do
    if [ -x "$p" ]; then PY="$(cd "$(dirname "$p")" && pwd)/python.exe"; break; fi
done
EXTERNAL="${S2R_EXTERNAL:-}"
if [ -z "$EXTERNAL" ]; then
    if [ -d "$ROOT/external" ]; then EXTERNAL="$ROOT/external"; elif [ -d "$ROOT/../external" ]; then EXTERNAL="$(cd "$ROOT/../external" && pwd)"; else EXTERNAL="$ROOT/external"; fi
fi
export EXTERNAL
