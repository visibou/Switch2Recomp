#!/bin/sh
# M4: every shader of the game to SPIR-V and reflection, in games/MKW/MKW_nvn/cache (the runtime loads <cache>/vk/<key>.spv + .json).
# First tools/nxshader/split_mkw.sh (nxnative's shader_split.py). Needs .NET 10 (the translator) and external/spirv-tools-build (spirv-val, spirv-opt). Limit: one hour.
cd "$(dirname "$0")/.."
R="$(pwd -W)"; G="$R/games/MKW/MKW_nvn"
PYTHONUTF8=1 PYTHONPATH="$R/tools/nxshader" timeout 3600 python tools/nxshader/nxshader.py --cache "$G/cache" build "$G/split" --jobs 4 --report "$G/shader_report.txt" "$@" > "$G/shader_build.log" 2>&1
echo "shaders exit $?" >> "$G/shader_build.log"
