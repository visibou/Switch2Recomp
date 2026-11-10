#!/bin/sh
# nxnative's shader_split.py on a game's RomFS and main, into games/<game>/<game>_nvn/split (first step of tools/nxshader)
# DM_GAME names nxnative's profile: its source groups and table notes. Local Python modules: pip install --target external/pylibs backports.zstd lz4 capstone
cd "$(dirname "$0")/../.."
. tools/env.sh
DUMP="${1:?usage: tools/nxshader/split_mkw.sh <your dump>/base}"
G="$(pwd -W)/games/MKW"
EXTW="$(cd "$EXTERNAL" && pwd -W)"
OUT="$G/MKW_nvn/split"
mkdir -p "$OUT/pkg"
cd "$EXTERNAL/nxnative/dm/gfx/tools"
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 DM_GAME=0400C3F00006E000 J=${J:-8} PYTHONPATH="$EXTW/pylibs" python "$G/../../tools/nxshader/run_split.py" "$DUMP/romfs" "$G/MKW_data/exefs/main" \
  "$OUT/pkg" "$OUT/shader_map.tsv" "$OUT/shader_facts.tsv" "$OUT/shader_pairs.tsv" "$OUT/pso_pairs.tsv" > "$OUT/split.log" 2>&1
echo "split exit $?" >> "$OUT/split.log"
