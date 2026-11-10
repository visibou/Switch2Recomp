#!/bin/sh
# Scripted MKW run to a race (Grand Prix, Mario Bros. Circuit) on the Eden backend; prints the frame
# rate from second 215 on (the race, if the menus went through: check the screenshots) and the host GPU's load. Usage: tools/race_bench.sh <name> [runtime option key=value ...]
#   e.g. tools/race_bench.sh half resolution=0.5 docked=0
# Environment variables (SWR_*, EDEN_*) pass through. Screenshots at 225 s and 258 s land in the run folder.
cd "$(dirname "$0")/.."
NAME=$1; shift
EXTRA=""; for kv in "$@"; do EXTRA="$EXTRA --extra $kv"; done
G=games/MKW
rm -rf $G/MKW_eden/log
# A every 4 s from 60 s to the end: menus advance whenever loading allows, in the race it taps accelerate
. tools/env.sh
S=$("$PY" -c "print(','.join('%d:1,%d:0'%(t,t+300) for t in range(60000,256001,4000)))")
GPU=$(mktemp)
( sleep 225; timeout 30 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > $GPU 2>/dev/null ) &
SWR_GPU=${SWR_GPU:-eden} "$PY" tools/run_game.py --seconds 260 --shots-at 225000,258000 \
  --script "$S" --name "bench_$NAME" $EXTRA > /dev/null 2>&1
wait
R=$(ls -td $G/runs/*bench_$NAME | head -1)
L=$G/MKW_eden/log/eden_log.txt
echo "== $NAME ($R): $(grep -m1 '^second' $R/threads.txt), $(grep -c 'without a new image' $R/game.log) stalls"
grep "Switch2Recomp: [0-9]* submissions" $L | sed -E 's/^\[ *([0-9.]+)\].*: ([0-9]+) submissions \(([0-9.]+) ms each, write watch ([0-9.]+) ms.* ([0-9]+) frames composited/\1 \2 \4 \5/' |
  awk 'NR>1 && $1>215 {n++; f+=($4-pf)/($1-pt); if(min==""||($4-pf)/($1-pt)<min) min=($4-pf)/($1-pt)} {pt=$1;pf=$4}
       END {if(n) printf "   from 215 s: %.1f FPS average, %.1f lowest of %d samples\n", f/n, min, n}'
awk -F', ' '{u+=$1; m=$2; n++} END {if(n) printf "   host GPU: %.0f%% busy, %d MiB VRAM\n", u/n, m}' $GPU
rm -f $GPU
