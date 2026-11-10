#!/bin/sh
# one run on the NVN path: tools/nvn_run.sh <name> <seconds> [backend null|vulkan] [more env as K=V ...]
name=$1; secs=$2; be=${3:-null}; [ $# -ge 3 ] && shift 3 || shift 2
cd "$(dirname "$0")/.."
env SWR_GPU=nvn NXRT_GFX=$be "$@" python tools/run_game.py --name $name --seconds $secs
