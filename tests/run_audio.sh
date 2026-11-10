#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
testbin=$(mktemp -d)
trap 'rm -rf "$testbin"' EXIT
cc -O1 -g -fsanitize=address,undefined -Iruntime tests/test_audio.c runtime/audio.c -lm -ldl -lpthread -o "$testbin/audio"
ASAN_OPTIONS=detect_leaks=0 SDL_AUDIODRIVER=dummy "$testbin/audio"
cc -O1 -g -fsanitize=address,undefined -Iruntime tests/test_record_time.c -lm -o "$testbin/time"
ASAN_OPTIONS=detect_leaks=0 "$testbin/time"
