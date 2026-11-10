#!/bin/sh
# SwitchRecomp (the tool) for Linux: needs the SDL2, SDL2_ttf and SDL2_image development packages to build. the package
# carries its own SDL2 in lib/ (found through the rpath), the system's is used when that folder is missing
cd "$(dirname "$0")"
cc -O2 -Wall -Wno-missing-braces main.c gfx.c music.c glibc_compat.c -o ${1:-../SwitchRecomp.bin} $(pkg-config --cflags --libs sdl2 SDL2_ttf SDL2_image) -lm -Wl,-rpath,'$ORIGIN/lib'
