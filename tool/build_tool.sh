#!/bin/sh
# SwitchRecomp.exe (the tool). needs the SDL2, SDL2_ttf and SDL2_image mingw devel packages in $SDLW
cd "$(dirname "$0")"
SDLW=${SDLW:-./deps}
I="-I$SDLW/SDL2-2.30.9/x86_64-w64-mingw32/include/SDL2 -I$SDLW/SDL2_ttf-2.22.0/x86_64-w64-mingw32/include/SDL2 -I$SDLW/SDL2_image-2.8.2/x86_64-w64-mingw32/include/SDL2"
L="-L$SDLW/SDL2-2.30.9/x86_64-w64-mingw32/lib -L$SDLW/SDL2_ttf-2.22.0/x86_64-w64-mingw32/lib -L$SDLW/SDL2_image-2.8.2/x86_64-w64-mingw32/lib"
x86_64-w64-mingw32-gcc -O2 -Wall -Wno-missing-braces -DSDL_MAIN_HANDLED main.c gfx.c music.c -o ${1:-SwitchRecomp.exe} $I $L -static \
  -Wl,-Bstatic -lSDL2_image -lSDL2_ttf -lSDL2 -mwindows \
  -lsetupapi -lwinmm -limm32 -lversion -lole32 -loleaut32 -luuid -lgdi32 -lcfgmgr32 -lshell32 -lcomdlg32 -lshlwapi -lusp10 -lrpcrt4 -lm
