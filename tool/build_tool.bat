@echo off
cd /d "%~dp0"
set "CC=..\gcc\bin\gcc.exe"
if not exist "%CC%" set "CC=..\gcc\Library\bin\x86_64-w64-mingw32-gcc.exe"
if not exist "%CC%" set "CC=gcc"
"%CC%" -O2 -DSDL_MAIN_HANDLED main.c gfx.c music.c -o ..\SwitchRecomp.exe -Ideps\SDL2-2.30.9\x86_64-w64-mingw32\include\SDL2 -Ideps\SDL2_image-2.8.2\x86_64-w64-mingw32\include\SDL2 -Ideps\SDL2_ttf-2.22.0\x86_64-w64-mingw32\include\SDL2 -Ldeps\SDL2-2.30.9\x86_64-w64-mingw32\lib -Ldeps\SDL2_image-2.8.2\x86_64-w64-mingw32\lib -Ldeps\SDL2_ttf-2.22.0\x86_64-w64-mingw32\lib -static -lSDL2_image -lSDL2_ttf -lSDL2 -mwindows -lsetupapi -lwinmm -limm32 -lversion -lole32 -loleaut32 -luuid -lgdi32 -lcfgmgr32 -lshell32 -lcomdlg32 -lshlwapi -lusp10 -lrpcrt4 -lm
exit /b %errorlevel%
