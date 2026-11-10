@echo off
cd /d "%~dp0"
set "CC=gcc\bin\gcc.exe"
if not exist "%CC%" set "CC=gcc\Library\bin\x86_64-w64-mingw32-gcc.exe"
if not exist "%CC%" set "CC=gcc"
python\python.exe switchrecomp.py --build-runtime --target win64 --backend win32 --cc "%CC%"
if errorlevel 1 exit /b %errorlevel%
