@echo off
rem SwitchRecomp: downloads what's missing into this folder, with internet and nothing else to do (no admin rights,
rem nothing gets installed on the system): Python (to recompile games) and GCC (makes recompiled games faster).
rem SwitchRecomp opens it by itself when Python is missing.
title SwitchRecomp - Install
cd /d "%~dp0"
echo SwitchRecomp: installing what is missing (into this folder, nothing goes on the system)
echo.
set "PY=python\python.exe"
if exist "%PY%" goto havepy
echo Python 3.14 (about 16 MB, once)
where curl >nul 2>nul || goto oldwindows
where tar >nul 2>nul || goto oldwindows
if not exist "_downloads" mkdir "_downloads"
curl -L --fail --retry 3 --progress-bar -o "_downloads\python.3.14.9.nupkg" "https://api.nuget.org/v3-flatcontainer/python/3.14.9/python.3.14.9.nupkg" || goto failed
certutil -hashfile "_downloads\python.3.14.9.nupkg" SHA256 | findstr /i /c:"7cd9d07aec975212344dc6f7e45019378cf6775dd564181e269010a1eddaaec7" >nul || goto badsum
if not exist "python" mkdir "python"
tar -xf "_downloads\python.3.14.9.nupkg" -C "python" --strip-components=1 tools || goto failed
if not exist "%PY%" goto failed
rmdir /s /q "_downloads"
echo   done
echo.
:havepy
"%PY%" -u -I "tool\get_deps.py"
if errorlevel 1 goto failed
echo.
echo This window closes by itself in 15 seconds.
timeout /t 15 >nul
exit /b 0
:badsum
echo The Python download is damaged (checksum). Run Install.bat again.
del "_downloads\python.3.14.9.nupkg" >nul 2>nul
goto failed
:oldwindows
echo This Windows has no curl or tar (they come with Windows 10 1803 and newer).
echo Install Python 3 from python.org and SwitchRecomp will use it.
:failed
echo.
echo Something failed. Check the internet connection and run Install.bat again.
pause
exit /b 1
