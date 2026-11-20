@echo off
setlocal
cd /d "%~dp0"
echo Running LANS, PPO, LQR and MPC on your Windows CPU.
echo Please let this window finish. The default run uses three repeats.
ver > host-tmp.txt
LANS-Timing.exe tape.bin 3 > timing-tmp.csv 2>> host-tmp.txt
if errorlevel 1 goto failed
move /y timing-tmp.csv timing.csv > nul
if errorlevel 1 goto failed
move /y host-tmp.txt host.txt > nul
if errorlevel 1 goto failed
echo.
type timing.csv
echo.
echo Done. Send timing.csv and host.txt for the comparison.
if not defined LANS_NO_PAUSE pause
exit /b 0

:failed
echo.
echo Benchmark failed. Details are shown below.
if exist host-tmp.txt type host-tmp.txt
if not defined LANS_NO_PAUSE pause
exit /b 1
