@echo off
setlocal
set "FALL_PROJECT_WIN=%~dp0"
for /f "delims=" %%P in ('wsl.exe -d Ubuntu -- wslpath -a "%FALL_PROJECT_WIN%scripts/run_baseline_wsl.sh"') do set "FALL_BASELINE_SCRIPT=%%P"
if not defined FALL_BASELINE_SCRIPT exit /b 1
wsl.exe -d Ubuntu -- bash "%FALL_BASELINE_SCRIPT%" "%~1"
