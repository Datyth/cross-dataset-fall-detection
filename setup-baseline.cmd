@echo off
setlocal
set "FALL_PROJECT_WIN=%~dp0"
for /f "delims=" %%P in ('wsl.exe -d Ubuntu -- wslpath -a "%FALL_PROJECT_WIN%scripts/setup_baseline_runtime.sh"') do set "FALL_SETUP_SCRIPT=%%P"
for /f "delims=" %%P in ('wsl.exe -d Ubuntu -- wslpath -a "%FALL_PROJECT_WIN%scripts/bootstrap_baseline_system.sh"') do set "FALL_SYSTEM_SCRIPT=%%P"
if not defined FALL_SETUP_SCRIPT exit /b 1
if not defined FALL_SYSTEM_SCRIPT exit /b 1
wsl.exe -d Ubuntu -u root -- bash "%FALL_SYSTEM_SCRIPT%"
if errorlevel 1 exit /b 1
wsl.exe -d Ubuntu -- bash "%FALL_SETUP_SCRIPT%"
