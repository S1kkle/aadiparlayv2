@echo off
setlocal
cd /d "%~dp0"

REM Launch PowerShell script (no typing required)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart-dev.ps1"

endlocal
