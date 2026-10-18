@echo off
powershell.exe -NoProfile -STA -File "%~dp0scripts\import_video.ps1"
if errorlevel 1 pause
