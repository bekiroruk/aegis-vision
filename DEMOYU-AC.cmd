@echo off
powershell.exe -NoProfile -File "%~dp0scripts\demo.ps1" -OpenBrowser
if errorlevel 1 pause
