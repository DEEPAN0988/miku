@echo off
echo ===================================================
echo Registering Miku Elevated Task for Wuthering Waves
echo ===================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ""%~dp0register_wuthering_waves.ps1""'"
echo Done. You can close this window.
