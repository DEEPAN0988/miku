@echo off
title Miku Elevated Task Registration
echo ===================================================
echo Registering Miku Elevated Task for Wuthering Waves
echo ===================================================
schtasks /create /tn "Miku_Elevated_wuthering_waves" /tr "\"\"C:\Program Files\Wuthering Waves\launcher.exe\"\"" /sc ONCE /st 00:00 /rl HIGHEST /f
echo.
if %errorlevel% neq 0 (
    echo [ERROR] Task registration failed.
    echo Please ensure you right-clicked this file and selected "Run as administrator".
) else (
    echo [SUCCESS] Task created successfully!
    echo Launching Wuthering Waves via task...
    schtasks /run /tn "Miku_Elevated_wuthering_waves"
)
echo.
pause
