@echo off
title Miku Elevated Task Registration

:: Check for Administrator privileges; auto-elevate if not admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Administrative privileges required. Requesting UAC elevation...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd.exe -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

echo ===================================================
echo Registering Miku Elevated Task for Wuthering Waves
echo ===================================================
schtasks /create /tn "Miku_Elevated_wuthering_waves" /tr "\"\"C:\Program Files\Wuthering Waves\launcher.exe\"\"" /sc ONCE /st 00:00 /rl HIGHEST /f
echo.
if %errorlevel% neq 0 (
    echo [ERROR] Task registration failed.
) else (
    echo [SUCCESS] Task created successfully!
    echo Launching Wuthering Waves via task...
    schtasks /run /tn "Miku_Elevated_wuthering_waves"
)
echo.
pause

