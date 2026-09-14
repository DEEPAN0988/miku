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

:: Delete existing task if present
schtasks /delete /tn "Miku_Elevated_wuthering_waves" /f >nul 2>&1

:: Register with exact Game .exe and Working Directory via PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command "$action = New-ScheduledTaskAction -Execute 'C:\Program Files\Wuthering Waves\Wuthering Waves Game\Wuthering Waves.exe' -WorkingDirectory 'C:\Program Files\Wuthering Waves\Wuthering Waves Game'; $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest; Register-ScheduledTask -TaskName 'Miku_Elevated_wuthering_waves' -Action $action -Principal $principal -Force"

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Task registration failed.
) else (
    echo [SUCCESS] Task created successfully with WorkingDirectory configured!
    echo Launching Wuthering Waves via task...
    schtasks /run /tn "Miku_Elevated_wuthering_waves"
)
echo.
pause
