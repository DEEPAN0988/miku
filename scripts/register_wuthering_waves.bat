@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k \"\"%~f0\"\"'"
    exit /b
)

echo ===================================================
echo Registering Miku Elevated Task for Wuthering Waves
echo ===================================================
schtasks /create /tn "Miku_Elevated_wuthering_waves" /tr "\"C:\Program Files\Wuthering Waves\launcher.exe\"" /sc ONCE /st 00:00 /rl HIGHEST /f
echo.
echo Launching Wuthering Waves via task...
schtasks /run /tn "Miku_Elevated_wuthering_waves"
echo.
echo Setup completed successfully!
timeout /t 5
