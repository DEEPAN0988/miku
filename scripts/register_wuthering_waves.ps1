$Action = New-ScheduledTaskAction -Execute 'C:\Program Files\Wuthering Waves\launcher.exe' -WorkingDirectory 'C:\Program Files\Wuthering Waves'
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Miku_Elevated_wuthering_waves' -Action $Action -Principal $Principal -Settings $Settings -Force
