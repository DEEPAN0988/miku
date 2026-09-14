# Miku Application Launch Resilience & Stress Test Report

- **Start Time**: 2026-09-14 18:52:56
- **Elapsed Time**: 14.6 minutes (876 seconds)
- **Rounds Completed**: 1
- **Total Tests Executed**: 117
- **Successful Launches**: 117
- **Failed Launches**: 0
- **Success Rate**: `100.0%`

## Recent Test Log (Last 40 Invocations)

| Time | Application Name | Mode | Status | PID | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 19:01:39 | `visual studio installer` | `DIRECT_STARTFILE` | **SUCCESS** | 19680 | Launched application 'visual studio installer' via target... |
| 19:01:43 | `developer command prompt for vs` | `DIRECT_STARTFILE` | **SUCCESS** | 8992 | Launched application 'developer command prompt for vs' vi... |
| 19:01:47 | `developer powershell for vs` | `DIRECT_STARTFILE` | **SUCCESS** | 35872 | Launched application 'developer powershell for vs' via ta... |
| 19:01:52 | `application verifier (x64)` | `COMPAT_RUNASINVOKER` | **SUCCESS** | 24500 | Launched application 'application verifier (x64)' via Run... |
| 19:02:10 | `application verifier (wow)` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 24232 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:02:14 | `windows app cert kit` | `COMPAT_RUNASINVOKER` | **SUCCESS** | 31860 | Launched application 'windows app cert kit' via RunAsInvo... |
| 19:02:31 | `sample desktop apps` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 11648 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:02:49 | `sample uwp apps` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 50944 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:03:08 | `tools for desktop apps` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 24960 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:03:26 | `tools for uwp apps` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 30112 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:03:32 | `windows software development kit` | `DIRECT_STARTFILE` | **SUCCESS** | 24760 | Launched application 'windows software development kit' v... |
| 19:03:36 | `windows powershell ise (x86)` | `DIRECT_STARTFILE` | **SUCCESS** | 40488 | Launched application 'windows powershell ise (x86)' via t... |
| 19:03:40 | `windows powershell ise` | `DIRECT_STARTFILE` | **SUCCESS** | 37128 | Launched application 'windows powershell ise' via target ... |
| 19:03:44 | `word` | `DIRECT_STARTFILE` | **SUCCESS** | 34460 | Launched application 'word' via target 'C:\ProgramData\Mi... |
| 19:04:01 | `wsl settings` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 43296 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:04:18 | `wsl` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 37132 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:04:25 | `code.exe` | `DIRECT_STARTFILE` | **SUCCESS** | 30848 | Launched application 'code.exe' via target 'C:\Users\deep... |
| 19:04:53 | `livecaptions` | `DIRECT_STARTFILE` | **SUCCESS** | 32248 | Launched application 'livecaptions' via target 'C:\Users\... |
| 19:04:57 | `magnify` | `DIRECT_STARTFILE` | **SUCCESS** | 25632 | Launched application 'magnify' via target 'C:\Users\deepa... |
| 19:05:02 | `narrator` | `DIRECT_STARTFILE` | **SUCCESS** | 39804 | Launched application 'narrator' via target 'C:\Users\deep... |
| 19:05:06 | `on-screen keyboard` | `DIRECT_STARTFILE` | **SUCCESS** | 26776 | Launched application 'on-screen keyboard' via target 'C:\... |
| 19:05:27 | `voiceaccess` | `DIRECT_RUNAS` | <span style='color:red;'>FAILED</span> | 39072 | [FALLBACK DIRECT ELEVATION] Task bridge unavailable for '... |
| 19:05:31 | `administrative tools` | `DIRECT_STARTFILE` | **SUCCESS** | 17104 | Launched application 'administrative tools' via target 'C... |
| 19:05:36 | `anaconda navigator` | `DIRECT_STARTFILE` | **SUCCESS** | 40060 | Launched application 'anaconda navigator' via target 'C:\... |
| 19:05:40 | `anaconda powershell prompt` | `DIRECT_STARTFILE` | **SUCCESS** | 37280 | Launched application 'anaconda powershell prompt' via tar... |
| 19:05:45 | `anaconda prompt` | `DIRECT_STARTFILE` | **SUCCESS** | 2528 | Launched application 'anaconda prompt' via target 'C:\Use... |
| 19:05:50 | `jupyter notebook` | `DIRECT_STARTFILE` | **SUCCESS** | 50240 | Launched application 'jupyter notebook' via target 'C:\Us... |
| 19:05:55 | `spyder` | `DIRECT_STARTFILE` | **SUCCESS** | 39424 | Launched application 'spyder' via target 'C:\Users\deepa\... |
| 19:05:59 | `brave` | `DIRECT_STARTFILE` | **SUCCESS** | 40900 | Launched application 'brave' via target 'C:\Users\deepa\A... |
| 19:06:05 | `nahimic companion` | `DIRECT_STARTFILE` | **SUCCESS** | 28112 | Launched application 'nahimic companion' via target 'C:\U... |
| 19:06:10 | `onedrive` | `DIRECT_STARTFILE` | **SUCCESS** | 50688 | Launched application 'onedrive' via target 'C:\Users\deep... |
| 19:06:16 | `code` | `DIRECT_STARTFILE` | **SUCCESS** | 7296 | Launched application 'visual studio code' via target 'C:\... |
| 19:07:00 | `windows powershell 5.1 (x86)` | `DIRECT_STARTFILE` | **SUCCESS** | 26272 | Launched application 'windows powershell 5.1 (x86)' via t... |
| 19:07:05 | `powershell` | `DIRECT_STARTFILE` | **SUCCESS** | 19712 | Launched application 'windows powershell 5.1' via target ... |
| 19:07:10 | `wuthering waves` | `DIRECT_STARTFILE` | **SUCCESS** | 27960 | Launched application 'wuthering waves' via target 'C:\Use... |
| 19:07:14 | `calc` | `DIRECT_STARTFILE` | **SUCCESS** | 17704 | Launched application 'calc' via target 'calc.exe'. - Clos... |
| 19:07:19 | `control panel` | `DIRECT_STARTFILE` | **SUCCESS** | 26884 | Launched application 'control panel' via target 'control.... |
| 19:07:22 | `paint` | `DIRECT_STARTFILE` | **SUCCESS** | 38612 | Launched application 'paint' via target 'mspaint.exe'. - ... |
| 19:07:26 | `notepad` | `DIRECT_STARTFILE` | **SUCCESS** | 40320 | Launched application 'notepad' via target 'notepad.exe'. ... |
| 19:07:31 | `terminal` | `DIRECT_STARTFILE` | **SUCCESS** | 44232 | Launched application 'terminal' via target 'wt.exe'. - Cl... |
