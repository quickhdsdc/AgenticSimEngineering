@echo off
setlocal
if not defined A4S_HOME set "A4S_HOME=%~dp0work\runs\harness\home"
if not defined A4S_WORK_DIRECTORY set "A4S_WORK_DIRECTORY=%~dp0work\runs\harness"
if not defined A4S_LIVE_MONITOR_DB set "A4S_LIVE_MONITOR_DB=%~dp0work\runs\harness\festo-live\monitor.db"
if not defined A4S_ENV_FILE if exist "%~dp0.env" set "A4S_ENV_FILE=%~dp0.env"
"%~dp0.venv\Scripts\python.exe" "%~dp0components\harness\scripts\start-local.py" %*
exit /b %errorlevel%
