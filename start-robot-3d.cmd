@echo off
setlocal DisableDelayedExpansion
title AgenticSimEngineering - Robot 3D
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing.
    echo Run: uv sync --extra harness --extra des-reference
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -u "scripts\launch_robot_3d.py" %*
set "ASE_LAUNCH_EXIT=%errorlevel%"
if not "%ASE_LAUNCH_EXIT%"=="0" (
    echo.
    echo Startup failed. See the message above.
    pause
)
popd
endlocal & exit /b %ASE_LAUNCH_EXIT%
