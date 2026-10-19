@echo off
REM Isaac Sim GUI launcher.
REM Why this exists: the venv was created under the old folder name (world-model\),
REM so uv's console-script shims (Scripts\isaacsim.exe) still point at that dead path
REM and die with "uv trampoline failed to canonicalize script path". python.exe itself
REM is a real binary and works fine, so we call the entry point directly.
REM
REM Usage:  run_gui.bat                            (default full GUI, PhysX)
REM         run_gui.bat isaacsim.exp.full.newton   (Newton engine GUI)
setlocal
set "OMNI_KIT_ACCEPT_EULA=YES"
set "HERE=%~dp0"
set "EXPERIENCE=%~1"
if "%EXPERIENCE%"=="" set "EXPERIENCE=isaacsim.exp.full.kit"
echo Launching %EXPERIENCE% ... (first run compiles shader cache, can take 5-10 min)
"%HERE%.venv\Scripts\python.exe" -c "import sys, isaacsim; sys.argv=['isaacsim']+sys.argv[1:]; isaacsim.main()" %EXPERIENCE%
endlocal
