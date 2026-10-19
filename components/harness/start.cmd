@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Missing project .venv. Follow the installation section in README.md first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\start-local.py" %*
exit /b %errorlevel%
