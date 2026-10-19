@echo off
REM mbs_isaac 统一入口。用法（在仓库根目录执行）：
REM   run_mbs.bat validate_franka
REM   run_mbs.bat validate_closed_loop 32 4
REM   run_mbs.bat run_example --model double_pendulum --episodes 3
REM
REM 环境：projects\isaac_sim\.venv（uv 管理，可用 `cd projects\isaac_sim && uv sync` 重建）
setlocal
set "OMNI_KIT_ACCEPT_EULA=YES"
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%projects"
set "PY=%HERE%projects\isaac_sim\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [错误] 找不到 Isaac 环境: %PY%
  echo 重建: cd projects\isaac_sim ^&^& uv sync
  exit /b 1
)
if "%~1"=="" (
  echo 用法: run_mbs.bat ^<模块名^> [参数...]
  echo   可用模块: run_example ^| validate_franka ^| validate_closed_loop ^| extract_usd
  echo   P1b 求解器对比: validate_closed_loop_newton ^| reference_four_bar ^| compare_p1b
  exit /b 1
)
set "MOD=%~1"
shift
"%PY%" -m mbs_isaac.%MOD% %1 %2 %3 %4 %5 %6 %7 %8
endlocal
