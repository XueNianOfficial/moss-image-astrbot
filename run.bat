@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem 1) repo venv  2) ai-toolkit venv from .install.json or AI_TOOLKIT_DIR  3) PATH
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not defined PY if defined AI_TOOLKIT_DIR if exist "%AI_TOOLKIT_DIR%\.venv\Scripts\python.exe" set "PY=%AI_TOOLKIT_DIR%\.venv\Scripts\python.exe"

if not defined PY if exist ".install.json" (
    for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "(Get-Content -Raw .install.json | ConvertFrom-Json).ai_toolkit"`) do (
        if exist "%%p\.venv\Scripts\python.exe" set "PY=%%p\.venv\Scripts\python.exe"
    )
)

if not defined PY set "PY=python"

echo Python : %PY%
"%PY%" -m moss_server %*
exit /b %ERRORLEVEL%
