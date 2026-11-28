@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the Python environment from this folder:
    echo   py -3 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r docker\requirements.txt
    exit /b 1
)
".venv\Scripts\python.exe" maya.py %*
exit /b %errorlevel%
