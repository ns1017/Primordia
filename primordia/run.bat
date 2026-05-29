@echo off
REM Quick launcher for Windows (double-click this file)
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing / updating dependencies (pyglet + numpy)...
pip install -e . --quiet

echo.
echo Starting Primordia...
echo.

python -m primordia

pause
