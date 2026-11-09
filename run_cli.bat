@echo off
REM See run.bat for the full explanation (v5 Bug #24 fix: cd into this
REM script's own directory first, so it works from any invocation location).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run: python install.py
    exit /b 1
)
.venv\Scripts\python.exe cli.py
