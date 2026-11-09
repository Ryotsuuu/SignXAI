@echo off
REM SignXAI convenience launcher for Windows.
REM v3 fix: users were copy-pasting ".venv/bin/python main.py" (the
REM macOS/Linux path shown in some docs) instead of the Windows path
REM ".venv\Scripts\python.exe main.py", which PowerShell correctly reports
REM as "not recognized". This script removes the need to remember which
REM path goes with which OS -- just run: run.bat
REM (This does NOT replace install.py/main.py as the required cross-platform
REM entry points -- it's an optional convenience on top of them.)
REM
REM v5 fix (Bug #24, PROJECT_MEMORY_v6.md): this used a bare relative path
REM (".venv\Scripts\python.exe"), which only worked if the CURRENT
REM DIRECTORY happened to already be the project folder -- e.g. running it
REM via a shortcut with a different "Start in" folder, or from
REM `cmd /c D:\signxai\run.bat` while sitting in a different directory,
REM silently failed with "not found". `%~dp0` is the directory THIS batch
REM file actually lives in, regardless of the caller's current directory or
REM what the project folder is named or where it's located -- so `cd /d
REM "%~dp0"` first makes this work from anywhere, exactly like main.py/
REM install.py already do via `Path(__file__).resolve().parent`.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run: python install.py
    exit /b 1
)
.venv\Scripts\python.exe main.py
