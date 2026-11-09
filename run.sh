#!/usr/bin/env bash
# SignXAI convenience launcher for macOS/Linux. Mirrors run.bat's purpose:
# one command, no need to remember the .venv path. See run.bat for context.
#
# v5 fix (Bug #24, PROJECT_MEMORY_v6.md): a bare relative path
# (".venv/bin/python") only worked if the caller's CURRENT directory
# happened to already be the project folder -- e.g. `bash
# /some/other/path/run.sh` from elsewhere silently failed. cd into this
# script's own directory first (works no matter where it's called from, or
# what the project folder is named), matching how main.py/install.py
# already resolve their own location via `Path(__file__).resolve().parent`.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ ! -f ".venv/bin/python" ]; then
    echo ".venv not found. Run: python3 install.py"
    exit 1
fi
.venv/bin/python main.py
