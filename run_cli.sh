#!/usr/bin/env bash
# See run.sh for the full explanation (v5 Bug #24 fix: cd into this
# script's own directory first, so it works from any invocation location).
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ ! -f ".venv/bin/python" ]; then
    echo ".venv not found. Run: python3 install.py"
    exit 1
fi
.venv/bin/python cli.py
