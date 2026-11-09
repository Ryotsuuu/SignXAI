#!/usr/bin/env python3
"""SignXAI — web application entry point.

BUG #2 FIX (PROJECT_MEMORY_v8.md): this file did not exist in the original
prototype. Run with:

    python main.py

Then open http://127.0.0.1:8420 in a browser. Cross-platform — no shell
scripts, works identically on Windows/macOS/Linux (see PROJECT_MEMORY_v8.md
Section 1, "cross-platform is a hard constraint").
"""
from __future__ import annotations

import logging
import sys
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.api.routes import router as api_router          # noqa: E402
from app.config import APP_TITLE, WEB_DIR, ensure_dirs   # noqa: E402
from app.llm.manager import get_manager                  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
log = logging.getLogger("signxai.main")

app = FastAPI(title=APP_TITLE, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local-first tool; tighten if you expose this beyond localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def _startup_llm_autostart() -> None:
    """v5 (Bug #22, PROJECT_MEMORY_v8.md): "when the web page is launched, it
    checks two things: [1] model path & size, [2] GPU availability, and
    starts loading the model on GPU" -- this is that check, run once when
    the server process starts. Runs in a background thread inside
    autostart() itself, so this never delays the server actually coming
    up. If the model isn't on disk yet, or hardware isn't viable, this is
    a fast no-op and the app runs exactly as before (template reports)."""
    try:
        get_manager().autostart()
    except Exception:
        log.exception("LLM autostart failed to even start — app continues "
                      "without it; template reports still work.")

# Serve the web frontend (index.html + assets) as static files, mounted
# LAST so it doesn't shadow /api/*.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    log.warning("web/ directory not found at %s — API-only mode, no UI.", WEB_DIR)


def main() -> None:
    ensure_dirs()
    import uvicorn

    host, port = "127.0.0.1", 8420
    url = f"http://{host}:{port}"
    log.info("%s starting at %s", APP_TITLE, url)
    log.info("API docs available at %s/docs", url)
    try:
        webbrowser.open(url)
    except Exception:
        pass  # headless environment — fine, just print the URL
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
