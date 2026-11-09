"""REST API routes — wires VerificationEngine + report generator to HTTP.

BUG #2 FIX (PROJECT_MEMORY_v8.md Section 3): this file did not exist before —
`engine.py` and `llm/report.py` were fully implemented but nothing called
them. This is the missing wiring.

Name-collision workflow (per project brief: "check if ID is new or old"):
POST /profiles first calls storage.find_profiles_by_name(). If matches
exist, it returns them with `needs_disambiguation: true` instead of
silently creating a duplicate profile OR silently merging into a stranger's
profile — the caller (web UI / CLI) then explicitly confirms which profile
to use, or passes `force_new=true` to create a new one anyway (two people
can share a name).
"""
from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from ..config import REPORTS_DIR
from ..core import storage
from ..core.engine import VerificationEngine
from ..llm import report as report_mod
from ..llm.manager import get_manager
from .schemas import (ErrorResponse, FeedbackRequest, ProfileCreate,
                      SamplesJSON, VerifyRequest)

router = APIRouter()
engine = VerificationEngine()


def _decode_data_url(data_url: str) -> bytes:
    """Accept either a raw base64 string or a full data: URL from a canvas."""
    if "," in data_url and data_url.strip().startswith("data:"):
        data_url = data_url.split(",", 1)[1]
    try:
        return base64.b64decode(data_url, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, f"Invalid base64 image data: {exc}")


# --------------------------------------------------------------- profiles
@router.post("/profiles/check-name")
def check_name(body: ProfileCreate):
    """Look up existing profiles by name BEFORE enrolling — lets the UI ask
    'is this the same person as an existing profile?' instead of guessing."""
    matches = storage.find_profiles_by_name(body.name)
    return {"matches": matches, "needs_disambiguation": len(matches) > 0}


@router.post("/profiles")
def create_profile(body: ProfileCreate, force_new: bool = False):
    if not force_new:
        matches = storage.find_profiles_by_name(body.name)
        if matches:
            raise HTTPException(
                409, detail=(
                    f"{len(matches)} existing profile(s) named '{body.name}'. "
                    "Call /profiles/{id}/enroll on the existing profile to add "
                    "samples for the same person, or retry with force_new=true "
                    "to create a separate profile."))
    profile = storage.create_profile(body.name)
    return {"profile_id": profile["id"], "name": profile["name"]}


@router.get("/profiles")
def list_profiles():
    return storage.list_profiles()


@router.get("/profiles/{profile_id}")
def get_profile(profile_id: str):
    try:
        return storage.load_profile(profile_id)
    except KeyError:
        raise HTTPException(404, "Profile not found")


@router.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str):
    ok = storage.delete_profile(profile_id)
    if not ok:
        raise HTTPException(404, "Profile not found")
    return {"deleted": True}


@router.post("/profiles/{profile_id}/enroll")
def enroll(profile_id: str, body: SamplesJSON):
    """Enroll 1-12 genuine samples (project brief: min 1, typical 1-5)."""
    try:
        images = [_decode_data_url(s) for s in body.images]
        return engine.enroll(profile_id, images)
    except KeyError:
        raise HTTPException(404, "Profile not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ------------------------------------------------------------ verification
@router.post("/verify")
def verify(body: VerifyRequest):
    try:
        img = _decode_data_url(body.image)
        case = engine.verify(body.profile_id, img)
    except KeyError:
        raise HTTPException(404, "Profile not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # Kick off the report (LLM narrative if ready, deterministic template
    # otherwise — report.py owns that decision, see its module docstring).
    report = report_mod.generate_report(case["case_id"])
    return {"case": case, "report": report}


@router.post("/cases/{case_id}/feedback")
def feedback(case_id: str, body: FeedbackRequest):
    try:
        return engine.feedback(case_id, body.correct, body.add_as_reference, body.remark)
    except KeyError:
        raise HTTPException(404, "Case not found")


@router.get("/cases")
def list_cases(limit: int = 20):
    return storage.list_cases(limit)


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    try:
        return storage.load_case(case_id)
    except KeyError:
        raise HTTPException(404, "Case not found")


@router.get("/cases/{case_id}/asset/{filename}")
def get_case_asset(case_id: str, filename: str):
    """Serve XAI evidence images (heatmap/overlay/skeleton/annotated/query/reference)."""
    path = REPORTS_DIR / case_id / filename
    if not path.exists() or ".." in filename:
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)


@router.get("/reports/{case_id}")
def get_report_json(case_id: str):
    path = REPORTS_DIR / case_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "Report not found — verify the case first")
    return FileResponse(path, media_type="application/json")


@router.get("/reports/{case_id}/html")
def get_report_html(case_id: str):
    path = REPORTS_DIR / case_id / "report.html"
    if not path.exists():
        raise HTTPException(404, "Report not found — verify the case first")
    return FileResponse(path, media_type="text/html")


@router.get("/reports/{case_id}/pdf")
def get_report_pdf(case_id: str):
    path = REPORTS_DIR / case_id / "report.pdf"
    if not path.exists():
        raise HTTPException(404, "Report PDF not found — verify the case first")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"signxai_report_{case_id}.pdf")


# --------------------------------------------------------------------- LLM
@router.get("/llm/status")
def llm_status():
    return get_manager().status()


@router.post("/llm/download")
def llm_download(background_tasks: BackgroundTasks):
    mgr = get_manager()
    if not mgr.device.viable:
        raise HTTPException(400, "Hardware below minimum requirement — LLM disabled")
    if mgr.status()["status"] in ("downloading", "loading"):
        return {"ok": True, "reason": "already in progress"}
    if mgr.status()["status"] in ("downloaded", "ready"):
        return {"ok": True, "reason": "model already present on disk — use "
                "POST /api/llm/load instead of re-downloading"}
    background_tasks.add_task(mgr.download_blocking)
    return {"ok": True, "reason": "download started"}


@router.post("/llm/load")
def llm_load(background_tasks: BackgroundTasks):
    """v5 (Bug #22, PROJECT_MEMORY_v8.md): explicit manual retry for loading a
    model that's already on disk (e.g. right after autostart's background
    thread finishes, or after a user manually copies weights in while the
    server is already running — refresh_device()/model_present_on_disk()
    checks the filesystem fresh every call, so this picks that up without
    a restart)."""
    mgr = get_manager()
    mgr.refresh_device()
    if not mgr.device.viable:
        raise HTTPException(400, "Hardware below minimum requirement — LLM disabled")
    status = mgr.status()["status"]
    if status in ("loading", "ready"):
        return {"ok": True, "reason": f"already {status}"}
    if status == "not_installed":
        raise HTTPException(400, "Model not found on disk yet — download it first "
                            "(POST /api/llm/download) or copy its files into "
                            "data/models/.")
    background_tasks.add_task(mgr.load_now)
    return {"ok": True, "reason": "load started"}
