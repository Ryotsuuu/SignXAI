"""JSON-backed persistence for signer profiles and verification cases.

Layout::

    data/profiles/<profile_id>/profile.json      profile + learned stats
    data/profiles/<profile_id>/ref_<i>.png       enrolled reference images
    data/reports/<case_id>/case.json             verification result
    data/reports/<case_id>/query.png, *.png      evidence & XAI assets
    data/reports/<case_id>/report.{json,html,pdf}

Writes are atomic (temp file + rename) so a crash cannot corrupt a profile.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import PROFILES_DIR, REPORTS_DIR, ensure_dirs


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- profiles
def profile_dir(profile_id: str) -> Path:
    return PROFILES_DIR / profile_id


def create_profile(name: str) -> dict:
    ensure_dirs()
    pid = new_id()
    profile = {
        "id": pid,
        "name": name.strip() or f"Signer {pid[:6]}",
        "created_at": _now(),
        "refs": [],                     # [{image: filename, features: {...}}]
        "stats": {
            "scalar_mean": {},
            "scalar_std": {},
            "threshold": None,
            "intra_scores": [],
            "weights": None,            # None -> engine defaults
        },
        "feedback": [],                 # [{case_id, correct, at}]
        "maturity": 0,                  # samples + feedback events
        "cases": 0,
    }
    _write_json(profile_dir(pid) / "profile.json", profile)
    return profile


def load_profile(profile_id: str) -> dict:
    path = profile_dir(profile_id) / "profile.json"
    if not path.exists():
        raise KeyError(f"Profile not found: {profile_id}")
    return _read_json(path)


def save_profile(profile: dict) -> None:
    _write_json(profile_dir(profile["id"]) / "profile.json", profile)


def list_profiles() -> list[dict]:
    ensure_dirs()
    out = []
    for d in sorted(PROFILES_DIR.iterdir()):
        p = d / "profile.json"
        if p.exists():
            prof = _read_json(p)
            out.append({
                "id": prof["id"], "name": prof["name"],
                "created_at": prof["created_at"],
                "n_references": len(prof["refs"]),
                "maturity": prof.get("maturity", 0),
                "cases": prof.get("cases", 0),
                "threshold": prof["stats"].get("threshold"),
            })
    return out


def find_profiles_by_name(name: str) -> list[dict]:
    """Case-insensitive exact-name match. Returns full profile summaries
    (via list_profiles) so a caller can decide: add to an existing profile,
    or force-create a new one (two people can share a name — this returns
    ALL matches, disambiguation is the caller's job, not silently guessed)."""
    needle = name.strip().lower()
    return [p for p in list_profiles() if p["name"].strip().lower() == needle]


def delete_profile(profile_id: str) -> bool:
    d = profile_dir(profile_id)
    if not d.exists():
        return False
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    return True


# ------------------------------------------------------------------- cases
def case_dir(case_id: str) -> Path:
    return REPORTS_DIR / case_id


def save_case(case: dict) -> None:
    _write_json(case_dir(case["case_id"]) / "case.json", case)


def load_case(case_id: str) -> dict:
    path = case_dir(case_id) / "case.json"
    if not path.exists():
        raise KeyError(f"Case not found: {case_id}")
    return _read_json(path)


def list_cases(limit: int = 20) -> list[dict]:
    ensure_dirs()
    cases = []
    for d in sorted(REPORTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        p = d / "case.json"
        if p.exists():
            c = _read_json(p)
            cases.append({k: c.get(k) for k in
                          ("case_id", "profile_id", "profile_name", "verdict",
                           "confidence", "score", "created_at")})
        if len(cases) >= limit:
            break
    return cases
