"""Pydantic request/response schemas for the REST API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SamplesJSON(BaseModel):
    """Enrolment samples sent as canvas data-URLs."""
    images: list[str] = Field(min_length=1, max_length=12)


class VerifyRequest(BaseModel):
    profile_id: str
    image: str                      # data-URL or base64 PNG from the UI


class FeedbackRequest(BaseModel):
    correct: bool
    # v5 (Bug #23, PROJECT_MEMORY_v8.md): opt-in test-time calibration -- if
    # true AND the case was a confirmed-correct GENUINE verdict, the query
    # signature is promoted into the profile as a new reference sample.
    # Defaults to false so existing callers/UI keep the v4 behaviour
    # (threshold/weight nudge only) unless they explicitly opt in.
    add_as_reference: bool = False
    # v6 (Bug #29, PROJECT_MEMORY_v8.md): an optional free-text remark the
    # examiner can attach to their feedback (e.g. "the tail flourish was
    # cut off in this photo") -- stored alongside the feedback event for
    # later review, never parsed or acted on automatically.
    remark: str | None = None


class ErrorResponse(BaseModel):
    detail: str
