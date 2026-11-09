"""Global configuration and filesystem paths for SignXAI.

All runtime data (profiles, reports, model cache pointers) lives under ``data/``
inside the project root so the application is fully self-contained and portable.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels above this file (app/config.py -> project root)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
PROFILES_DIR: Path = DATA_DIR / "profiles"
REPORTS_DIR: Path = DATA_DIR / "reports"
MODELS_DIR: Path = DATA_DIR / "models"          # local HF cache for vision LLMs
WEB_DIR: Path = ROOT_DIR / "web"

# v2: population-level feature-fusion calibration (see docs/TRAINING.md).
# This is NOT a pretrained black-box model — it's sklearn logistic-regression
# coefficients over the same interpretable classical features engine.py
# already computes, fit once on a public labelled dataset to replace guessed
# fusion weights with data-informed ones. See PROJECT_MEMORY_v8.md Section 1
# for why this doesn't violate the "no pretraining" architectural invariant.
KAGGLE_ZIP_PATH: Path = ROOT_DIR / "kaggle-dataset-sign.zip"
SIGN_DATASET_DIR: Path = ROOT_DIR / "Sign Dataset"
POPULATION_WEIGHTS_PATH: Path = DATA_DIR / "population_weights.json"

# Normalized canvas every signature is fitted onto (width, height).
CANVAS_W: int = 512
CANVAS_H: int = 256

# Score-fusion / verdict behaviour
INCONCLUSIVE_BAND: float = 0.07     # scores within this band below threshold -> INCONCLUSIVE
MIN_THRESHOLD: float = 0.30
DEFAULT_THRESHOLD: float = 0.55     # used when only one reference sample exists
CONFIDENCE_STEEPNESS: float = 14.0  # logistic slope for confidence mapping

# Learning rates for online feedback adaptation
FEEDBACK_THRESHOLD_STEP: float = 0.015
FEEDBACK_WEIGHT_STEP: float = 0.08

# v5 (Bug #23, PROJECT_MEMORY_v8.md): cap on how many reference samples a
# profile can grow to via confirmed-GENUINE feedback (test-time
# calibration, see engine.py::feedback). Bounded so a profile doesn't grow
# unboundedly large and slow verify() down (every verify compares against
# every reference, see Bug #21) -- 12 matches the existing SamplesJSON
# enrolment cap in api/schemas.py, kept as a single source of truth here.
MAX_REFERENCES: int = 12

# NOTE (Bug #3, PROJECT_MEMORY_v8.md): a USE_DEEP_EMBEDDING flag used to live
# here but was never wired into engine.py/features.py anywhere — dead config
# that implied a future torch/torchvision dependency in the CORE engine,
# which would violate the project's hard "no pretrained model, no torch
# outside app/llm/" rule (see PROJECT_MEMORY_v8.md Section 1). Removed rather
# than implemented — if per-writer classical features prove insufficient
# after Phase 2 benchmarking, revisit as an explicit decision, not a
# silently-reintroduced flag.

APP_NAME: str = "SignXAI"
APP_TITLE: str = "SignXAI — Handwritten Signature Verification Using Explainable AI"


def ensure_dirs() -> None:
    """Create the runtime data directories if they do not exist yet."""
    for d in (DATA_DIR, PROFILES_DIR, REPORTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
