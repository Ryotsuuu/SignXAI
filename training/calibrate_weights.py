#!/usr/bin/env python3
"""v2: Calibrate signature-comparison fusion weights on a real labelled
dataset (per user request — see PROJECT_MEMORY_v8.md v2 section).

WHAT THIS DOES AND DOES NOT DO (read this before running)
-----------------------------------------------------------
This does NOT train a neural network / embedding model, and it does NOT
change how any individual person's profile is calibrated (that stays
100% per-person, zero external data, exactly as in v1 — see
app/core/engine.py's module docstring).

What it DOES: the engine already computes 7 classical, interpretable
comparison scores between two signatures (ssim, iou, hog-cosine,
lbp-cosine, hu-cosine, dtw, scalar-profile) and averages them with
hand-guessed weights (see DEFAULT_WEIGHTS in engine.py). This script
fits those 7 weights properly — via logistic regression — using a public
labelled dataset of genuine/forged signature PAIRS, so "how much should
stroke-width matter vs. overall shape" is answered by evidence instead of
a guess. It writes exactly 7 numbers (+ metadata) to
data/population_weights.json. engine.py automatically picks this file up
if it exists (see _load_default_weights()); delete the file to revert to
the v1 hand-guessed weights.

USAGE
-----
1. Place a Kaggle-style signature dataset zip at the project root, named
   exactly: kaggle-dataset-sign.zip
   Expected internal structure (per the user's dataset description):
     001/            <- 10 genuine images for user 001
     001_forg/       <- 10 forged images for user 001
     002/ 002_forg/  <- etc.
2. Run:
     .venv/bin/python training/calibrate_weights.py
   This auto-extracts the zip into "Sign Dataset/" (only once — skips
   extraction if that folder already has content), builds genuine-vs-
   genuine and genuine-vs-forged pairs per user, computes the 7 comparison
   scores for every pair (reusing the EXACT SAME functions engine.py uses
   for real verification — see _score_pair below), fits a logistic
   regression, and writes data/population_weights.json.
3. Restart the app (or it picks it up on next request) — DEFAULT_WEIGHTS
   in engine.py will now use the calibrated values.

This was validated end-to-end against a CEDAR-shaped dataset (same
NNN / NNN_forg folder pattern) during development — see docs/TRAINING.md
for the actual measured results of that validation run. It has not yet
been run against the user's specific Kaggle dataset (the zip wasn't
available in the build sandbox) — that is the one thing to verify on your
own machine and report back on.
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (KAGGLE_ZIP_PATH, POPULATION_WEIGHTS_PATH,   # noqa: E402
                        SIGN_DATASET_DIR, ensure_dirs)
from app.core import align, features, preprocess, similarity, zoning    # noqa: E402
from app.core.engine import _HAND_GUESSED_WEIGHTS                       # noqa: E402

GENUINE_RE = re.compile(r"^(\d+)$")
FORGED_RE = re.compile(r"^(\d+)_forg$")
# v7 FIX (Bug #38, PROJECT_MEMORY_v8.md, self-run, not user-reported --
# found while auditing this script for consistency with engine.py):
# this list previously had only 7 entries and engine.py's real fusion
# (_HAND_GUESSED_WEIGHTS) has had 9 since Bug #31/#33 added "zonal" and
# "construction" -- so `_load_default_weights()`'s exact-key-match check
# meant ANY weights file this script produced was SILENTLY IGNORED at
# load time, forever, no error anywhere. Fixed by computing all 9
# components here too, using the same pairwise (reference-specific)
# functions engine.py now uses (Bug #34/#35) instead of the old
# population-mean/cosine versions, so a freshly-calibrated file is both
# loadable AND calibrated against the same math that will actually score
# real verifications.
FEATURE_ORDER = ["ssim", "iou", "hog", "lbp", "hu", "dtw", "scalar_profile",
                 "zonal", "construction"]


def extract_dataset(zip_path: Path = KAGGLE_ZIP_PATH,
                    dest: Path = SIGN_DATASET_DIR, force: bool = False) -> Path:
    """Auto-extract the Kaggle zip into 'Sign Dataset/', idempotent."""
    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"[skip] {dest} already has content — not re-extracting "
             f"(pass force=True / --force to redo it)")
        return dest
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected dataset zip at {zip_path} — place your Kaggle "
            f"signature dataset there, named exactly '{zip_path.name}'.")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {dest} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # Some Kaggle zips wrap everything in one extra top-level folder —
    # flatten one level if that's the only thing extracted.
    children = list(dest.iterdir())
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        print(f"[flatten] single wrapper folder detected ({inner.name}) — flattening")
        for item in inner.iterdir():
            item.rename(dest / item.name)
        inner.rmdir()
    print(f"[ok] extracted to {dest}")
    return dest


def discover_users(dataset_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Scan for NNN / NNN_forg folder pairs -> {user_id: {genuine: [...], forged: [...]}}"""
    users: dict[str, dict[str, list[Path]]] = {}
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if m := GENUINE_RE.match(child.name):
            users.setdefault(m.group(1), {}).setdefault("genuine", [])
            users[m.group(1)]["genuine"] = sorted(child.glob("*"))
        elif m := FORGED_RE.match(child.name):
            users.setdefault(m.group(1), {}).setdefault("forged", [])
            users[m.group(1)]["forged"] = sorted(child.glob("*"))
    complete = {u: d for u, d in users.items()
               if d.get("genuine") and d.get("forged")}
    print(f"[ok] found {len(users)} user folders, {len(complete)} with both "
         f"genuine + forged samples")
    return complete


def _score_pair(path_a: Path, path_b: Path) -> dict[str, float] | None:
    """Compute the SAME 7 comparison scores engine.py uses for real
    verification, on one pair of images. Returns None on any read/CV
    failure (corrupt image, unreadable format) rather than crashing the
    whole training run over one bad file."""
    try:
        pre_a = preprocess.preprocess(path_a.read_bytes())
        pre_b = preprocess.preprocess(path_b.read_bytes())
        fa = features.extract(pre_a)
        fb = features.extract(pre_b)
        w_gray, w_bin, ok, _best_iou = align.align(pre_a, pre_b)
        if not ok:
            w_gray, w_bin = pre_a.gray, pre_a.binary
        s_ssim = similarity.ssim(w_gray, pre_b.gray)
        s_iou = similarity.ink_iou(w_bin, pre_b.binary)
        s_hog = similarity.cosine_sim(fa["vectors"]["hog"], fb["vectors"]["hog"])
        s_lbp = similarity.cosine_sim(fa["vectors"]["lbp"], fb["vectors"]["lbp"])
        s_dtw = similarity.dtw_similarity(pre_a.binary, pre_b.binary)
        # v7 (Bug #34/#35/#38): std here approximates a real profile's
        # natural-variation estimate from just this one pair, using the
        # same shrinkage-toward-a-relative-prior idea as
        # engine.py::_relearn (a single pair alone is exactly the n=2
        # degenerate case that fix exists for) rather than the old
        # "half the raw difference, no prior" estimate.
        keys = fa["scalars"].keys()
        # v7.2: kept in sync with engine.py::_relearn's `_FEATURE_PRIOR_REL`
        # (Bug #49, PROJECT_MEMORY_v8.md) -- if you change one, change both,
        # or a freshly-calibrated weights file would be tuned against
        # different tolerance assumptions than the engine actually uses at
        # verification time.
        _PRIOR_REL = 0.22
        std = {}
        for k in keys:
            observed = abs(fa["scalars"][k] - fb["scalars"][k]) / 2
            prior = max(_PRIOR_REL * abs((fa["scalars"][k] + fb["scalars"][k]) / 2), 1e-2)
            std[k] = 0.5 * observed + 0.5 * prior   # single pair -> fixed 50/50 blend
        s_hu = similarity.hu_pairwise_sim(fa["scalars"], fb["scalars"], std)
        s_scalar, _ = similarity.pairwise_scalar_sim(
            fa["scalars"], fb["scalars"], std,
            exclude_keys=similarity.CONSTRUCTION_KEYS | similarity.HU_KEYS)
        s_construction, _ = similarity.pairwise_scalar_sim(
            fa["scalars"], fb["scalars"], std, keys=similarity.CONSTRUCTION_KEYS)
        zonal_result = zoning.compare_zones(w_gray, w_bin, pre_b.gray, pre_b.binary)
        s_zonal = zonal_result["aggregate_score"]
        return dict(ssim=s_ssim, iou=s_iou, hog=s_hog, lbp=s_lbp,
                   hu=s_hu, dtw=s_dtw, scalar_profile=s_scalar,
                   zonal=s_zonal, construction=s_construction)
    except Exception as exc:
        print(f"    [skip] {path_a.name} vs {path_b.name}: {exc}")
        return None


def build_pairs(users: dict, max_users: int | None = None,
                max_pairs_per_user: int = 8) -> tuple[list[dict], list[int]]:
    """Genuine-vs-genuine (label 1) and genuine-vs-forged (label 0) pairs,
    capped per user so one prolific user's folder can't dominate the fit."""
    X, y = [], []
    items = list(users.items())[:max_users] if max_users else list(users.items())
    t0 = time.time()
    for i, (uid, d) in enumerate(items):
        genuine, forged = d["genuine"], d["forged"]
        n = 0
        for a in range(len(genuine)):
            for b in range(a + 1, len(genuine)):
                if n >= max_pairs_per_user:
                    break
                scores = _score_pair(genuine[a], genuine[b])
                if scores:
                    X.append(scores); y.append(1); n += 1
        n = 0
        for g in genuine:
            for f in forged:
                if n >= max_pairs_per_user:
                    break
                scores = _score_pair(g, f)
                if scores:
                    X.append(scores); y.append(0); n += 1
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(items)} users processed "
                 f"({len(X)} pairs so far, {time.time()-t0:.0f}s elapsed)")
    return X, y


def fit_weights(X: list[dict], y: list[int]) -> tuple[dict[str, float], float]:
    """Logistic regression over the 7 comparison scores -> normalised
    non-negative weights (same semantics as the hand-guessed ones: they
    sum to 1 and are all >= 0, so the fused score stays in a comparable
    0..1 range to what engine.py/similarity thresholds already expect)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    Xmat = np.array([[row[k] for k in FEATURE_ORDER] for row in X])
    yarr = np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(Xmat, yarr, test_size=0.25,
                                          random_state=42, stratify=yarr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"[eval] held-out AUC={auc:.4f}  accuracy={acc:.4f}  "
         f"(n_train={len(ytr)} n_test={len(yte)})")

    coefs = np.clip(clf.coef_[0], 0, None)   # weights should be non-negative
                                              # (a HIGHER comparison score
                                              # should never argue for FORGED)
    if coefs.sum() <= 0:
        print("[warn] all coefficients clipped to 0 — dataset may be too "
             "small or noisy to calibrate reliably; keeping hand-guessed "
             "weights instead.")
        return dict(_HAND_GUESSED_WEIGHTS), auc
    weights = {k: float(c / coefs.sum()) for k, c in zip(FEATURE_ORDER, coefs)}
    return weights, auc


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--max-users", type=int, default=None,
                    help="cap dataset size for a quick test run")
    args = ap.parse_args()

    ensure_dirs()
    dataset_dir = extract_dataset(force=args.force_extract)
    users = discover_users(dataset_dir)
    if not users:
        print("[FATAL] No user folders with both genuine + forged samples "
             "found. Check the dataset structure matches NNN / NNN_forg.")
        sys.exit(1)

    print(f"\nBuilding comparison pairs (this re-runs the full preprocess -> "
         f"align -> feature-extract -> compare pipeline per pair, so it "
         f"takes a while for large datasets)...")
    X, y = build_pairs(users, max_users=args.max_users)
    print(f"[ok] {len(X)} total pairs ({sum(y)} genuine, {len(y)-sum(y)} forged)")

    if len(X) < 20:
        print("[FATAL] Too few usable pairs to calibrate reliably (<20). "
             "Check that images are readable and the dataset isn't tiny.")
        sys.exit(1)

    weights, auc = fit_weights(X, y)
    print("\nCalibrated weights:")
    for k, v in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {k:16s} {v:.4f}")

    POPULATION_WEIGHTS_PATH.write_text(json.dumps({
        "weights": weights,
        "held_out_auc": auc,
        "n_pairs": len(X),
        "n_users": len(users),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(KAGGLE_ZIP_PATH.name),
    }, indent=2))
    print(f"\n[ok] Saved -> {POPULATION_WEIGHTS_PATH}")
    print("Restart the app for engine.py to pick up the new weights "
         "(or it will on next process start).")


if __name__ == "__main__":
    main()
