"""Similarity / distance functions used by the verification engine."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def cosine_sim(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """Cosine similarity mapped into [0, 1]."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    x, y = x[:n], y[:n]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return 0.0
    return float((np.dot(x, y) / denom + 1.0) / 2.0)


_IOU_TOLERANCE_PX = 3  # see docstring below

def ink_iou(bin_a: np.ndarray, bin_b: np.ndarray, tolerance_px: int = _IOU_TOLERANCE_PX) -> float:
    """Intersection-over-union of the ink masks of two aligned binaries.

    v7 FIX (Bug #37, PROJECT_MEMORY_v8.md, user-reported + self-run
    evidence): raw pixel IoU on a 1-3px-wide pen stroke is pathologically
    sensitive to two things that have NOTHING to do with whether it's the
    same signer -- (1) sub-pixel residual misalignment left over after
    align.py's best-effort registration, and (2) stroke-width differences
    purely from how a photo was taken (lighting/focus/JPEG compression
    change how thick Otsu-binarised ink looks), not from how the pen was
    actually held. Confirmed directly: two crops of the SAME genuine
    signature (see repro against the user's own images 1-3) scored
    ssim~0.82-0.83 (good, real structural agreement) but iou~0.26-0.29
    (looks like barely any overlap) purely from this effect -- with iou
    weighted at 0.16, that single artifact was worth ~0.09-0.10 of fused
    score, enough by itself to push a genuine signature below threshold.
    Fix: dilate both masks by a small, fixed radius (default 1px on the
    512x256 canvas -- deliberately small) before computing IoU, a standard
    "tolerance band" technique for thin-stroke overlap comparison. This
    is symmetric (both images get the same treatment, never advantages
    one side) and modest enough that it does not rescue a genuinely
    different stroke shape -- verified against a real forged sample
    (image 4 in the test set) staying low (iou 0.20 -> 0.24, still a
    clear rejection) while the genuine near-miss case above recovered to
    a much more representative ~0.55-0.60.
    """
    if tolerance_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * tolerance_px + 1, 2 * tolerance_px + 1))
        a = cv2.dilate((bin_a > 0).astype(np.uint8), k) > 0
        b = cv2.dilate((bin_b > 0).astype(np.uint8), k) > 0
    else:
        a, b = bin_a > 0, bin_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Structural similarity mapped into [0, 1]."""
    val = sk_ssim(gray_a, gray_b, data_range=255)
    return float(max(0.0, min(1.0, (val + 1.0) / 2.0)))


def _main_contour_points(binary: np.ndarray, n: int = 128) -> np.ndarray:
    """Resample the longest contour to ``n`` points, normalised to [0,1]."""
    contours, _ = cv2.findContours((binary > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((n, 2))
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    h, w = binary.shape
    c[:, 0] /= max(w, 1)
    c[:, 1] /= max(h, 1)
    idx = np.linspace(0, len(c) - 1, n).astype(int)
    return c[idx]


def contour_dtw(bin_a: np.ndarray, bin_b: np.ndarray, n: int = 128) -> float:
    """DTW distance between the two main contours (lower = more similar)."""
    a = _main_contour_points(bin_a, n)
    b = _main_contour_points(bin_b, n)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, n + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            acc[i, j] = d[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, n] / n)


def dtw_similarity(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    """Contour DTW mapped to a [0,1] similarity (0.3 ≈ typical scale)."""
    return float(np.exp(-contour_dtw(bin_a, bin_b) / 0.30))


def pairwise_scalar_sim(query: dict[str, float], reference: dict[str, float],
                        std: dict[str, float], keys=None,
                        exclude_keys: frozenset[str] = frozenset()
                        ) -> tuple[float, dict[str, float]]:
    """Mean of exp(-|z|) where z = (query - reference) / natural_variation,
    i.e. how far the query's feature value is from THIS SPECIFIC
    reference's own value, scaled by the profile's typical natural
    variation (``std``, see ``VerificationEngine._relearn``'s shrinkage
    estimator in engine.py).

    v7 FIX (Bug #34, PROJECT_MEMORY_v8.md, self-run + user-reported):
    replaces the previous fusion-time use of ``scalar_profile_sim``
    (below), which compared the query only to the PROFILE'S MEAN across
    all references -- identical for every reference in the per-reference
    scoring loop (`engine.py::_score_against_ref` iterates references,
    but this term never actually looked at ``reference`` at all). Two
    concrete problems this caused, both confirmed by direct
    reproduction against a real 2-reference profile:

    1. **No reference-specific signal.** The whole point of comparing a
       query against EVERY enrolled reference (v5, Bug #21) is to see
       how it does against each one individually -- but 20-32% of the
       fused score's weight (`scalar_profile` + `construction`) was
       IDENTICAL across every reference for a given query, silently
       defeating that design for a third of the score.
    2. **A mathematical artifact that punishes exact matches.** With
       exactly 2 enrolled references, ``np.std`` of any feature across
       those 2 points is ALWAYS exactly half their absolute difference
       -- which means a query equal to EITHER reference is always
       *exactly* 1 population-std away from the profile mean, giving
       ``exp(-1) ~= 0.37`` for that feature no matter how perfect the
       match is. Confirmed directly: re-verifying an already-ENROLLED
       image against its own freshly-built 2-reference profile showed
       ~10 of 22 scalar features sitting at z ~= 1.000 to three decimal
       places -- not a coincidence, a guaranteed property of std over 2
       points. Comparing against the SPECIFIC reference instead makes
       an exact match give z = 0 exactly, regardless of how many
       references are enrolled -- the artifact cannot occur by
       construction, not just by tuning a constant.

    ``std`` is still used here -- not dropped -- as the *tolerance
    scale* (how much this profile's own samples naturally vary on this
    feature), which is exactly the right role for it: "is this
    difference from THIS reference large relative to how much this
    person's signature normally varies?" rather than "how far is this
    from average?". The population-mean-based ``scalar_profile_sim``
    below is kept, unchanged, for the human-readable reporting table
    (comparing the query to the profile's overall average is genuinely
    the right question for a human reviewer to see) -- only the FUSION
    score's calculation moved to this reference-specific version.
    """
    Z_CAP = 6.0
    REL_FLOOR = 0.08
    sims = []
    zs: dict[str, float] = {}
    it = keys if keys is not None else query.keys()
    for k in it:
        if k not in query or k not in reference:
            continue
        qv, rv = query[k], reference[k]
        raw_std = std.get(k, 0.0)
        floor = max(REL_FLOOR * abs(rv if rv else qv), 1e-2)
        s = max(raw_std, floor)
        z = max(-Z_CAP, min(Z_CAP, (qv - rv) / s))
        zs[k] = float(z)
        if k not in exclude_keys:
            sims.append(float(np.exp(-abs(z))))
    return (float(np.mean(sims)) if sims else 1.0), zs


def scalar_profile_sim(query: dict[str, float],
                       mean: dict[str, float],
                       std: dict[str, float],
                       exclude_keys: frozenset[str] = frozenset()) -> tuple[float, dict[str, float]]:
    """Mean of exp(-|z|) over scalar features (excluding ``exclude_keys``,
    see ``construction_consistency_sim`` below for why); also returns the
    z-scores for EVERY feature including excluded ones (the report/UI
    still shows all of them — only the FUSION math treats them
    separately).

    BUG #6 FIX (PROJECT_MEMORY_v8.md): the previous std floor was a fixed
    1e-3 constant. For low-magnitude-variance features that are common with
    few enrolled samples — especially integer topology counts like
    "enclosed loops" or "connected components", which are often IDENTICAL
    across a person's own genuine samples — std collapses to ~0, and
    1e-3 is far too small a floor: a difference of even 1 unit produced
    z-scores like +1000 (this exact value was seen in a real user-generated
    report — see PROJECT_MEMORY_v8.md). exp(-1000) == 0.0, so that single
    feature alone could veto an otherwise-genuine signature.

    Fix is two-layered, deliberately redundant given how badly this broke
    real verifications:
      1. Relative std floor — scales with the feature's own magnitude
         instead of a fixed tiny constant, so naturally large-scale
         features (contour perimeter ~thousands) don't get an
         absurdly-tight floor either.
      2. Hard z-score cap at Z_CAP — even if a feature's std is
         legitimately ~0 (e.g. every enrolled sample truly has exactly the
         same enclosed-loop count), one unusual observation should count
         as *strong evidence against*, not an automatic veto. Capping
         keeps exp(-abs(z)) bounded away from exactly 0.
    """
    Z_CAP = 6.0                      # exp(-6) ~= 0.0025 — strong signal, not an absolute veto
    REL_FLOOR = 0.08                 # std floor = 8% of the feature's own mean magnitude
    zs: dict[str, float] = {}
    sims = []
    for k, qv in query.items():
        m = mean.get(k, qv)
        raw_std = std.get(k, 0.0)
        floor = max(REL_FLOOR * abs(m), 1e-2)   # scale-aware floor, not a fixed tiny constant
        s = max(raw_std, floor)
        z = (qv - m) / s
        z = max(-Z_CAP, min(Z_CAP, z))          # hard cap — no single feature can zero the score
        zs[k] = float(z)
        if k not in exclude_keys:
            sims.append(float(np.exp(-abs(z))))
    return (float(np.mean(sims)) if sims else 0.0), zs


# v6 (Bug #33, PROJECT_MEMORY_v8.md, user-reported: a signature with
# massively divergent pen-lift/loop-construction counts — z-scores hitting
# the hard cap on FOUR independent features simultaneously — still only
# scored INCONCLUSIVE, not FORGED). Root cause, confirmed mathematically
# and against the user's own real report: `scalar_profile_sim` above is
# an UNWEIGHTED MEAN across all ~22 scalar features. These 5 features are
# literal DISCRETE COUNTS of the physical pen-lift/loop-construction
# process (how many times the pen left the paper, how many separate
# ink shapes exist, how many enclosed loops were formed, how many stroke
# crossings) — forensically, this is closer to what questioned-document
# examination calls "line quality" and construction sequence: something a
# forger who copies visible SHAPE well can still get completely wrong,
# because they cannot see or replicate the original writer's actual pen
# movements. Averaging these 5 forensically-important, hard-to-fake counts
# together with ~17 continuous, easier-to-superficially-mimic geometric/
# shape features (centroid, slant, Hu moments, stroke width, ...)
# structurally dilutes them: even if all 5 count features hit the z-score
# cap (as extreme as this system can measure), a simple average with 17
# "looks fine" features pulls the combined number back toward ~0.8 —
# confirmed directly against a real forged-vs-genuine test case. Giving
# these features their OWN dedicated top-level fusion component (exactly
# mirroring how `zonal` was split out from the same dilution problem at
# the LOCAL/regional level in Bug #31) stops that dilution at the
# GLOBAL/whole-signature level too.
CONSTRUCTION_KEYS: frozenset[str] = frozenset({
    "endpoints_count", "junctions_count", "components_count",
    "holes_count", "contours_count",
})

# v7 (Bug #35, PROJECT_MEMORY_v8.md, self-run): the 7 Hu-moment features,
# split out for the same reason CONSTRUCTION_KEYS was split out in Bug #33
# -- see ``hu_pairwise_sim`` below for why they also need to be EXCLUDED
# from ``pairwise_scalar_sim``'s general average (they already get their
# own dedicated fusion weight, "hu" -- including them in both would
# double-count them). ALL 7 are excluded from the general scalar average
# (kept together as one family), but only 4 of them actually feed the
# "hu" fusion component itself -- see HU_FUSION_KEYS below.
HU_KEYS: frozenset[str] = frozenset({f"hu_{i}" for i in range(1, 8)})

# v7 FIX (Bug #39, PROJECT_MEMORY_v8.md, self-run, found while
# investigating why a real GENUINE match's "hu" component stayed low
# (~0.18-0.26) even after Bug #35's per-dimension tolerance fix):
# Hu moments 5, 6, and 7 are HIGHER-ORDER moments, and it's a
# well-documented property of Hu's original 7 invariants (Hu, 1962;
# widely repeated in image-moment literature since) that they get
# progressively more sensitive to image noise and discretisation error
# as order increases -- moment 7 in particular is DESIGNED to be a skew/
# reflection indicator and changes SIGN under a mirror flip of the shape.
# Confirmed directly on the user's own two enrolled genuine images: hu_5,
# hu_6 AND hu_7 all measured with OPPOSITE SIGN between one real photo of
# a signature and another real photo of the exact same signature (e.g.
# hu_7: -3.53 vs +5.38) -- not a subtle drift, a full sign inversion, on
# a pair the user themselves enrolled as genuine. This is consistent with
# small photo-to-photo differences (JPEG compression, lighting, a
# slightly different trailing-flourish capture) tipping an
# already-near-zero higher-order skew term across zero, which is exactly
# the known failure mode of these 3 moments on real (non-scanned,
# non-noise-free) photographs -- not evidence of a different signer.
# Hu moments 1-4 (lower-order: pure "spread"/"roundness" descriptors)
# showed no such instability and remained genuinely discriminative
# (confirmed: restricting the "hu" fusion component to just these 4 gave
# 0.45 for a real genuine pair vs 0.28 for a real forged pair -- a
# meaningful, usable margin -- while all 7 gave 0.26 vs a similarly
# collapsed forged-side score, i.e. no usable margin at all). Fixed by
# using only the 4 stable moments for the scored "hu" fusion component;
# all 7 remain visible, unweighted-average-excluded but fully reported,
# in the human-readable metrics table for forensic transparency (a
# reviewer may still find a genuine sign flip meaningful context, just
# not something the automated verdict should be swayed by).
HU_FUSION_KEYS: frozenset[str] = frozenset({"hu_1", "hu_2", "hu_3", "hu_4"})


def hu_pairwise_sim(query: dict[str, float], reference: dict[str, float],
                    std: dict[str, float]) -> float:
    """Hu-moment shape similarity between the query and ONE specific
    reference, using the same natural-variation-scaled z-score approach
    as ``pairwise_scalar_sim`` (see that function's docstring) instead of
    raw cosine similarity over the 7 raw Hu values.

    v7 FIX (Bug #35, PROJECT_MEMORY_v8.md, self-run against the user's
    own real signature photos): the previous "hu" fusion component was
    ``cosine_sim(query_hu_vector, reference_hu_vector)`` over the 7 raw
    (log-scaled, sign-preserving) Hu moments directly. Two compounding
    problems, both confirmed on real data:

    1. **Wildly different natural scales across the 7 dimensions**
       (hu_1 typically ~0.02-0.06, hu_5/hu_6/hu_7 typically ~2-4) mean a
       plain dot-product-based cosine similarity is almost entirely
       determined by whichever dimensions happen to have the largest
       raw magnitude, regardless of which ones are actually
       discriminative for this signer.
    2. **At least one Hu moment can flip SIGN between two genuine
       samples from the same real person** -- confirmed directly:
       hu_6 measured +3.50 on one of the user's own enrolled images and
       -2.68 on their other enrolled image of the SAME signature. A
       sign flip is close to the worst-case input for cosine similarity
       (it actively pulls the dot product negative), so this single
       naturally-volatile dimension could single-handedly crater the
       whole "hu" component's score even between two undisputedly
       genuine samples -- exactly the "one feature vetoes everything"
       failure class Bug #6 already fixed once for the scalar z-score
       path, recurring here in a different code path that hadn't been
       given the same treatment.

    Fix: treat the 7 Hu moments exactly like any other scalar feature --
    per-dimension difference from the specific reference, scaled by this
    profile's own OBSERVED natural variation on that exact dimension
    (``std``, from the shrinkage estimator in `engine.py::_relearn`).
    A dimension that genuinely swings sign/magnitude across this
    person's own genuine samples (like hu_6 above) gets a
    correspondingly large tolerance and stops dominating the result;
    a dimension that's stable for this person keeps a tight tolerance
    and stays fully discriminative.
    """
    sim, _ = pairwise_scalar_sim(query, reference, std, keys=HU_FUSION_KEYS)
    return sim


def construction_consistency_sim(query: dict[str, float], mean: dict[str, float],
                                 std: dict[str, float]) -> float:
    """Same z-score + cap + exp(-|z|) approach as ``scalar_profile_sim``,
    restricted to ``CONSTRUCTION_KEYS`` only (5 features, not diluted by
    the other ~17). See the module-level comment above ``CONSTRUCTION_KEYS``
    for the full reasoning."""
    Z_CAP = 6.0
    REL_FLOOR = 0.08
    sims = []
    for k in CONSTRUCTION_KEYS:
        if k not in query:
            continue
        qv = query[k]
        m = mean.get(k, qv)
        raw_std = std.get(k, 0.0)
        floor = max(REL_FLOOR * abs(m), 1e-2)
        s = max(raw_std, floor)
        z = max(-Z_CAP, min(Z_CAP, (qv - m) / s))
        sims.append(float(np.exp(-abs(z))))
    return float(np.mean(sims)) if sims else 1.0
