"""SignXAI regression tests — v9 (PROJECT_MEMORY_v9.md).

Plain-assertion test runner, no pytest dependency (none of the project's
other tooling requires it — see requirements.txt). Run directly:

    python tests/test_regressions.py

Every test here codifies a real bug found and fixed during the v9 pass,
using either the user's own real (contaminated) enrollment photos when
available, or a small synthetic case when the point being tested is a pure
logic/arithmetic question that doesn't need real image data. The goal is
that a future change which reintroduces any of these bugs fails loudly
here instead of silently shipping — see PROJECT_MEMORY_v9.md for the full
before/after evidence each of these was originally diagnosed from.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from app.core import preprocess as pp
from app.core.engine import VerificationEngine

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _synthetic_signature_bgr(w=600, h=250, ink_bgr=(180, 60, 40)) -> np.ndarray:
    """A plain synthetic 'signature' (a few connected curvy strokes) on
    white paper, with no contamination — used as a clean baseline so
    contamination tests can be layered on top of a known-good starting
    point rather than only ever testing against the user's real photos
    (which may not always be present in every environment this runs in).
    """
    img = np.full((h, w, 3), 250, np.uint8)
    pts = np.array([[60, 150], [120, 60], [160, 180], [220, 70], [280, 150],
                    [340, 90], [400, 150], [460, 100]], np.int32)
    cv2.polylines(img, [pts], False, ink_bgr, 6, cv2.LINE_AA)
    cv2.line(img, (60, 200), (460, 210), ink_bgr, 4, cv2.LINE_AA)
    return img


def test_chromatic_outlier_rejects_neutral_shadow():
    """Bug #55: a large neutral-grey shadow/fold-line band must be
    rejected even when it is the single LARGEST component in the frame —
    anchoring must not itself be fooled by size alone."""
    img = _synthetic_signature_bgr()
    # paint a large, low-saturation (neutral grey) wavy band clearly
    # SEPARATED from the signature strokes (own connected component),
    # mimicking the real paper-fold shadow found in the user's ref_1
    # photo, which sat above the signature with a clean gap.
    cv2.line(img, (0, 8), (599, 20), (150, 150, 150), 12, cv2.LINE_AA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step1 = pp._remove_specks(pp._remove_ruled_lines(pp._binarize(gray)))
    n_before = int((step1 > 0).sum())
    cleaned = pp._remove_chromatic_outliers(img, step1)
    n_after = int((cleaned > 0).sum())
    check("chromatic_outlier_rejects_neutral_shadow", n_after < n_before * 0.7,
         f"before={n_before} after={n_after} (expected a large drop)")


def test_chromatic_outlier_preserves_multipart_signature():
    """Bug #55 regression guard: a real multi-part signature (several
    components, all sharing the SAME ink hue) must be fully preserved —
    the fix must never remove genuine, differently-positioned ink just for
    being a separate connected component."""
    img = _synthetic_signature_bgr()
    # a second, spatially separate stroke group in the SAME ink colour,
    # standing in for a detached initial / letter group.
    cv2.circle(img, (540, 190), 18, (180, 60, 40), 5, cv2.LINE_AA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step1 = pp._remove_specks(pp._remove_ruled_lines(pp._binarize(gray)))
    cleaned = pp._remove_chromatic_outliers(img, step1)
    check("chromatic_outlier_preserves_multipart_signature",
         int((cleaned > 0).sum()) == int((step1 > 0).sum()),
         "same-colour multi-part content must be fully preserved")


def test_distant_outlier_v9_margin_uses_anchor_footprint():
    """Bug #55 refinement: a tiny, barely-'substantial' fragment far from
    the real signature (e.g. bleed-through from an adjacent sample) must
    be excluded even though it clears the raw area threshold, while a
    real detached letter-group proportionate to the signature's own size
    must still be kept unconditionally."""
    h, w = 300, 700
    binary = np.zeros((h, w), np.uint8)
    cv2.ellipse(binary, (300, 150), (120, 60), 0, 0, 360, 255, 8)   # anchor "body"
    cv2.circle(binary, (520, 140), 14, 255, -1)                     # legit nearby detached mark (small, close)
    cv2.rectangle(binary, (10, 290), (34, 299), 255, -1)            # tiny far fragment (bottom-left corner)
    cleaned = pp._remove_distant_outliers(binary)
    far_fragment_gone = cleaned[290:300, 10:34].sum() == 0
    nearby_mark_kept = cleaned[126:154, 506:534].sum() > 0
    check("distant_outlier_rejects_tiny_far_fragment", far_fragment_gone)
    check("distant_outlier_keeps_nearby_detached_mark", nearby_mark_kept)


def test_edge_case_review_opposing_evidence_fires_at_n2():
    """Bug #59 fix #1: with exactly 2 references split 50/50 across the
    threshold, the opposing-evidence check must fire (previously gated at
    n>=3, silently never running for 2-reference profiles)."""
    eng = VerificationEngine()
    result = eng._edge_case_review("GENUINE", threshold=0.5, fused_score=0.6,
                                   per_ref_scores=[0.75, 0.30], per_ref_best_iou=[0.4, 0.4])
    check("edge_case_opposing_evidence_fires_at_n2", result["triggered"] is True,
         str(result))


def test_edge_case_review_n1_alignment_not_vacuously_waived():
    """Bug #59 fix #2: with a single reference and a catastrophic alignment
    failure (IoU far below the sanity floor), the check must fire — a lone
    verdict trivially 'agreeing with itself' must not waive this."""
    eng = VerificationEngine()
    result = eng._edge_case_review("GENUINE", threshold=0.5, fused_score=0.6,
                                   per_ref_scores=[0.6], per_ref_best_iou=[0.02])
    check("edge_case_n1_alignment_not_vacuously_waived", result["triggered"] is True,
         str(result))


def test_edge_case_review_bug51_regression_still_holds():
    """Bug #51 regression guard: n=2, both references UNANIMOUSLY agree
    FORGED despite poor alignment overlap (a real difference, not a
    registration catastrophe) — must stay a clean, un-downgraded FORGED."""
    eng = VerificationEngine()
    result = eng._edge_case_review("FORGED", threshold=0.5, fused_score=0.2,
                                   per_ref_scores=[0.15, 0.22], per_ref_best_iou=[0.03, 0.05])
    check("edge_case_bug51_unanimous_agreement_still_waived", result["triggered"] is False,
         str(result))


def test_edge_case_review_diagnostics_always_present():
    """Bug #59 fix #4: diagnostics (reference_agreement, alignment_reliability,
    opposing_evidence_fraction) must be populated even when the verdict is
    already INCONCLUSIVE from the primary band — previously discarded."""
    eng = VerificationEngine()
    result = eng._edge_case_review("INCONCLUSIVE", threshold=0.5, fused_score=0.51,
                                   per_ref_scores=[0.55, 0.47], per_ref_best_iou=[0.3, 0.3])
    check("edge_case_diagnostics_present_when_already_inconclusive",
         result["reference_agreement"] is not None and result["alignment_reliability"] is not None,
         str(result))


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as exc:                                    # noqa: BLE001
            check(t.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print(f"All {len(tests)} regression tests passed.")


if __name__ == "__main__":
    _run_all()
