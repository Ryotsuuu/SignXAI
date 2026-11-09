"""Spatial alignment of a query signature onto a reference.

v5 CHANGE (Bug #20, PROJECT_MEMORY_v8.md): the v4 aligner trusted ORB+ECC
unconditionally -- if ORB found >=8 keypoint matches, the resulting affine
was used as-is, with no check that it actually made the two signatures
overlap better than doing nothing. ORB is a corner/blob detector built for
richly-textured natural images; a pen-stroke signature on white paper has
almost none of that texture, so the handful of "keypoints" ORB does find
are usually just points along stroke edges -- weak, ambiguous correspondences.
With RANSAC fitting a full affine (rotation + scale + shear + translation)
to 8-ish such points, it can converge on a plausible-looking but SPURIOUS
transform, including large rotations that have nothing to do with the
actual signatures. That is exactly what the user saw: a visibly rotated
query that no longer overlapped its reference in the heatmap/overlay, with
no error anywhere in the pipeline -- the alignment "succeeded" by the old
definition (found enough matches), it just produced a bad transform.

Fix, three parts, each independently defensible:

1. Several candidate transforms are generated, not just ORB+ECC:
   (a) identity (no transform at all);
   (b) centroid-translation-only (just re-center the ink masses --
       correct for the very common case where nothing but position
       differs);
   (c) an image-moments / PCA-based similarity transform: rotate the
       query's principal ink axis onto the reference's, translate
       centroids to match. Robust for sparse ink where ORB keypoints
       are unreliable, because it uses EVERY ink pixel, not a handful
       of ambiguous corners;
   (d) the v4 ORB-affine, refined by ECC.
   Both (c) and (d)'s rotation component is clipped to +/-25 degrees
   (`_MAX_ROTATION_DEG`) -- signatures are captured near-upright by
   design (see preprocess.py's EXIF-orientation fix), so an implied
   rotation past that is far more likely a bad fit than a real photo
   angle, and clipping it can only pull a spurious transform back
   toward identity, never create a worse one.

2. Every candidate is actually SCORED (ink IoU of the warped query
   against the reference) and the highest-scoring one wins -- never
   assumed. This makes alignment monotonic: the chosen transform is
   structurally guaranteed to match-or-beat doing nothing, which is
   the concrete guarantee "images get rotated and not able to properly
   overlap" was missing.

3. ``aligned_ok`` now means "the winning candidate is non-identity AND
   improved ink IoU over identity by a real margin" -- not merely "ORB
   found >=8 points", which said nothing about whether the resulting
   transform actually helped.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import CANVAS_W, CANVAS_H
from .preprocess import Preprocessed

_MAX_ROTATION_DEG = 25.0   # see module docstring, part 1
_MIN_IOU_IMPROVEMENT = 0.01  # candidate must beat identity by more than this
                              # to be considered a genuine, useful alignment


# ------------------------------------------------------------------ scoring
def _ink_iou(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    a, b = bin_a > 0, bin_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _warp_pair(gray: np.ndarray, binary: np.ndarray, warp: np.ndarray):
    w_gray = cv2.warpAffine(gray, warp, (CANVAS_W, CANVAS_H),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderValue=255)
    w_bin = cv2.warpAffine(binary, warp, (CANVAS_W, CANVAS_H),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderValue=0)
    return w_gray, w_bin


# --------------------------------------------------------------- transforms
def _rotation_deg(warp: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))


def _clip_rotation(warp: np.ndarray, max_deg: float = _MAX_ROTATION_DEG) -> np.ndarray:
    """Clip a 2x3 affine's implied rotation to +/-max_deg, dropping any
    shear beyond what that clipped rotation implies. A "confident" large
    shear/rotation from a handful of noisy keypoints is exactly the
    failure mode being guarded against here, not a real transform worth
    preserving."""
    angle = _rotation_deg(warp)
    scale = float(np.hypot(warp[0, 0], warp[1, 0])) or 1.0
    scale = min(max(scale, 0.85), 1.15)   # also guard against runaway scale
    if abs(angle) <= max_deg and 0.85 <= scale <= 1.15:
        return warp
    clipped = max(-max_deg, min(max_deg, angle))
    theta = np.radians(clipped)
    r = scale * np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    out = warp.copy()
    out[:2, :2] = r
    return out


def _orb_affine(q_gray: np.ndarray, r_gray: np.ndarray) -> np.ndarray | None:
    """2x3 affine from ORB keypoints; ``None`` when matches are too weak
    to trust at all (caller falls back to other candidates)."""
    orb = cv2.ORB_create(nfeatures=800)
    kp1, d1 = orb.detectAndCompute(q_gray, None)
    kp2, d2 = orb.detectAndCompute(r_gray, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)
    good = matches[: max(8, int(len(matches) * 0.3))]
    if len(good) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    mat, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                               ransacReprojThreshold=4.0)
    if mat is None or (inliers is not None and inliers.sum() < 6):
        return None
    return _clip_rotation(mat.astype(np.float32))


def _centroid_translation(q_bin: np.ndarray, r_bin: np.ndarray) -> np.ndarray:
    """Pure re-centring of ink masses -- correct whenever position is the
    only real difference (the common case for already-cropped, canvas-
    normalised signatures)."""
    warp = np.eye(2, 3, dtype=np.float32)
    qys, qxs = np.where(q_bin > 0)
    rys, rxs = np.where(r_bin > 0)
    if len(qxs) == 0 or len(rxs) == 0:
        return warp
    qc = np.array([qxs.mean(), qys.mean()], dtype=np.float32)
    rc = np.array([rxs.mean(), rys.mean()], dtype=np.float32)
    # cv2.warpAffine below uses WARP_INVERSE_MAP: `warp` maps REFERENCE
    # coords -> QUERY coords when that flag is set, so translation must be
    # (query_centroid - reference_centroid).
    warp[0, 2] = qc[0] - rc[0]
    warp[1, 2] = qc[1] - rc[1]
    return warp


def _pca_similarity(q_bin: np.ndarray, r_bin: np.ndarray) -> np.ndarray | None:
    """Centroid + principal-ink-axis rotation via image moments. Uses
    every ink pixel (not a handful of ambiguous ORB corners), so it stays
    reliable even on sparse, low-texture pen strokes. Rotation clipped
    like every other candidate (see module docstring)."""
    def _axis(binary):
        m = cv2.moments((binary > 0).astype(np.uint8), binaryImage=True)
        if m["m00"] < 1:
            return None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        return np.array([cx, cy], dtype=np.float32), angle

    q, r = _axis(q_bin), _axis(r_bin)
    if q is None or r is None:
        return None
    (qc, qang), (rc, rang) = q, r
    dtheta = np.degrees(qang - rang)
    dtheta = ((dtheta + 90) % 180) - 90          # stroke axis is 180-symmetric
    dtheta = max(-_MAX_ROTATION_DEG, min(_MAX_ROTATION_DEG, dtheta))
    theta = np.radians(dtheta)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    warp[:2, :2] = rot
    warp[:, 2] = qc - rot @ rc
    return warp


def _ecc_refine(q_gray: np.ndarray, r_gray: np.ndarray, warp: np.ndarray) -> np.ndarray:
    try:
        qf = q_gray.astype(np.float32) / 255.0
        rf = r_gray.astype(np.float32) / 255.0
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        _, refined = cv2.findTransformECC(rf, qf, warp.copy(), cv2.MOTION_AFFINE,
                                          criteria, None, 3)
        return _clip_rotation(refined.astype(np.float32))
    except cv2.error:
        return warp


# ----------------------------------------------------------------- public
def align(query: Preprocessed, reference: Preprocessed):
    """Align query onto reference by trying several transform families and
    keeping whichever one actually improves ink overlap (see module
    docstring). Returns ``(warped_gray, warped_binary, aligned_ok, best_iou)``.

    v7 (Bug #36, PROJECT_MEMORY_v8.md, self-run): now also returns the
    actual achieved ink-IoU of the winning candidate (``best_iou``), not
    just the boolean ``aligned_ok``. ``aligned_ok`` only ever meant "a
    non-identity transform beat identity" -- true in the ordinary case
    where the query needed some rotation/translation to line up, but
    ALSO false in the *best* case (query already near-perfectly
    registered, identity wins because nothing can beat it) as well as
    the worst case (nothing lines up at all). Callers that need to know
    whether the resulting overlap is actually trustworthy for
    pixel-level comparison should threshold ``best_iou`` directly instead
    of relying on ``aligned_ok`` for that purpose -- see
    `engine.py::_score_against_ref`'s use of it.
    """
    identity = np.eye(2, 3, dtype=np.float32)
    candidates: list[tuple[str, np.ndarray]] = [("identity", identity)]

    centroid = _centroid_translation(query.binary, reference.binary)
    candidates.append(("centroid", centroid))
    candidates.append(("centroid_ecc",
                       _ecc_refine(query.gray, reference.gray, centroid)))

    pca = _pca_similarity(query.binary, reference.binary)
    if pca is not None:
        candidates.append(("pca", pca))

    orb = _orb_affine(query.gray, reference.gray)
    if orb is not None:
        candidates.append(("orb_ecc", _ecc_refine(query.gray, reference.gray, orb)))

    scored = []
    for name, warp in candidates:
        _, w_bin = _warp_pair(query.gray, query.binary, warp)
        scored.append((name, warp, _ink_iou(w_bin, reference.binary)))

    identity_iou = scored[0][2]
    best_name, best_warp, best_iou = max(scored, key=lambda t: t[2])

    w_gray, w_bin = _warp_pair(query.gray, query.binary, best_warp)
    aligned_ok = (best_name != "identity") and (best_iou > identity_iou + _MIN_IOU_IMPROVEMENT)
    return w_gray, w_bin, aligned_ok, best_iou
