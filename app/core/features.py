"""Forensic feature extraction.

Produces two families of features from a :class:`Preprocessed` signature:

* **scalars** — interpretable forensic measurements (geometry, stroke,
  topology, shape moments). Each maps to a label + category used in the
  metrics table and the forensic report.
* **vectors** — high-dimensional descriptors (HOG, LBP) compared with
  cosine similarity.

All values are guaranteed finite; degenerate inputs yield zeros rather
than NaNs so downstream statistics never break.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize

from .preprocess import Preprocessed

# name -> (human label, forensic category)
SCALAR_CATALOG: dict[str, tuple[str, str, str]] = {
    # key: (display label, category, plain-language explanation)
    "aspect_ratio": ("Aspect ratio (w/h)", "Geometry",
                     "Width divided by height of the signature's ink bounding box. "
                     "People tend to keep this ratio fairly consistent — a signature "
                     "squeezed much narrower or stretched much wider than usual can "
                     "indicate a different hand, a rushed forgery attempt, or an "
                     "unusually cropped/rotated photo."),
    "ink_ratio": ("Ink density ratio", "Geometry",
                 "Fraction of the bounding box actually covered by ink. Forgers "
                 "tracing a signature slowly often press differently, leaving "
                 "noticeably thicker or thinner ink coverage than the genuine writer's "
                 "natural, fluid strokes."),
    "centroid_x": ("Ink centroid X", "Geometry",
                   "Horizontal center of mass of the ink. Shifts can indicate a "
                   "different overall composition/balance to the signature."),
    "centroid_y": ("Ink centroid Y", "Geometry",
                   "Vertical center of mass of the ink. Shifts can indicate a "
                   "different baseline or vertical balance than the writer's norm."),
    "pca_orientation_deg": ("Overall slant angle", "Geometry",
                            "The dominant angle the signature leans at. Genuine "
                            "signatures from one person have a fairly stable natural "
                            "slant; a meaningfully different angle is a classic "
                            "forgery indicator since it's hard for a forger to "
                            "consciously match someone else's exact slant while also "
                            "copying letterforms."),
    "stroke_width_mean": ("Mean stroke width", "Stroke",
                          "Average ink stroke thickness. Reflects pen type, pressure, "
                          "and writing speed — a proxy for pen pressure since this "
                          "system works from static images without real pressure "
                          "sensors."),
    "stroke_width_std": ("Stroke width variation", "Stroke",
                         "How much stroke thickness varies within the signature. "
                         "Natural handwriting has organic, rhythmic variation; slow, "
                         "traced forgeries often look more mechanically uniform (low "
                         "variation) or erratic in a different pattern than the "
                         "genuine writer's own rhythm."),
    "stroke_width_cv": ("Pressure proxy (width CV)", "Stroke",
                        "Coefficient of variation of stroke width (std/mean) — "
                        "normalises the variation above by the average thickness, so "
                        "it's comparable across pens of different tip sizes."),
    "skeleton_length": ("Total stroke length", "Stroke",
                        "Total length of the 1px-wide stroke centerline. Roughly "
                        "tracks how much pen travel the signature took — a "
                        "much shorter or longer path than usual can indicate missing "
                        "or added flourishes/strokes."),
    "endpoints_count": ("Stroke endpoints", "Topology",
                        "Number of places where a stroke starts or ends (pen-lift "
                        "points). Directly reflects how many times the pen left the "
                        "paper — a very different count suggests a different stroke "
                        "sequence, which is hard for a forger to get exactly right "
                        "since they can't see how the original was actually written."),
    "junctions_count": ("Stroke junctions", "Topology",
                        "Number of places where strokes cross or meet. Reflects the "
                        "signature's structural complexity and connectivity pattern."),
    "components_count": ("Connected components", "Topology",
                         "Number of separate, disconnected ink shapes (e.g. a "
                         "signature plus a separate underline counts as 2). A "
                         "different count often means a missing/added flourish, "
                         "underline, or dot."),
    "holes_count": ("Enclosed loops", "Topology",
                    "Number of fully enclosed loops the ink forms (e.g. inside "
                    "letters like 'o', 'a', or decorative loops). NOTE: for "
                    "profiles with very few enrolled samples, this integer count can "
                    "be identical across all references (std=0), which used to cause "
                    "a scoring bug — see PROJECT_MEMORY_v8.md Bug #6 for the fix."),
    "contour_perimeter": ("Contour perimeter", "Shape",
                          "Total outline length of the ink shape. A broad, easily "
                          "computed proxy for the signature's overall complexity/size."),
    "contours_count": ("Outer contours", "Shape",
                       "Number of separate outer boundary shapes."),
    **{f"hu_{i}": (f"Hu moment {i}", "Shape",
                   "One of 7 mathematical shape descriptors (Hu moments) that are "
                   "invariant to translation, scale, and rotation — they capture the "
                   "signature's overall silhouette shape independent of exactly "
                   "where/how big/how rotated it was drawn. Hard for a forger to "
                   "consciously replicate since they're not directly visible or "
                   "intuitive quantities.")
      for i in range(1, 8)},
}
# Backward-compat 2-tuple view for any code that only needs (label, category).
SCALAR_CATALOG_SHORT: dict[str, tuple[str, str]] = {
    k: (v[0], v[1]) for k, v in SCALAR_CATALOG.items()
}

VECTOR_CATALOG: dict[str, tuple[str, str]] = {
    "hog": ("Gradient orientation descriptor (HOG)", "Shape"),
    "lbp": ("Local texture descriptor (LBP)", "Texture"),
}


def _safe(v: float) -> float:
    v = float(v)
    return v if np.isfinite(v) else 0.0


def extract(pre: Preprocessed) -> dict:
    """Extract all forensic features. Returns {"scalars": {...}, "vectors": {...}}."""
    ink = (pre.binary > 0).astype(np.uint8)
    scalars: dict[str, float] = {}

    ys, xs = np.where(ink > 0)
    n_ink = max(len(xs), 1)
    if len(xs) > 1:
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        scalars["aspect_ratio"] = w / max(h, 1)
        scalars["centroid_x"] = xs.mean() / ink.shape[1]
        scalars["centroid_y"] = ys.mean() / ink.shape[0]
        pts = np.column_stack([xs, ys]).astype(np.float64)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, int(np.argmax(eigvals))]
        scalars["pca_orientation_deg"] = float(np.degrees(np.arctan2(v[1], v[0])))
    else:
        scalars.update(aspect_ratio=1.0, centroid_x=0.5, centroid_y=0.5,
                       pca_orientation_deg=0.0)
    scalars["ink_ratio"] = n_ink / ink.size

    # Stroke width via distance transform sampled on the skeleton
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    skel_mask = pre.skeleton > 0
    widths = dist[skel_mask] * 2.0
    if widths.size:
        scalars["stroke_width_mean"] = widths.mean()
        scalars["stroke_width_std"] = widths.std()
        scalars["stroke_width_cv"] = widths.std() / max(widths.mean(), 1e-6)
    else:
        scalars.update(stroke_width_mean=0.0, stroke_width_std=0.0, stroke_width_cv=0.0)
    scalars["skeleton_length"] = float(skel_mask.sum())

    # Topology: endpoints / junctions / components / holes.
    #
    # v6 (Bug #33, PROJECT_MEMORY_v8.md): these counts are computed from a
    # morphologically-CLOSED copy of the ink mask, not the raw one used
    # everywhere else. Real-world PHOTOGRAPHED signatures (phone camera,
    # variable lighting/exposure) can have a stroke that's fractionally
    # thinner in one shot than another purely from lighting — enough to
    # drop a couple of pixels below the Otsu threshold and open a tiny gap
    # in what is, physically, one continuous stroke or one fully-closed
    # loop. Because these features are DISCRETE COUNTS (this is exactly
    # why they were split into their own "construction" fusion component
    # in the first place — see similarity.py's CONSTRUCTION_KEYS
    # docstring), a single such gap doesn't just nudge a continuous
    # measurement, it flips a whole count by one integer: one connected
    # component becomes two, one enclosed loop becomes zero. Confirmed
    # directly: re-verifying an ENROLLED reference against its own
    # profile (a case that should be as close to "certainly genuine" as
    # this system can produce) showed unstable construction-count scores
    # across real phone-photographed samples of the same physical
    # signature, exactly consistent with this mechanism (see
    # docs/BUILD_LOG_v6.md for the numbers). A small morphological CLOSE
    # (dilate then erode, 3x3) bridges gaps at this scale while leaving a
    # REAL topological difference — a genuinely missing loop, a
    # genuinely different number of pen-lifts, differences the size of an
    # actual stroke, not a lighting artifact — completely unaffected,
    # since those are far larger than a 3x3 closing kernel can bridge.
    topo_ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    topo_skel = (skeletonize(topo_ink > 0).astype(np.uint8)) * 255
    sk = (topo_skel > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT) - sk
    scalars["endpoints_count"] = float(np.logical_and(sk == 1, neighbors == 1).sum())
    scalars["junctions_count"] = float(np.logical_and(sk == 1, neighbors >= 3).sum())

    n_comp, _ = cv2.connectedComponents(topo_ink, connectivity=8)
    scalars["components_count"] = float(n_comp - 1)
    contours, hierarchy = cv2.findContours(topo_ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer = [c for i, c in enumerate(contours)
             if hierarchy is None or hierarchy[0][i][3] == -1]
    scalars["contours_count"] = float(len(outer))
    scalars["contour_perimeter"] = float(sum(cv2.arcLength(c, True) for c in outer))
    n_holes = sum(1 for i in range(len(contours))
                  if hierarchy is not None and hierarchy[0][i][3] != -1)
    scalars["holes_count"] = float(n_holes)

    # Shape: Hu moments on the ink mask (log-scaled, sign preserved)
    moments = cv2.moments(ink, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    for i, val in enumerate(hu, start=1):
        scalars[f"hu_{i}"] = -np.sign(val) * np.log10(abs(val) + 1e-30)

    # Vectors: HOG on normalised grayscale + LBP histogram
    hog_vec = hog(pre.gray, orientations=9, pixels_per_cell=(32, 32),
                  cells_per_block=(2, 2), block_norm="L2-Hys",
                  feature_vector=True)
    lbp = local_binary_pattern(pre.gray, P=8, R=1.0, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    scalars = {k: _safe(v) for k, v in scalars.items()}
    vectors = {
        "hog": np.asarray(hog_vec, dtype=np.float64).tolist(),
        "lbp": np.asarray(lbp_hist, dtype=np.float64).tolist(),
    }
    return {"scalars": scalars, "vectors": vectors}
