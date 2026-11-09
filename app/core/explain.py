"""Explainability (XAI) image generation.

For every verification the engine saves a set of visual evidence images:

* ``heatmap.png``      — where the aligned query diverges from the reference
* ``overlay.png``      — green = matching ink, red = query-only, blue = ref-only
* ``skeleton.png``     — stroke centreline comparison
* ``annotated.png``    — top divergent regions boxed + labelled (engine)
* ``annotated_llm.png``— regions reported by the vision LLM (when available)

Design decision: the LLM never edits pixels itself — it reports normalised
region boxes and this module draws them, keeping full visual consistency.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import CANVAS_W, CANVAS_H


def _save(img: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path.name


def heatmap_overlay(query_gray: np.ndarray, ref_gray: np.ndarray,
                    out_dir: Path, ref_label: str | None = None) -> tuple[str, np.ndarray]:
    """Intensity-modulated divergence heatmap of |query - reference|.

    v6 FIX (Bug #27, PROJECT_MEMORY_v8.md, user-reported: "there is not
    heatmap present"): the previous version blended a JET-colormapped
    difference image over the WHOLE canvas at a flat 45% opacity,
    including the large blank white background that dominates most
    signature photos. JET's low end is a saturated dark blue, not white
    or transparent, so blank background (near-zero difference almost
    everywhere) rendered as a solid blue wash across the entire image --
    visually reading as "no heatmap", or a flat tinted rectangle, not a
    real heatmap, exactly what was reported. A real heatmap (the user's
    own pasted definition: "shows the magnitude of a phenomenon as
    color... giving obvious visual cues about how it's clustered or
    varies") needs near-zero regions to stay visually neutral so only
    genuine divergence pops out.

    v9 FIX (Bug #57, PROJECT_MEMORY_v9.md, user-reported: "it does not
    show which image is get based match [name is missing]"): the heatmap
    alone never named WHICH of the (possibly many) enrolled references it
    was actually diffing against, so a human reviewer had no way to check
    the comparison was reasonable without cross-referencing the JSON
    response. ``ref_label`` (e.g. "Reference sample 3 of 5"), when given,
    is now burned directly into the image as a caption strip, matching
    how ``reference_agreement_chart`` already names every reference.

    Fix, two parts:
    1. Contrast-stretch the blurred difference map to its own observed
       range before colormapping, so the colormap's full range is actually
       used by this image's real divergence values (a typically-small
       raw pixel difference would otherwise only ever reach the low end
       of any colormap, muting every case to near-uniform color).
    2. Alpha-modulate the blend by divergence magnitude (gamma-shaped, so
       low-end noise is suppressed further) instead of a flat blend ratio
       -- near-zero-divergence pixels stay essentially the clean original
       image, and only genuinely divergent regions are colorized,
       increasingly opaquely as divergence increases. This is what makes
       it read as an actual heatmap: color clusters exactly where the two
       signatures actually differ, nowhere else.
    """
    diff = cv2.absdiff(query_gray, ref_gray).astype(np.float32) / 255.0
    diff = cv2.GaussianBlur(diff, (21, 21), 0)
    d = diff / (float(diff.max()) + 1e-6)
    d = np.clip(d, 0.0, 1.0)
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(query_gray, cv2.COLOR_GRAY2BGR)
    alpha = np.power(d, 1.4)[..., None]
    blended = (base.astype(np.float32) * (1 - alpha)
              + heat.astype(np.float32) * alpha).astype(np.uint8)
    if ref_label:
        blended = _caption_strip(blended, f"Compared against: {ref_label}")
    return _save(blended, out_dir / "heatmap.png"), diff


def _caption_strip(img: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    """Prepend a small dark caption bar with white text above an image --
    shared by every v9 XAI view that needs to name what it's showing
    (which reference, which panel) directly on the saved PNG itself, since
    these images are also downloaded/printed standalone in report.pdf
    where accompanying JSON context is not visible alongside them
    (Bug #57/#58, PROJECT_MEMORY_v9.md)."""
    w = img.shape[1]
    bar = np.full((height, w, 3), (40, 40, 40), np.uint8)
    cv2.putText(bar, text, (6, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
               (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def side_by_side_panels(panels: list[tuple[np.ndarray, str]], out_path: Path,
                        gap: int = 6) -> str:
    """Stitch N same-size (or auto-padded) BGR/gray panels horizontally,
    each with its own caption strip -- the shared building block behind
    every v9 "show me query / reference / result together" request
    (Bug #57 and #58, PROJECT_MEMORY_v9.md): the heatmap now ships beside
    the query and matched reference it was computed from, and the ink
    agreement map ships beside the two raw inputs it was computed from,
    instead of each XAI image appearing in isolation with the viewer left
    to mentally reconstruct what it's being compared against.
    """
    prepped = []
    max_h = 0
    for img, label in panels:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = _caption_strip(img, label)
        prepped.append(img)
        max_h = max(max_h, img.shape[0])
    padded = []
    for img in prepped:
        if img.shape[0] < max_h:
            pad = np.full((max_h - img.shape[0], img.shape[1], 3), 255, np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    gap_col = np.full((max_h, gap, 3), 230, np.uint8)
    row = padded[0]
    for img in padded[1:]:
        row = np.hstack([row, gap_col, img])
    return _save(row, out_path)


def alignment_overlay(query_bin: np.ndarray, ref_bin: np.ndarray,
                      out_dir: Path) -> str:
    """Three-colour ink agreement map."""
    q = query_bin > 0
    r = ref_bin > 0
    img = np.full((*q.shape, 3), 255, np.uint8)
    img[q & r] = (60, 170, 60)      # match — green (BGR)
    img[q & ~r] = (60, 60, 220)     # query-only — red
    img[~q & r] = (220, 130, 40)    # reference-only — blue
    return _save(img, out_dir / "overlay.png")


def ink_agreement_strip(query_gray: np.ndarray, ref_gray: np.ndarray,
                        query_bin: np.ndarray, ref_bin: np.ndarray,
                        out_dir: Path, ref_label: str = "reference",
                        ink_iou: float | None = None) -> str:
    """Query | Reference | Ink-agreement, side by side with a plain-English
    caption pointing at the biggest disagreement region.

    v9 FIX (Bug #58, PROJECT_MEMORY_v9.md, user-reported: the ink
    agreement view had "the same dis-oriented and oversizing Question
    image" problem and the user asked to "show three things, side by side
    comparing and down below it... point out what it gets wrong"). Two
    distinct problems were folded into this one complaint, addressed
    separately:

    1. The "dis-oriented and oversizing" look was, on inspection, almost
       entirely a symptom of Bug #55's preprocessing contamination (a
       shadow/fabric/bleed-through fragment corrupting the crop bounding
       box upstream) rather than anything wrong in THIS rendering step --
       fixing the crop at the source fixes how every downstream view
       looks, this one included. See ``preprocess._remove_chromatic_outliers``.
    2. The actual, remaining request -- seeing the raw query and raw
       reference next to the agreement map, not just the map alone -- is
       a genuine gap this function closes directly: it assembles all
       three panels in one image plus a caption line naming the single
       largest contiguous disagreement region by its canvas position
       (e.g. "largest mismatch: lower-right area"), so a reviewer doesn't
       have to mentally align three separate report images by eye.
    """
    overlay_name = alignment_overlay(query_bin, ref_bin, out_dir)
    overlay_img = cv2.imread(str(out_dir / overlay_name))

    mismatch = (query_bin > 0) ^ (ref_bin > 0)
    caption = f"Ink agreement vs. {ref_label}"
    if ink_iou is not None:
        caption += f" (ink overlap {ink_iou * 100:.0f}%)"
    if mismatch.any():
        ys, xs = np.where(mismatch)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        h, w = mismatch.shape
        vert = "upper" if cy < h * 0.4 else ("lower" if cy > h * 0.6 else "middle")
        horiz = "left" if cx < w * 0.4 else ("right" if cx > w * 0.6 else "center")
        caption += f" — largest mismatch concentration: {vert}-{horiz} area"

    strip = side_by_side_panels(
        [(query_gray, "Query (as processed)"),
         (ref_gray, f"Reference: {ref_label}"),
         (overlay_img, "green=match  red=query-only  blue=ref-only")],
        out_dir / "agreement_strip.png")
    # Burn the pointed-at-mismatch caption across the full composite too,
    # so it survives being viewed/printed as a single standalone image.
    img = cv2.imread(str(out_dir / strip))
    img = _caption_strip(img, caption, height=24)
    return _save(img, out_dir / "agreement_strip.png")


def skeleton_compare(query_skel: np.ndarray, ref_skel: np.ndarray,
                     out_dir: Path) -> str:
    img = np.full((CANVAS_H, CANVAS_W, 3), 255, np.uint8)
    img[ref_skel > 0] = (180, 130, 30)    # reference skeleton — amber/blue
    img[query_skel > 0] = (50, 50, 210)   # query skeleton — red
    both = (query_skel > 0) & (ref_skel > 0)
    img[both] = (60, 170, 60)             # overlap — green
    return _save(img, out_dir / "skeleton.png")


def annotated_regions(query_gray: np.ndarray, diff: np.ndarray,
                      top_deviants: list[str], out_dir: Path,
                      k: int = 3) -> tuple[str, list[dict]]:
    """Box the actual divergence hot-spots in the image, not a rigid grid.

    v6 FIX (Bug #27, PROJECT_MEMORY_v8.md, user-reported: "those rectangle
    detector are too generic and choppy... it don't get collide with other
    metric data"): the previous version carved the canvas into a FIXED
    4x8 grid and boxed whichever cells happened to score highest, with no
    relationship to the actual shape of any real divergence in the image
    -- boxes routinely cut a single stroke in half, or landed on a mostly-
    blank grid cell adjacent to the real difference, and every label was
    placed at the same fixed offset from its own box with no collision
    checking against other boxes' labels, so labels on nearby/overlapping
    top-ranked cells (common, since a rigid grid frequently ranks several
    adjacent cells similarly) visibly overlapped each other -- exactly
    what the user's own generated report showed ("R1: Aspect Ru moment 1"
    running into "R2 Hu moment 1").

    Fix: find the actual divergence blobs via thresholding + contour
    detection on the (already Gaussian-blurred) difference map, and box
    their real bounding rectangles -- content-adaptive, not grid-locked.
    Falls back to the old grid method only in the degenerate case where
    no clear blob exists (e.g. near-zero divergence everywhere). Label
    placement is delegated entirely to ``annotate_regions``, which already
    implements bounded collision-avoidance (Bug #11, PROJECT_MEMORY_v8.md)
    -- one label-placement implementation, reused, instead of a second,
    weaker one duplicated here.
    """
    d = diff / (float(diff.max()) + 1e-6)
    mask = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []   # (energy, x0, y0, x1, y1)
    min_area = (CANVAS_W * CANVAS_H) * 0.0015   # ignore specks below ~0.15% of canvas
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        energy = float(d[y:y + h, x:x + w].mean())
        boxes.append((energy, x, y, x + w, y + h))
    boxes.sort(reverse=True, key=lambda b: b[0])

    def _iou(a, b):
        ax0, ay0, ax1, ay1 = a[1:]
        bx0, by0, bx1, by1 = b[1:]
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        return inter / min(area_a, area_b)   # overlap relative to the smaller box

    # v6 fix (Bug #27, PROJECT_MEMORY_v8.md): drop boxes that substantially
    # overlap one already kept, highest-energy first (simple greedy NMS) --
    # otherwise a single real divergence blob that the morphological close
    # step didn't fully merge can be picked as 2-3 separate "top regions",
    # producing nested/overlapping boxes and colliding labels.
    picked: list[tuple[float, int, int, int, int]] = []
    for b in boxes:
        if all(_iou(b, p) < 0.2 for p in picked):
            picked.append(b)
    boxes = picked

    if not boxes:
        # Degenerate fallback: no clear divergence blob at all (e.g. a
        # near-perfect match) -- fall back to the coarse grid so there's
        # still SOMETHING to show, rather than an empty result.
        rows, cols = 4, 8
        ch, cw = CANVAS_H // rows, CANVAS_W // cols
        grid = []
        for gy in range(rows):
            for gx in range(cols):
                cell = diff[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
                grid.append((float(cell.mean()), gx * cw, gy * ch, (gx + 1) * cw, (gy + 1) * ch))
        grid.sort(reverse=True, key=lambda b: b[0])
        boxes = grid[:k]

    labels = top_deviants or ["stroke pattern"]
    findings = []
    for rank, (e, x0, y0, x1, y1) in enumerate(boxes[:k], start=1):
        label = labels[(rank - 1) % len(labels)]
        findings.append({
            "region": [round(x0 / CANVAS_W * 1000), round(y0 / CANVAS_H * 1000),
                       round(x1 / CANVAS_W * 1000), round(y1 / CANVAS_H * 1000)],
            "title": f"Region {rank} divergence: {label}",
            "detail": f"High pixel-level divergence (energy {e:.3f}); most deviant "
                      f"feature group: {label}.",
            "severity": "high" if rank == 1 else "medium",
        })
    name = annotate_regions(query_gray, findings, out_dir, name="annotated.png")
    return name, findings




def zone_grid_overlay(query_gray: np.ndarray, zonal_result: dict, out_dir: Path) -> str:
    """v6 (Bug #31, PROJECT_MEMORY_v8.md): visualize the content-adaptive
    zone grid (zoning.py) directly on the query image -- every base zone
    boundary drawn, color-coded by its local agreement score (green =
    high local agreement, amber = borderline, red = flagged structural
    mismatch), with each flagged zone's corner-count comparison labeled.
    This is the direct visual answer to "which specific region differs
    and how" that a single global heatmap cannot give on its own — the
    zone grid literally shows where the signature was subdivided and
    what was found in each piece, addressing the user's explicit request
    to see the image "divided into sections" with each section's local
    match reported.
    """
    img = cv2.cvtColor(query_gray, cv2.COLOR_GRAY2BGR)
    flagged_keys = {(z["row"], z["col"]) for z in zonal_result["flagged_zones"]}
    for z in zonal_result["zones"]:
        x0, y0, x1, y1 = z["x0"], z["y0"], z["x1"], z["y1"]
        key = (z["row"], z["col"])
        if key in flagged_keys:
            color = (40, 40, 230)     # red (BGR) -- flagged structural mismatch
        elif z["zone_score"] < 0.85:
            color = (0, 165, 255)     # amber -- borderline, not flagged
        else:
            color = (90, 170, 90)     # green -- high local agreement
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        if key in flagged_keys:
            label = f"{z['query_corners']}/{z['ref_corners']}"
            cv2.putText(img, label, (x0 + 3, min(y0 + 14, CANVAS_H - 4)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return _save(img, out_dir / "zone_grid.png")


def reference_agreement_chart(ref_names: list[str], ref_scores: list[float],
                              prototype_score: float, threshold: float,
                              out_dir: Path) -> str:
    """v5 (Bug #21, PROJECT_MEMORY_v8.md): bar chart of the query's fused score
    against EACH enrolled reference individually, plus the aggregate
    prototype score, against the decision threshold. This is the visual
    counterpart of ``case["reference_scores"]``/``case["aggregate"]`` in
    the API response -- direct evidence that the verdict came from
    comparing the questioned signature against every sample image, not
    just one, addressing the "only feedback for sample images, not the
    query itself" complaint at its root: before this, no single visual
    artifact showed the query being compared to more than one reference
    at a time.
    """
    n = len(ref_scores)
    w, h = CANVAS_W, max(220, 40 * (n + 1) + 60)
    img = np.full((h, w, 3), 255, np.uint8)
    margin_l, margin_r = 150, 40
    plot_w = w - margin_l - margin_r
    lo, hi = 0.0, 1.0

    def x_of(v: float) -> int:
        return margin_l + int(np.clip((v - lo) / (hi - lo), 0, 1) * plot_w)

    thr_x = x_of(threshold)
    band_lo_x = x_of(max(lo, threshold - 0.07))
    band_hi_x = x_of(min(hi, threshold + 0.07))
    cv2.rectangle(img, (band_lo_x, 10), (band_hi_x, h - 30), (230, 230, 230), -1)
    cv2.line(img, (thr_x, 10), (thr_x, h - 30), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "threshold", (min(thr_x + 4, w - 90), 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)

    rows = [(name, s, False) for name, s in zip(ref_names, ref_scores)]
    rows.append(("PROTOTYPE (all samples)", prototype_score, True))
    bar_h = 20
    for i, (name, s, is_proto) in enumerate(rows):
        y = 34 + i * 40
        label = name if len(name) <= 20 else name[:17] + "..."
        cv2.putText(img, label, (8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                   (30, 30, 30), 1, cv2.LINE_AA)
        color = (150, 90, 0) if is_proto else (
            (60, 150, 60) if s >= threshold else (60, 60, 210))
        x0 = margin_l
        x1 = max(x_of(s), x0 + 2)
        cv2.rectangle(img, (x0, y), (x1, y + bar_h), color, -1)
        cv2.putText(img, f"{s:.3f}", (x1 + 6, y + bar_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    return _save(img, out_dir / "agreement.png")


def annotate_regions(image: np.ndarray, findings: list[dict],
                     out_dir: Path, name: str = "annotated_llm.png") -> str:
    """Draw LLM-reported findings (0..1000 normalised boxes) onto an image."""
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    colors = {"low": (60, 180, 75), "medium": (30, 170, 230), "high": (50, 50, 230)}
    placed_label_rects: list[tuple[int, int, int, int]] = []   # (x0,y0,x1,y1) of drawn text
    for i, f in enumerate(findings, start=1):
        box = f.get("region", [0, 0, 0, 0])
        x0 = int(np.clip(box[0], 0, 1000) / 1000 * CANVAS_W)
        y0 = int(np.clip(box[1], 0, 1000) / 1000 * CANVAS_H)
        x1 = int(np.clip(box[2], 0, 1000) / 1000 * CANVAS_W)
        y1 = int(np.clip(box[3], 0, 1000) / 1000 * CANVAS_H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        color = colors.get(str(f.get("severity", "medium")).lower(),
                           colors["medium"])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        text = f"{i}. {f.get('title', 'finding')[:26]}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # v6 fix (Bug #27, PROJECT_MEMORY_v8.md): clamp the label's
        # starting x so a box near the right edge can't push its text
        # past the canvas boundary (previously only the y position was
        # kept on-canvas) -- this was part of the same "choppy, colliding
        # rectangle labels" report.
        lx = min(max(x0 + 3, 0), CANVAS_W - tw - 4)
        ly = max(y0 + 15, 13)
        # BUG #11 FIX (PROJECT_MEMORY_v8.md): push this label down past any
        # previously-drawn label it would overlap, instead of always
        # drawing at a fixed offset from its own box (which collides badly
        # when several findings share the same region — common with the
        # template's grouped deviant-feature findings).
        for _ in range(20):                              # bounded, never infinite-loop
            candidate = (lx, ly - th, lx + tw, ly + 4)
            overlap = any(not (candidate[2] < r[0] or candidate[0] > r[2]
                              or candidate[3] < r[1] or candidate[1] > r[3])
                         for r in placed_label_rects)
            if not overlap:
                break
            ly += th + 6
        ly = min(ly, CANVAS_H - 4)                        # keep on-canvas
        placed_label_rects.append((lx, ly - th, lx + tw, ly + 4))
        # Small filled backing rectangle so text stays legible over ink/other boxes.
        cv2.rectangle(img, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 3), (255, 255, 255), -1)
        cv2.putText(img, text, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return _save(img, out_dir / name)
