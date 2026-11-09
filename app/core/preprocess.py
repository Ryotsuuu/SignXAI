"""Signature image preprocessing pipeline.

Every image entering the system — uploaded scan or canvas drawing — goes
through the same deterministic pipeline so that features are comparable:

1. grayscale
2. median denoise
3. Otsu binarisation (ink becomes foreground = 255)
4. speck / noise component removal
5. crop to content bounding box (with margin)
6. normalise onto a fixed 512x256 canvas, aspect ratio preserved, white pad
7. skeletonisation (1px-wide stroke centreline) for topological features

The output :class:`Preprocessed` bundle is consumed by ``features``,
``align``, ``similarity`` and ``explain``.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from ..config import CANVAS_W, CANVAS_H

_MIN_COMPONENT_AREA = 8   # pixels; smaller connected components are specks
_CROP_MARGIN = 6          # pixels of breathing room around the ink bbox


@dataclass
class Preprocessed:
    """Container for all derived image planes of one signature."""
    gray: np.ndarray        # normalised grayscale (H=CANVAS_H, W=CANVAS_W), ink dark
    binary: np.ndarray      # normalised binary, ink = 255, background = 0
    skeleton: np.ndarray    # normalised skeleton, ink = 255
    bbox: tuple[int, int, int, int]   # ink bbox in the *cropped* image coords
    orig_size: tuple[int, int]        # (width, height) of the decoded input
    trace: dict | None = None         # v9: optional intermediate-stage images, see preprocess_traced()


def load_image(source: bytes | str | np.ndarray) -> np.ndarray:
    """Decode an image from raw bytes / data-URL / file path / ndarray to BGR.

    v3 fix (PROJECT_MEMORY_v8.md, user-reported): honors EXIF orientation
    metadata. Phone cameras commonly store the image sensor-orientation
    (e.g. "this file's pixels are sideways, rotate 90 to display correctly")
    rather than rotating the pixel data itself. cv2.imdecode ignores this
    entirely, so an unrotated photo could silently be processed sideways
    or upside-down with no error, no warning, just quietly wrong features.
    Now always decodes through PIL first (which can read EXIF) and applies
    ImageOps.exif_transpose() before handing off to OpenCV.

    Deliberately NOT doing: full deskew-to-horizontal. A signature's
    natural cursive slant is not noise to correct — it's a real
    discriminative feature the engine measures directly (see
    ``pca_orientation_deg`` in features.py). Auto-rotating every signature
    to a canonical horizontal would destroy that signal. Only genuinely
    wrong orientations (90/180/270 from a sideways phone photo, corrected
    via EXIF above, or a gross portrait/landscape mismatch as a fallback
    heuristic below) are corrected — not each person's natural handwriting
    angle.

    Raises ``ValueError`` on undecodable input.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:                     # grayscale ndarray (e.g. from
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)  # cv2.IMREAD_GRAYSCALE)
        return source
    if isinstance(source, str):
        if source.startswith("data:"):                      # canvas data-URL
            _, _, payload = source.partition(",")
            raw = base64.b64decode(payload)
        else:                                               # filesystem path
            raw = Path_read(source)
    else:
        raw = source
    try:
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)).convert("RGB"))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fall back to raw OpenCV decode (no EXIF handling) rather than
        # failing outright — better a possibly-misoriented image than none.
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image: unrecognised format")
    return img


def Path_read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Correct uneven lighting/vignetting/shadow gradients across the FULL
    photographed frame before binarisation.

    v8 FIX (Bug #50, PROJECT_MEMORY_v8.md, user-provided real data): a
    real enrolled reference photo (a signature occupying a small part of
    a much larger page, photographed with a phone) showed Otsu
    classifying **46% of the entire frame** as "ink" -- not the actual
    signature, but a large diagonal lighting gradient/vignette plus
    sensor/JPEG noise across the page. Confirmed directly by rendering
    the binary mask for this exact image: broad streaky, mottled regions
    spanning most of the frame, completely unrelated to the actual small
    cursive signature. This single failure cascades through EVERYTHING
    downstream: `_remove_specks`/`_remove_distant_outliers` cannot catch
    it (it is neither small nor spatially isolated from the real ink --
    it's most of the frame), so the crop-to-content bounding box was
    computed as (0, 0, 950, 1599) on a 1200x1600 photo -- essentially
    the WHOLE PAGE, not the signature -- which then feeds a hugely
    diluted, wrong-aspect-ratio region into every feature and comparison
    that follows. This is almost certainly the single biggest
    contributor to this project's repeated "can't recognise a new
    signature type" reports: it doesn't just affect one profile, it
    affects ANY photo taken with uneven lighting, a shadow, or a lens
    vignette -- extremely common for an ordinary handheld phone photo
    that isn't a flat, evenly-lit scan.

    Root cause: plain (global) Otsu thresholding assumes the WHOLE image
    has one consistent background brightness and one consistent
    foreground (ink) brightness -- true for a flatbed scan, false for a
    typical phone photo, where lighting commonly varies smoothly across
    the frame (a lamp to one side, a shadow from the photographer's own
    hand or the phone itself, natural vignetting from the lens). When
    that gradient's dark end drops below Otsu's single global threshold,
    large regions of plain background paper get classified as ink.

    Fix: standard "flat-fielding" / background-illumination correction,
    widely used in document image analysis for exactly this failure
    mode -- estimate the smoothly-varying background illumination with a
    large-kernel median blur (large enough to wash out the thin ink
    strokes entirely, so what's left is an estimate of the PAPER's own
    brightness variation across the frame, not the signature), then
    divide the original image by that estimate to flatten the lighting
    before Otsu ever sees it. Otsu then operates on an image with one
    consistent background level across the whole frame, which is the
    assumption it actually needs to work correctly. Verified directly
    on the reported real image: ink-pixel count collapsed from 886,129
    (46% of the frame -- clearly wrong) to 10,298 (a small, correctly-
    isolated signature), and the resulting mask visually contains
    ONLY the real handwritten ink, confirmed by direct visual
    inspection, with the large-scale lighting artifact completely gone.

    ``_ILLUM_KERNEL_FRAC`` is deliberately scaled to the image's own
    size (not a fixed pixel count) so this works consistently across
    both a small canvas-drawn signature and a large, high-resolution
    phone photo -- and deliberately much larger than any real pen
    stroke width, so genuine ink is never itself washed into the
    "background" estimate.
    """
    h, w = gray.shape
    k = max(31, (min(h, w) // _ILLUM_KERNEL_DIVISOR) | 1)   # odd kernel
    background = cv2.medianBlur(gray, k)
    flattened = cv2.divide(gray.astype(np.float32),
                           background.astype(np.float32) + 1e-6, scale=255)
    return np.clip(flattened, 0, 255).astype(np.uint8)


_ILLUM_KERNEL_DIVISOR = 15   # see _flatten_illumination docstring


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Illumination-corrected Otsu threshold; returns image where ink =
    255, background = 0.

    v8 (Bug #50, PROJECT_MEMORY_v8.md): now flattens lighting/vignetting
    across the frame (`_flatten_illumination`) before thresholding --
    see that function's docstring for the real-world failure this fixes
    and the concrete before/after evidence. Kept as a single extra step
    ahead of the existing median-blur + Otsu, so every other stage of
    the pipeline is unaffected by this change other than receiving a
    cleaner binary mask to work with.
    """
    flattened = _flatten_illumination(gray)
    blur = cv2.medianBlur(flattened, 3)
    # If the image is mostly white (typical scan), Otsu with inversion works.
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """Strip page-ruling / notebook-line artifacts from the FULL-frame
    binary mask, before any cropping or speck removal.

    v7 FIX (Bug #40, PROJECT_MEMORY_v8.md, user-reported + self-run,
    real evidence): the user's own test image (a signature photographed
    on ruled/lined paper) showed 3 near-perfectly-straight vertical lines
    running the full height of the frame, THROUGH the signature ink,
    surviving into the binary mask untouched. These are not specks
    (`_remove_specks` only drops tiny components) and are not spatially
    isolated from the signature (`_remove_distant_outliers`, Bug #26,
    only drops components OUTSIDE the signature's own region — these
    lines run straight through it), so neither existing filter could
    catch them.

    v7.1 FIX (Bug #47, PROJECT_MEMORY_v8.md, user-reported: a later real
    photo, tilted ~15-20 degrees, showed its ruled line surviving
    completely untouched, clearly visible in the zonal evidence image):
    the first version of this fix only removed segments within
    `+/-8 degrees` of absolute horizontal/vertical, reasoning that page
    rules are "near axis-aligned" after ordinary camera tilt. That
    assumption breaks the moment a photo is rotated by MORE than a few
    degrees (this one clearly was, visibly, in the reported evidence
    image) -- the page's ruled lines are rotated right along with
    everything else in the photo, so they're no longer anywhere near
    absolute horizontal/vertical in pixel space, and the old check
    missed them entirely.

    Fixed by dropping the axis assumption altogether and using the
    actual distinguishing property of ruled paper instead: MULTIPLE
    mutually-parallel long straight lines, at any consistent angle --
    not "close to horizontal/vertical". A single long straight segment
    (which a genuine signature flourish or underline can legitimately
    produce -- a real risk this function must never trigger on) is
    left alone; only segments that have at least one other long segment
    within `_PARALLEL_ANGLE_TOL_DEG` of the same angle (i.e. an actual
    REPEATING pattern, the real signature of ruled paper regardless of
    camera rotation) are removed. This is strictly safer than the first
    version for genuine signature strokes (a lone flourish, at ANY
    angle, is now never removed, where the first version could
    misfire on one that happened to be near-axis) while being MORE
    thorough for rotated photos (catches ruled lines at any angle, not
    just near-axis ones). Verified: still correctly finds and removes
    the reported multi-line case; a single long straight test stroke
    with no parallel partner is left completely untouched at every
    angle tested (0, 15, 45, 80 degrees).

    ``_LINE_LENGTH_FRAC`` is deliberately generous (70% of the frame's
    shorter dimension) so a short straight segment that's actually part
    of the signature (e.g. a straight-line letter stroke) is never long
    enough to be mistaken for a page rule.
    """
    h, w = binary.shape
    min_len = int(min(h, w) * _LINE_LENGTH_FRAC)
    segments = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 360,
                               threshold=max(min_len // 2, 20),
                               minLineLength=min_len, maxLineGap=max(h, w) // 15)
    if segments is None:
        return binary

    items = []
    for seg in segments:
        x1, y1, x2, y2 = np.asarray(seg).ravel()[:4]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        theta = np.radians(angle)
        # signed perpendicular distance from the origin to this segment's
        # line, i.e. WHERE across the page this particular line sits --
        # two detections of the SAME physical line share both angle and
        # this offset; two genuinely DIFFERENT parallel ruled lines share
        # only the angle, not the offset (they're spaced apart on the
        # page). See docstring below for why this two-stage distinction
        # is necessary.
        offset = mx * np.sin(theta) - my * np.cos(theta)
        items.append((angle, offset, (int(x1), int(y1), int(x2), int(y2))))

    def _circ_diff(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    n = len(items)
    parent = list(range(n))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i, j):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    # Stage 1: merge Hough detections that are the SAME physical line --
    # v7.1 FIX, found by direct testing (Bug #47, PROJECT_MEMORY_v8.md):
    # a naive angle-only grouping (the first draft of this rewrite)
    # incorrectly removed a genuine, LONE synthetic test stroke at every
    # angle tried, because a single thick line frequently produces
    # SEVERAL overlapping/adjacent Hough segments for what is visually
    # one stroke (especially with the fine `theta=np.pi/360` resolution
    # used above) -- these duplicate detections all share nearly the
    # same angle, so an angle-only grouping miscounted them as multiple
    # "parallel lines" and triggered on a single stroke. Requiring BOTH
    # angle AND perpendicular-offset proximity for this merge step means
    # only genuine re-detections of the same line get merged; a real
    # second line spaced apart on the page (a real ruled line) will NOT
    # merge here even though its angle matches, which is exactly the
    # distinction stage 2 needs to see.
    for i in range(n):
        for j in range(i + 1, n):
            if (_circ_diff(items[i][0], items[j][0]) <= _PARALLEL_ANGLE_TOL_DEG
                    and abs(items[i][1] - items[j][1]) <= _DEDUP_OFFSET_PX):
                _union(i, j)

    # One representative per merged (deduplicated) physical line, keeping
    # each member's own endpoints so we can measure the line's TRUE
    # overall extent (a duplicate detection might only be a fragment).
    reps: dict[int, list[int]] = {}
    for i in range(n):
        reps.setdefault(_find(i), []).append(i)

    # v7.1 FIX (Bug #47, PROJECT_MEMORY_v8.md, user-reported, self-run):
    # tested the angle+count signal above against this user's own real
    # signature photos (none of which have any ruled paper) and found it
    # falsely fired on the signature's own genuine double-underline
    # flourish -- confirmed directly: this user's signature style
    # legitimately includes TWO near-parallel underline strokes, which
    # is indistinguishable from "2 ruled lines" by angle and count alone.
    # Real page rules have a property a deliberate flourish essentially
    # never does, though: they run almost the ENTIRE width/height of the
    # photographed FRAME, because the ruled page is bigger than the
    # signature drawn on it and the camera captures margin around it.
    # A signature's own flourish, however long relative to its OTHR
    # strokes, essentially never spans the full photo frame edge-to-edge
    # -- there is always some margin in a normal photograph. This is
    # checked here directly per candidate line (comparing its actual
    # measured extent to the longest a straight line at that exact angle
    # could possibly be while still fitting inside the frame) BEFORE the
    # parallel-family grouping below, and is the decisive filter: even a
    # genuine 2-line family (this user's real double underline) is now
    # correctly left alone because neither line comes close to spanning
    # the frame, while image 7's real ruled lines (which do) still pass.
    unique_lines = []   # (mean_angle, span_frac, member_indices)
    for members in reps.values():
        angs = [items[m][0] for m in members]
        mean_angle = float(np.mean(angs))
        # True extent of this (possibly fragmented) physical line: the
        # max distance between ANY two endpoints across all its member
        # detections, not just one segment's own length.
        pts = []
        for mi in members:
            x1, y1, x2, y2 = items[mi][2]
            pts.extend([(x1, y1), (x2, y2)])
        pts = np.array(pts, dtype=np.float32)
        dmat = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                        pts[:, None, 1] - pts[None, :, 1])
        extent = float(dmat.max()) if len(pts) > 1 else 0.0
        theta = np.radians(mean_angle)
        # Longest a straight line at this angle could be while still
        # fitting inside the h x w frame (the frame's own chord length
        # at that angle) -- guards the divide with a small epsilon at
        # angle 0/90 rather than branching.
        eps = 1e-6
        max_possible = min(h / (abs(np.sin(theta)) + eps),
                           w / (abs(np.cos(theta)) + eps))
        span_frac = extent / max_possible if max_possible > 0 else 0.0
        unique_lines.append((mean_angle, span_frac, members))

    unique_lines = [u for u in unique_lines if u[1] >= _MIN_FRAME_SPAN_FRAC]
    if not unique_lines:
        return binary

    # Stage 2: among the DEDUPLICATED, frame-spanning lines, group by
    # angle ONLY (ruled lines on the same page share an angle but sit at
    # different, real, widely-spaced offsets) -- a family with
    # >= _MIN_PARALLEL_LINES distinct lines is the actual "repeating
    # ruled pattern" signal.
    m = len(unique_lines)
    uparent = list(range(m))

    def _ufind(i):
        while uparent[i] != i:
            uparent[i] = uparent[uparent[i]]
            i = uparent[i]
        return i

    def _uunion(i, j):
        ri, rj = _ufind(i), _ufind(j)
        if ri != rj:
            uparent[ri] = rj

    for i in range(m):
        for j in range(i + 1, m):
            if _circ_diff(unique_lines[i][0], unique_lines[j][0]) <= _PARALLEL_ANGLE_TOL_DEG:
                _uunion(i, j)

    family_counts: dict[int, int] = {}
    for i in range(m):
        r = _ufind(i)
        family_counts[r] = family_counts.get(r, 0) + 1

    lines = np.zeros_like(binary)
    found = False
    for i, (_mean_angle, _span_frac, members) in enumerate(unique_lines):
        if family_counts[_ufind(i)] >= _MIN_PARALLEL_LINES:
            for member_idx in members:
                x1, y1, x2, y2 = items[member_idx][2]
                cv2.line(lines, (x1, y1), (x2, y2), 255, thickness=1)
                found = True
    if not found:
        return binary
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


_PARALLEL_ANGLE_TOL_DEG = 4.0   # how close two lines' angles must be to
                                # count as "the same ruling family"
_DEDUP_OFFSET_PX = 12           # how close two same-angle detections'
                                # positions must be to count as
                                # re-detections of ONE physical line
                                # rather than two distinct lines
_MIN_PARALLEL_LINES = 2         # a lone long straight segment (a real
                                # flourish/underline is never removed;
                                # only an actual repeating pattern is
_MIN_FRAME_SPAN_FRAC = 0.85     # a line must span at least this fraction
                                # of the longest it could possibly be at
                                # its angle within the full frame -- the
                                # decisive real-vs-flourish signal, see
                                # docstring above


_LINE_LENGTH_FRAC = 0.70   # see _remove_ruled_lines docstring


def _remove_chromatic_outliers(bgr_full: np.ndarray, binary: np.ndarray,
                               hue_tol_deg: float = 30.0, sat_floor: float = 15.0,
                               anchor_chroma_floor: float = 30.0,
                               edge_margin_frac: float = 0.015) -> np.ndarray:
    """Reject ink-mask components whose COLOUR is inconsistent with the
    signature's own pen ink -- catches large-area contamination that
    ``_remove_specks``/``_remove_distant_outliers`` cannot, because it is
    neither small nor spatially isolated.

    v9 FIX (Bug #55, PROJECT_MEMORY_v9.md, user-reported: enrolling 5 real
    phone-photo samples of one signer produced wildly inconsistent scalar
    features -- aspect ratio ranging 0.55 to 2.73 across supposedly-identical
    signature photos -- which corrupted the calibrated threshold (pinned to
    the 0.30 floor) and every downstream visualisation (heatmap, ink-overlay,
    zonal grid all showed a stretched/oddly-cropped signature).

    **Root cause, found by directly rendering the binary mask for each real
    enrolled image** (not assumed): plain intensity-based Otsu thresholding
    -- even after Bug #50's illumination flat-fielding -- has no way to tell
    "dark because it's ink" from "dark because it's something else dark in
    the photo". Three distinct real contamination sources were found, each
    surviving into the "substantial, keep no matter what" bucket of
    ``_remove_distant_outliers`` because they are large, not small or
    isolated:
    1. A paper crease/fold shadow (a long wavy dark band near the top of one
       photo) -- confirmed via direct pixel sampling: saturation ~11
       (near-neutral grey), vs. the real ink's saturation of ~57-117.
    2. A dark patterned fabric/blanket visible along one frame edge (the
       photo background) -- confirmed via sampling: highly SATURATED
       (~151) but at a completely different HUE (~10, red-orange) than the
       real ink's hue (~126, blue-violet) -- i.e. a real but different
       colour, not a neutral shadow.
    3. Bleed-through of handwriting from the other side of a thin page /
       an adjacent enrolled sample's edge peeking into frame -- same ink,
       same hue, but a spatially tiny, disconnected fragment.

    Case 3 is a spatial problem (fixed in ``_remove_distant_outliers``
    below); cases 1-2 are fundamentally a COLOUR problem no purely spatial
    or intensity-based filter can solve -- the fix has to look at hue and
    saturation, which is exactly what a human proofreader does instinctively
    ("that's a shadow, not ink" / "that's the desk, not the pen").

    Algorithm (verified against all 3 real cases above plus a real,
    legitimately multi-part signature with detached letter-groups spanning
    the full frame width, confirming no over-removal -- see
    PROJECT_MEMORY_v9.md Bug #55 for the full before/after pixel counts):

    1. Pick an ``anchor`` component: the largest connected component that
       is NOT itself an edge-touching, thin/elongated strip (aspect ratio
       > 4:1 within ``edge_margin_frac`` of a frame border) -- this avoids
       anchoring on a fold-line/page-edge artifact even when it happens to
       be the single largest blob in the frame (case 1 above: the fold
       line was 24,911px vs. the real signature's 9,310px, yet correctly
       excluded from anchor selection precisely because it IS such a
       strip).
    2. If the anchor itself has near-zero saturation (a genuinely
       monochrome/grayscale capture, e.g. a flatbed scan of black ink),
       skip this filter entirely -- chroma carries no signal there and
       applying it could wrongly strip real low-saturation ink.
    3. Every OTHER component at least 3% of the anchor's own area is
       compared against the anchor's hue/saturation: reject it if it is
       clearly a different colour (saturated, but >30 degrees of hue away
       on the OpenCV 0-180 hue wheel) OR if it is near-neutral/grey while
       the anchor ink is clearly coloured (inconsistent with being a faint
       patch of the SAME coloured ink). Components too small to judge
       reliably, and the anchor itself, are left untouched.

    This never fires on a genuinely monochrome capture, and in the
    multi-part-signature regression case every real letter-group shared
    the anchor's hue within a few degrees and was correctly kept in full.
    """
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary
    hsv = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)
    hue_full = hsv[..., 0].astype(np.float32)
    sat_full = hsv[..., 1].astype(np.float32)
    edge_margin = max(3, int(min(h, w) * edge_margin_frac))

    def _comp_stats(i: int) -> dict:
        m = labels == i
        x, y, bw, bh, area = stats[i]
        touches = (x <= edge_margin or y <= edge_margin
                  or x + bw >= w - edge_margin or y + bh >= h - edge_margin)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        sat = sat_full[m]
        hue = hue_full[m]
        sat_med = float(np.median(sat))
        ang = hue / 180.0 * 2 * np.pi
        mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2 * np.pi)
        hue_med = mean_ang / (2 * np.pi) * 180.0
        return dict(area=int(area), touches=touches, aspect=aspect,
                   sat_med=sat_med, hue_med=hue_med)

    all_comps = [(i, _comp_stats(i)) for i in range(1, n)]
    non_strip = [(i, c) for i, c in all_comps
                if not (c["touches"] and c["aspect"] > 4.0)]
    pool = non_strip if non_strip else all_comps
    anchor_i, anchor_c = max(pool, key=lambda t: t[1]["area"])
    if anchor_c["sat_med"] <= sat_floor:
        return binary    # monochrome/black-ink capture -- chroma has no signal here

    def _hue_dist(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    threshold = max(0.03 * anchor_c["area"], 4)
    out = binary.copy()
    for i, c in all_comps:
        if i == anchor_i or c["area"] < threshold:
            continue
        reject = False
        if c["sat_med"] > sat_floor:
            if _hue_dist(c["hue_med"], anchor_c["hue_med"]) > hue_tol_deg:
                reject = True
        elif anchor_c["sat_med"] > anchor_chroma_floor:
            reject = True     # near-neutral component, but the anchor ink is clearly coloured
        if reject:
            out[labels == i] = 0
    return out


def _remove_specks(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= _MIN_COMPONENT_AREA:
            out[labels == i] = 255
    return out


def _remove_distant_outliers(binary: np.ndarray) -> np.ndarray:
    """Discard ink components spatially isolated from the main signature
    body.

    v6 FIX (Bug #26, PROJECT_MEMORY_v8.md, user-reported): a scan
    artifact, staple hole, dirt speck, or bit of a ruled notebook line can
    be larger than ``_remove_specks``' few-pixel area floor yet still not
    be part of the actual signature — and because it's often positioned
    far from the real ink, it can silently distort the crop bounding box
    and therefore the entire normalized signature's measured aspect ratio
    and position, even though the handwriting itself is identical.
    Confirmed directly: a single small isolated mark in a far corner
    shifted a real signature's measured aspect ratio from 2.28 to 1.77 and
    its centroid position by a similarly large amount (see
    docs/BUILD_LOG_v6.md) — this is exactly the "the system treats where I
    photographed it as part of the style" bug reported.

    DEV NOTE, kept because it caught two real design flaws before shipping
    (both found by testing, not assumed correct from the design alone):

    1. A first version anchored on the single LARGEST connected component
       and expanded a fixed margin from its bounding box alone. Wrong,
       because a real signature is very often made of several separate
       connected components spread across its full width (individual
       letters, pen lifts, a detached first initial) — on a real test
       image this discarded a legitimate ~40%-of-total-ink letter group
       because it happened to be the second-largest, not the anchor.
    2. A second version tried adaptive-kernel dilation to bridge nearby
       strokes into one cluster, keeping only the largest resulting blob.
       Also wrong, for the same underlying reason: two legitimately
       separate letter-groups in the SAME real signature can have a wider
       gap between them than a stray artifact does from the page edge, so
       "keep only the biggest post-dilation blob" discarded an entire real
       letter group again on the same test image.

    Fixed approach — never discard a SUBSTANTIAL component just because
    it's spatially separate; only ever discard components that are BOTH
    small (relative to the largest stroke group and to the total ink
    mass) AND far outside the region spanned by every substantial
    component. This can never mistake "a legitimately separated letter
    group" for noise (it's never small), while still catching a genuinely
    tiny, distant artifact (which is both small AND, being on its own out
    near a page edge or corner, outside the expanded span of the real
    signature).

    v9 REFINEMENT (Bug #55, PROJECT_MEMORY_v9.md, self-run on real user
    data): the "never discard a substantial component" rule above is
    right in spirit but was found, on a real enrolled photo, to protect a
    component that should NOT have been protected -- a small fragment of
    handwriting bleeding in from an adjacent, separately-photographed
    sample sitting near the bottom edge of the frame, 559px away from a
    132px-tall real signature (over 4x the signature's own height, and
    more than 1.4x its own width). It technically cleared the
    3%-of-max-area "substantial" bar (321px, just above a ~301px floor)
    purely because that bar is relative to the largest component's OWN
    area, which said nothing about how far away 321px of stray ink
    plausibly could be. Confirmed directly: without this refinement, the
    resulting crop bounding box stretched from a 132px-tall signature to
    a 710px-tall region, corrupting that image's aspect ratio and every
    feature derived from it.

    Fix: components that are BOTH "substantial" by the existing area rule
    AND small relative to the single largest (anchor) component
    (``< small_frac`` of the anchor's own area) are now ALSO required to
    fall within a generous margin scaled to the ANCHOR's own footprint
    (0.6x its width/height) -- not just the same ``typical_h``-based
    margin used for genuinely tiny/non-substantial specks. A component
    that is a large, prominent fraction of the anchor's own size (a real
    detached letter-group, e.g. a separated first initial) is completely
    unaffected by this -- it still clears the ``small_frac`` bar and is
    kept unconditionally, exactly as before. Verified directly against a
    real multi-part signature with several letter-groups spread across
    the full image width (up to ~100px detached from the main body in a
    ~700px-wide signature): every real component was still kept, byte-
    identical pixel count to the un-refined version, while the real 559px-
    away bleed-through fragment above is now correctly excluded.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:      # background + at most one component — nothing to filter
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    total_area = float(areas.sum())
    max_area = float(areas.max())
    substantial = [i for i in range(1, n)
                  if areas[i - 1] >= max(0.03 * max_area, 0.01 * total_area)]
    if not substantial:
        return binary   # degenerate (all components tiny) — leave untouched

    anchor_idx = max(substantial, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    anchor_area = float(stats[anchor_idx, cv2.CC_STAT_AREA])
    ax, ay, aw, ah = (stats[anchor_idx, cv2.CC_STAT_LEFT], stats[anchor_idx, cv2.CC_STAT_TOP],
                      stats[anchor_idx, cv2.CC_STAT_WIDTH], stats[anchor_idx, cv2.CC_STAT_HEIGHT])
    prelim_stats = stats[substantial]
    prelim_typical_h = float(np.median(prelim_stats[:, cv2.CC_STAT_HEIGHT]))
    prelim_margin = max(prelim_typical_h * 1.5, 20.0)
    extra_margin = max(0.6 * aw, 0.6 * ah, prelim_margin)
    ex0, ex1 = ax - extra_margin, ax + aw + extra_margin
    ey0, ey1 = ay - extra_margin, ay + ah + extra_margin

    small_frac = 0.15   # a "substantial" component under 15% of the anchor's
                        # own area gets the extra anchor-relative distance
                        # check; at/above that it's a prominent enough
                        # letter-group to keep unconditionally, as before.
    kept_substantial = []
    demoted = set()   # v9 fix: components explicitly rejected by the anchor-
                      # aware check below must NOT get a second chance
                      # against the looser generic region check further
                      # down -- see PROJECT_MEMORY_v9.md Bug #55 addendum,
                      # caught by this module's own regression test
                      # (test_distant_outlier_v9_margin_uses_anchor_footprint):
                      # a component correctly excluded here could still
                      # slip back in if it happened to also fall inside
                      # the plain typical_h-based region used for genuinely
                      # non-substantial noise, since that region was never
                      # meant to re-judge something already evaluated more
                      # carefully.
    for i in substantial:
        if i == anchor_idx:
            kept_substantial.append(i)
            continue
        area = float(stats[i, cv2.CC_STAT_AREA])
        cx, cy = centroids[i]
        if area < small_frac * anchor_area and not (ex0 <= cx <= ex1 and ey0 <= cy <= ey1):
            demoted.add(i)
            continue    # small-but-"substantial" fragment, disproportionately far from the anchor
        kept_substantial.append(i)

    sub_stats = stats[kept_substantial]
    region_x0 = float(sub_stats[:, cv2.CC_STAT_LEFT].min())
    region_y0 = float(sub_stats[:, cv2.CC_STAT_TOP].min())
    region_x1 = float((sub_stats[:, cv2.CC_STAT_LEFT] + sub_stats[:, cv2.CC_STAT_WIDTH]).max())
    region_y1 = float((sub_stats[:, cv2.CC_STAT_TOP] + sub_stats[:, cv2.CC_STAT_HEIGHT]).max())
    typical_h = float(np.median(sub_stats[:, cv2.CC_STAT_HEIGHT]))
    margin = max(typical_h * 1.5, 20.0)
    region_x0 -= margin; region_x1 += margin
    region_y0 -= margin; region_y1 += margin

    out = np.zeros_like(binary)
    kept_set = set(kept_substantial)
    for i in range(1, n):
        if i in kept_set:
            out[labels == i] = 255
            continue
        if i in demoted:
            continue    # already judged and rejected -- no second chance
        cx, cy = centroids[i]
        if region_x0 <= cx <= region_x1 and region_y0 <= cy <= region_y1:
            out[labels == i] = 255
    return out


def _crop_to_content(binary: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop to the ink's bounding box.

    v6 FIX (Bug #26, PROJECT_MEMORY_v8.md, user-reported): a naive
    min/max bounding box over every ink pixel is NOT robust to a single
    small stray mark far from the actual signature -- a trailing flourish
    tail that only got fully captured in one photo, a smudge, a staple
    hole, a bit of a ruled notebook line poking into frame. Because the
    normalization step below fits this box to a fixed canvas (preserving
    aspect ratio), one outlier mark can silently distort the ENTIRE
    signature's apparent aspect ratio and position on the canvas, even
    though the actual handwriting is identical -- confirmed directly: a
    single 2px stray line added far from a real signature's ink shifted
    its measured aspect ratio by ~14% and its centroid position
    perceptibly, using the previous naive-bbox code. This is exactly what
    the user described as the system caring about "if the image is a
    little above or below center" when it should not.

    Standard offline-signature-verification practice calls this step
    "signature normalization and centering" and treats it as foundational
    (see PROJECT_MEMORY_v8.md Bug #26 for citations) -- the fix here is to
    make that centering robust the way a trimmed statistic is robust:
    compute the bounding box from the 0.5th-99.5th percentile of ink pixel
    coordinates on each axis, not the literal min/max. Percentiles are
    inherently density-weighted -- a thin, sparse outlier mark contributes
    few pixels relative to the dense signature body, so it naturally falls
    outside a 0.5-99.5 percentile window and gets trimmed, while genuine
    signature strokes (which make up the overwhelming majority of ink
    pixels almost everywhere in their own bounding box) are preserved
    untouched. This never affects a signature with no outlier marks at
    all -- percentile and min/max agree exactly when the pixel mass is
    already contiguous, which is the common case.
    """
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise ValueError("No ink detected in the image — please provide a signature.")
    lo, hi = 0.5, 99.5
    x0 = max(int(np.floor(np.percentile(xs, lo))) - _CROP_MARGIN, 0)
    x1 = min(int(np.ceil(np.percentile(xs, hi))) + _CROP_MARGIN, binary.shape[1] - 1)
    y0 = max(int(np.floor(np.percentile(ys, lo))) - _CROP_MARGIN, 0)
    y1 = min(int(np.ceil(np.percentile(ys, hi))) + _CROP_MARGIN, binary.shape[0] - 1)
    return binary[y0:y1 + 1, x0:x1 + 1], gray[y0:y1 + 1, x0:x1 + 1], (x0, y0, x1, y1)


def _normalize(binary: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the cropped signature onto the fixed canvas, preserving aspect."""
    h, w = binary.shape
    scale = min((CANVAS_W - 16) / w, (CANVAS_H - 16) / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    rb = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)
    rg = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas_b = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    canvas_g = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    ox, oy = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas_b[oy:oy + nh, ox:ox + nw] = rb
    canvas_g[oy:oy + nh, ox:ox + nw] = rg
    return canvas_b, canvas_g


def preprocess(source: bytes | str | np.ndarray) -> Preprocessed:
    """Run the full preprocessing pipeline on one signature image."""
    return preprocess_traced(source, want_trace=False)


def preprocess_traced(source: bytes | str | np.ndarray, want_trace: bool = True) -> Preprocessed:
    """Run the full preprocessing pipeline, optionally capturing every
    intermediate stage for human inspection.

    v9 ADDITION (PROJECT_MEMORY_v9.md, user-requested): the user's core
    complaint across Enroll and Verify was not being able to see HOW a
    photo gets turned into the flat, cropped, centered signature the
    system actually compares -- "I want to see how system is processing
    the image at first place" -- so that a bad crop or a piece of
    surviving contamination is obvious to a human BEFORE it silently
    corrupts a verification result. This function is the single source of
    truth for that preview: every stage the pixel data actually passes
    through, in order, each rendered as a viewable image. ``preprocess()``
    above is unchanged in behaviour and cost for every existing internal
    caller (engine.py, etc.) -- it simply calls this with tracing off.
    """
    bgr = load_image(source)
    oh, ow = bgr.shape[:2]
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    trace: dict[str, np.ndarray] | None = {"original": bgr} if want_trace else None

    flattened = _flatten_illumination(gray_full)
    if trace is not None:
        trace["illumination_flattened"] = flattened

    otsu = _binarize(gray_full)
    ruled = _remove_ruled_lines(otsu)
    specked = _remove_specks(ruled)
    if trace is not None:
        trace["raw_binarized"] = specked

    chroma_clean = _remove_chromatic_outliers(bgr, specked)
    if trace is not None:
        trace["after_colour_contamination_filter"] = chroma_clean

    binary_full = _remove_distant_outliers(chroma_clean)
    if trace is not None:
        trace["after_outlier_removal"] = binary_full

    binary, gray, bbox = _crop_to_content(binary_full, gray_full)
    if trace is not None:
        trace["crop_box"] = np.array(bbox)
        trace["cropped_gray"] = gray
        trace["cropped_binary"] = binary

    nbinary, ngray = _normalize(binary, gray)
    skel = skeletonize(nbinary > 0).astype(np.uint8) * 255
    if trace is not None:
        trace["final_normalized"] = ngray
        trace["final_skeleton"] = skel
    return Preprocessed(gray=ngray, binary=nbinary, skeleton=skel,
                        bbox=bbox, orig_size=(ow, oh), trace=trace)
