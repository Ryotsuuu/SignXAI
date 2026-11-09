# SignXAI — Complete System Walkthrough

**What this document is:** a step-by-step trace of what the code
actually does, file by file and method by method, from the moment a
photo is uploaded to the moment a verdict and report exist — including
exactly which numbers ("signals") drive the final remark and where in
the code each one is computed. Where `docs/ALGORITHM_DESIGN.md` explains
*why* each algorithm was chosen, this document explains *what happens,
in what order, calling what*. Read this one to understand or debug the
system's behavior; read that one to understand or change its design.

`PROJECT_MEMORY_v8.md` is the third document in this set — the
chronological history of every bug found and fixed. This document
describes the system AS IT STANDS TODAY, assuming all of that history's
fixes are already applied.

---

## 0. The two-phase shape of the whole system

Every request to this system is one of exactly two phases, and nothing
else:

1. **Enrollment** (`VerificationEngine.enroll`) — teach the system what
   one person's genuine signature looks like, from 1 or more sample
   images.
2. **Verification** (`VerificationEngine.verify`) — compare one new
   signature image against everything enrolled for one profile, and
   produce a verdict.

Everything else in the codebase (the LLM, the PDF report, the web UI,
the CLI) is a consumer of the output of these two phases, or plumbing to
get images in and results out. **The verdict itself is decided entirely
by classical, deterministic code before either the LLM or any neural
network ever runs** — this is the single most important architectural
fact about the system, and it's why every bug-fix in this project's
history has focused on `app/core/`, not `app/llm/`.

```
photo bytes  →  preprocess.py  →  features.py  →  align.py + similarity.py + zoning.py
                                                          │
                                                          ▼
                                              engine.py (fuses everything,
                                              decides GENUINE/FORGED/INCONCLUSIVE)
                                                          │
                                     ┌────────────────────┼────────────────────┐
                                     ▼                    ▼                    ▼
                              shap_explain.py       explain.py            app/llm/*
                            (WHY this score)   (heatmaps/overlays,    (turns the verdict +
                                                  human-readable)      evidence into a written
                                                                       report — NEVER changes
                                                                       the verdict itself)
```

---

## 1. File-by-file reference

### `app/core/preprocess.py` — turn a photo into a clean, comparable canvas

**Entry point:** `preprocess(source) -> Preprocessed`. `Preprocessed` is
a small dataclass holding `.gray`, `.binary`, `.skeleton`, and `.bbox` —
everything every later stage needs.

Called once per image (both at enrollment, for every reference, and at
verification, for the query). Internally, in exact order:

1. **`load_image(source)`** — accepts bytes, a file path, or an already-
   decoded array. Uses PIL first (`Image.open` + `.convert("RGB")`,
   which safely normalizes CMYK/palette/grayscale/RGBA input) and
   applies `ImageOps.exif_transpose` — this is what corrects a photo
   that a phone saved sideways/upside-down via its EXIF orientation
   tag, BEFORE anything else touches it. Falls back to raw OpenCV
   decoding if PIL can't open the file. Returns a BGR array.
2. **Grayscale conversion** — `cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)`.
   This becomes `gray_full`, kept and later returned as `.gray` — every
   pixel-level comparison (SSIM, HOG, LBP) operates on THIS, not on the
   binary mask.
3. **`_binarize(gray_full)`** — the ink/background split. Internally:
   - **`_flatten_illumination(gray)`** (v8, Bug #50) runs FIRST: estimates
     the page's own smoothly-varying brightness with a large-kernel
     median blur (kernel scaled to image size, `_ILLUM_KERNEL_DIVISOR`),
     then divides the original by that estimate. This is what stops an
     uneven lighting gradient or lens vignette from being misread as ink
     — the single most impactful preprocessing fix in this project's
     history (see `PROJECT_MEMORY_v8.md` Bug #50 for the real numbers).
   - A light 3x3 median blur for noise, then **Otsu's threshold**
     (`cv2.threshold(..., THRESH_BINARY_INV + THRESH_OTSU)`) — picks the
     ink/background cutpoint automatically from the image's own
     brightness histogram, now that the histogram is illumination-
     corrected. Output: `binary_full`, ink = 255, background = 0.
4. **`_remove_ruled_lines(binary_full)`** (v7/v8, Bugs #40/#47) — strips
   notebook/page-ruling lines. Uses `cv2.HoughLinesP` to find long
   straight segments at ANY angle (not axis-restricted — a rotated photo
   rotates its page lines too), deduplicates multiple detections of the
   same physical line (by matching angle AND position), then only
   removes a line if (a) at least 2 mutually-parallel deduplicated lines
   share that angle (a real repeating ruling pattern — a lone signature
   flourish is never removed) AND (b) the line spans at least 85% of the
   longest it could possibly be at that angle within the full frame
   (`_MIN_FRAME_SPAN_FRAC` — the decisive check: a real page rule spans
   the whole photographed page; a signature's own flourish never does,
   even a long one).
5. **`_remove_specks(binary)`** — drops tiny connected components below
   a minimum pixel area (camera sensor noise, JPEG artifacts, dust).
6. **`_remove_distant_outliers(binary)`** — drops components that are
   large enough to survive step 5 but spatially far from the main ink
   cluster (a stray pen mark, a staple hole, a fold shadow). Density-
   based, not simple bounding-box distance (Bug #26).
7. **`_crop_to_content(binary, gray)`** — finds the tight bounding box
   around whatever ink survived steps 3-6 (a percentile-trimmed
   bounding box, not naive min/max, so one remaining stray pixel can't
   blow up the crop — also Bug #26), and crops both `binary` and `gray`
   to it. This bounding box is stored as `.bbox` — it's what
   `aspect_ratio` and several other features are computed from.
8. **`_normalize(binary, gray)`** — fits the cropped content onto the
   fixed `CANVAS_W x CANVAS_H = 512x256` canvas, preserving aspect ratio
   with padding. This is what makes two photos taken at different
   distances/crops/resolutions comparable at all — every feature and
   comparison from this point on operates in this normalized space.
9. **Skeletonization** — `skimage.morphology.skeletonize` on the final
   binary mask, producing `.skeleton` (a 1-pixel-wide stroke centerline)
   — used by `features.py` for topology counts and by `zoning.py` for
   corner-point detection.

**What comes out:** a `Preprocessed` object. Nothing downstream ever
looks at the original photo bytes again except for the LLM's evidence
images (which show the human-readable grayscale, not raw bytes).

### `app/core/features.py` — turn a `Preprocessed` image into named numbers

**Entry point:** `extract(pre) -> dict` with two keys, `"scalars"` (a
flat dict of ~22 named numbers) and `"vectors"` (HOG/LBP arrays for
cosine comparison). Called once per image, result cached on the
profile's reference records so it's never recomputed at verify time for
existing references — only the new query gets computed fresh.

The scalar catalogue, by category (these are exactly the labels shown in
the human-readable "Quantitative feature analysis" table in the PDF
report):

- **Geometry:** `aspect_ratio` (bbox width/height), `centroid_x`/
  `centroid_y` (ink's center of mass, normalized), `pca_orientation_deg`
  (dominant stroke angle via PCA on ink pixel coordinates), `ink_ratio`
  (fraction of the canvas that's ink — a density measure).
- **Stroke:** `stroke_width_mean`/`stroke_width_std`/`stroke_width_cv`
  (via a distance transform on the binary mask — how thick the pen line
  is and how consistent that thickness is, a pen-pressure proxy),
  `skeleton_length` (total stroke path length), `total_stroke_length`.
- **Topology** (computed on the skeleton, via 8-neighbor connectivity):
  `endpoints_count` (pixels with exactly 1 neighbor — where a stroke
  starts or ends, i.e. pen lifts), `junctions_count` (pixels with 3+
  neighbors — where strokes cross), `components_count` (separate
  connected ink blobs — how many times the pen was lifted),
  `contours_count`/`holes_count` (via `cv2.findContours` with hierarchy —
  outer shapes and enclosed loops, e.g. the loop in a cursive "l").
- **Shape:** `contour_perimeter`, and `hu_1` through `hu_7` — the seven
  classical Hu moment invariants (rotation/scale/translation-invariant
  shape descriptors), each log-transformed
  (`-sign(v) * log10(|v| + eps)`) for numerical stability, per the
  standard convention for this feature family.

**Important, hard-won caveat documented here because it drives real
design decisions elsewhere:** `hu_5`, `hu_6`, and `hu_7` are
confirmed (Bug #35) to be noise-sensitive enough on real photographed
(non-scanned) signatures that they can flip SIGN between two genuine
photos of the same signature. They are computed and reported here
(visible in the metrics table) but excluded from the automated fusion
score in `similarity.py` — see that file's section below.

### `app/core/align.py` — find the best geometric correspondence between two images

**Entry point:** `align(query: Preprocessed, reference: Preprocessed) ->
(warped_gray, warped_binary, aligned_ok, best_iou)`. Called once per
query-reference PAIR (so for an N-reference profile, N times per
verification).

A SMALL, bounded set of candidate transforms is tried — not a general
optical-flow search, because real signatures are rigid and don't warp
locally:

1. `identity` — no transform at all.
2. `_centroid_translation` — align by center of mass only.
3. `_pca_similarity` — align by matching each image's principal axis
   (rotation + uniform scale), clipped to a maximum rotation
   (`_clip_rotation`, `_MAX_ROTATION_DEG`) so a bad PCA estimate on an
   asymmetric signature can't produce a wild rotation.
4. `_orb_affine` — ORB feature-point matching + RANSAC, when enough
   keypoints exist (works even when PCA alone is unreliable).
5. `_ecc_refine` — a final sub-pixel refinement pass (Enhanced
   Correlation Coefficient maximization) on top of whichever candidate
   above won.

Each candidate is scored by the ACTUAL ink-mask IoU it achieves
(`_ink_iou` — non-tolerant, raw pixel overlap, used only for internal
candidate ranking); the winner is whichever candidate produces the
highest IoU, and identity is always in the running, so alignment can
never make things WORSE than doing nothing. `best_iou` (the winning
candidate's raw IoU) and `aligned_ok` (was a non-identity transform
actually better than identity) are both returned — `engine.py` uses
`best_iou` for a real-overlap-quality decision and no longer conflates
it with `aligned_ok` (that conflation was Bug #36).

### `app/core/similarity.py` — the 9 individual similarity measurements

This file has NO knowledge of "profiles" or "verdicts" — it's a pure
toolbox of comparison functions, each producing one number in `[0, 1]`
(mostly), called by `engine.py::_score_against_ref` once per
query-reference pair:

| function | produces fusion component | compares |
|---|---|---|
| `ssim(gray_a, gray_b)` | `ssim` | structural similarity of aligned grayscale |
| `ink_iou(bin_a, bin_b, tolerance_px=3)` | `iou` | ink-mask overlap, dilated by 3px first (Bug #37 — thin-stroke IoU is hypersensitive to sub-pixel misalignment without this) |
| `cosine_sim(hog_a, hog_b)` | `hog` | gradient/edge-orientation histograms |
| `cosine_sim(lbp_a, lbp_b)` | `lbp` | local binary pattern texture |
| `hu_pairwise_sim(query, reference, std)` | `hu` | Hu moments 1-4 ONLY (`HU_FUSION_KEYS`) via the same natural-variation-scaled z-score as below — moments 5-7 excluded (Bug #35) |
| `dtw_similarity(bin_a, bin_b)` | `dtw` | dynamic-time-warped stroke path (via `contour_dtw` on 128 resampled contour points) |
| `pairwise_scalar_sim(query, reference, std, exclude=...)` | `scalar_profile` | ~17 general scalar features (all of features.py's catalogue except the 5 construction/topology keys and the 7 Hu keys) |
| `compare_zones(...)` (in `zoning.py`, not this file) | `zonal` | localized region-by-region comparison |
| `pairwise_scalar_sim(..., keys=CONSTRUCTION_KEYS)` | `construction` | the 5 topology counts (`endpoints_count`, `junctions_count`, `components_count`, `holes_count`, `contours_count`) |

**The core formula behind `hu`, `scalar_profile`, and `construction`**
(all three call `pairwise_scalar_sim`, just with different key subsets):

```
z = clip((query_value - reference_value) / natural_variation_scale, ±6)
similarity = exp(-|z|)
```

averaged across the relevant keys. `natural_variation_scale` is NOT a
fixed number — it's `std`, passed in from `engine.py`, computed fresh
per PROFILE from that signer's own enrolled references (see the
`_relearn` section below). This is the single most important formula in
the whole system for understanding "why did this score come out the way
it did" — it asks, per feature, "how far is the query from this specific
reference, relative to how much THIS SIGNER's own genuine samples
normally vary on that exact feature" — not a fixed, one-size-fits-all
tolerance.

### `app/core/zoning.py` — catch a difference a whole-signature average would hide

**Entry point:** `compare_zones(query_gray, query_binary, ref_gray,
ref_binary) -> dict` with `aggregate_score` (the `zonal` fusion
component) and `flagged_zones` (which specific region diverged, used for
the heatmap and the "point to where the anomaly is" requirement). Splits
the canvas into content-adaptive regions (`_adaptive_splits`, based on
where the ink mass actually is, not a fixed grid), compares each region
independently, and separately extracts skeleton corner points
per-region (`_skeleton_corner_points`) to report a LOCAL
stroke-direction-change count mismatch — this is what produces lines
like "this region shows 12 corners in the query vs 5 in the reference"
in the report.

### `app/core/engine.py` — where every signal gets fused into a verdict

This is the file that answers "what does the system actually DO," and
the one to read line-by-line if you want to trace exactly how a specific
score was produced. Its class, `VerificationEngine`, has exactly 3
public methods a caller uses (`enroll`, `verify`, `feedback`) plus one
public helper (`_profile_stats`, called by the API for the profile
detail view) and 3 private helpers doing the real work.

#### `enroll(profile_id, images) -> stats`

For each image: `preprocess.preprocess()` then `features.extract()`,
saved as a new reference record (`{"image": filename, "features":
{...}}`) on the profile. Then calls `_relearn()` to (re)calibrate
everything the profile knows about itself. This is also what gets called
again, with just the ONE new image, when a TTC-promoted feedback case
adds a reference (see `feedback` below) — enrollment isn't a one-time
event, it's "add reference(s), then recalibrate," and can happen
repeatedly over a profile's life.

#### `_relearn(profile, profile_id)` — the profile's self-calibration step

Runs every time the reference set changes (enroll, or a promoted
feedback event). In order:

1. **Per-feature natural-variation std** (`scalar_std`, Bug #34): for
   each of the ~22 scalar features, computes the sample standard
   deviation across the profile's own enrolled references (with
   Bessel's correction, `ddof=1`), then blends it toward a relative
   prior (`_FEATURE_PRIOR_REL = 0.22`, Bug #49) using a pair-count-based
   blend factor that fades the prior out as more real reference pairs
   accumulate (`min(1, n_pairs / _RELIABLE_PAIR_COUNT)`, `
   _RELIABLE_PAIR_COUNT = 6`). This exists because, with very few
   samples, the raw sample std is proven unreliable (with exactly 2
   references, it's mathematically forced to equal half their absolute
   difference, regardless of what those values are — Bug #34's core
   finding). This `std` is what every `pairwise_scalar_sim` call
   throughout the rest of the system uses as its tolerance scale.
2. **Pairwise intra-reference comparison**: every pair of the profile's
   OWN enrolled references is compared against each other via
   `_score_against_ref` (treating one as "query," the other as
   "reference," symmetric conceptually) — this produces `intra_scores`
   (used for threshold calibration, next) AND (v8, Bug #54) the full
   per-COMPONENT breakdown for each pair.
3. **Per-profile adaptive weighting** (v8, Bug #54): for each of the 9
   fusion components, averages that component's value across all the
   pairwise intra-reference comparisons above — a component that stays
   high and consistent between THIS signer's own genuine samples is
   trustworthy for them; one that swings around even between their own
   confirmed-genuine samples is noisy for them. This per-profile
   "reliability" score multiplies into the global weight vector
   (`_HAND_GUESSED_WEIGHTS`, or a population-calibrated file if present
   — see `training/calibrate_weights.py` below), renormalized to sum to
   1, and blended toward the global weights using the same pair-count
   fade as step 1 (needs 3+ references / 2+ pairs to say anything —
   with exactly 2 references, falls through to the global weights
   unchanged). Stored as `profile["stats"]["weights"]` — this is what
   `_score_against_ref` reads for every future verification against
   this profile.
4. **Threshold calibration**: `threshold = mean(intra_scores) - 2.0 *
   std_used`, where `std_used` is the SAME shrinkage-blended-toward-a-
   prior approach as step 1, applied to the intra_scores themselves
   (Bug #13's original fix) — deliberately conservative (`mean - 2*std`
   approximates a ~95%-coverage lower bound), clamped to
   `[MIN_THRESHOLD, 0.92]`. Stored as both `threshold` (the LIVE value,
   which online feedback can nudge — see `feedback` below) and
   `calibrated_threshold` (the fresh, purely-statistical value, used to
   BOUND how far feedback can nudge the live one, Bug #42).

#### `_score_against_ref(q_pre, q_feat, r_pre, r_feat, mean, std, profile, case_dir) -> (score, components, zscores, warped_images, aligned_ok, zonal_result, best_iou)`

The single-pair comparison at the heart of everything — called once per
query-reference pair, both during real verification and during
`_relearn`'s intra-reference comparisons. In order:

1. `align.align(q_pre, r_pre)` → the best warp found, plus `best_iou`.
2. Computes each of the 9 components (table above) using the ALIGNED
   query image where appropriate (`ssim`, `iou`, `zonal`) or the
   original per-image features where appropriate (`hu`,
   `scalar_profile`, `construction`, `dtw`, `hog`, `lbp` — computed from
   each image's own unwarped `features.extract()` output, since these
   are meant to be reference-specific comparisons of inherently-
   invariant descriptors, not pixel-warp-dependent).
3. **Pixel-term reliability penalty** (Bug #36): if `best_iou <
   _MIN_RELIABLE_IOU (0.30)`, the `ssim` and `iou` components are
   multiplied by 0.9 — this only fires when overlap is genuinely poor in
   an absolute sense, never merely because identity happened to already
   be the best available transform (which is the BEST case, not a
   penalty case).
4. Weighted sum: `score = Σ weight[k] * component[k]` using
   `profile["stats"]["weights"]` (the per-profile adaptive vector from
   `_relearn`, or the global default).
5. Returns everything, including the raw `components` dict (used both
   for the fused score AND, separately, for the SHAP explanation and the
   human-readable metrics table).

#### `verify(profile_id, query) -> case`

1. `preprocess` + `features.extract` on the query (once).
2. Loop `_score_against_ref` over EVERY enrolled reference — this is the
   literal implementation of "for 1 verification image, all N sample
   images pass through separate analysis": there is no shortcut,
   sampling, or early-exit; an N-reference profile always does N full
   comparisons, and the resulting list length is recorded verbatim as
   `n_references_compared` in the returned case, alongside the full
   per-reference breakdown in `reference_scores` (every one of the 9
   components, for every single reference, for full transparency).
3. **Combining N individual scores into one case score**: `fused_score`
   is a blend of `best_ref` (the single highest-scoring reference — "is
   there at least one enrolled sample this clearly resembles") and
   `prototype`/mean-across-all-references ("does it resemble this
   signer's samples on average") — roughly a 50/50 blend (see the exact
   formula in `verify`'s body). This is deliberate, not incidental:
   averaging all references into one prototype BEFORE comparison would
   blur exactly the kind of natural stroke-level variation the
   construction/hu components depend on (see `docs/ALGORITHM_DESIGN.md`
   §6) — comparing against each individually and combining after
   preserves it.
4. **Verdict banding**: `GENUINE` if `fused_score >= threshold +
   INCONCLUSIVE_BAND`; `FORGED` if `fused_score < threshold -
   INCONCLUSIVE_BAND`; `INCONCLUSIVE` otherwise (`INCONCLUSIVE_BAND =
   0.07`). A third verdict exists deliberately — see
   `docs/ALGORITHM_DESIGN.md` §7 for why a forced binary call on a
   borderline score is treated as worse than an honest "not sure."
5. `_edge_case_review(...)` — a SEPARATE pass that can downgrade
   GENUINE/FORGED to INCONCLUSIVE (never upgrade) if the evidence
   doesn't actually support confidence in that verdict — see next
   section. Can also attach `informational_flags` that don't change the
   verdict but are surfaced for a human reviewer.
6. `zoning`/`explain` calls to generate the evidence images
   (heatmap, overlay, zone grid, skeleton comparison, agreement chart),
   saved to `storage.case_dir(case_id)`.
7. `storage.save_case(case)` — persists the full case (including every
   per-reference component breakdown) to `data/reports/<case_id>/case.json`.

#### `_edge_case_review(verdict, threshold, per_ref_scores, per_ref_best_iou) -> dict`

Three independent checks, each of which can add to `flags` (which
downgrade the verdict) — but see Bug #51's fix: none of them fire when
the fused verdict is already `INCONCLUSIVE` (nothing to downgrade
further), and the alignment-quality check specifically (#3) only counts
toward a downgrade when it's ALSO paired with genuine disagreement:

1. **Opposing evidence** — do more than 30% of individual references
   land on the OPPOSITE side of the threshold from the fused verdict
   (only checked with 3+ references; with 2, "opposing" and
   "disagreement" collapse into the same question, check #2 already
   covers it).
2. **High disagreement** — is the standard deviation across individual
   reference scores large relative to how far the fused score sits from
   the threshold (i.e., is this verdict sensitive to WHICH reference
   happened to be used).
3. **Alignment reliability** — did spatial alignment find at least SOME
   overlap (`best_iou >= _MIN_SANE_IOU = 0.08`, a deliberately very low
   bar) against most references. Only degrades the verdict when
   `agree_frac < 1.0` (i.e., references don't already unanimously
   agree) — a unanimous verdict despite poor overlap is treated as a
   real, well-supported measurement (a real difference correctly
   producing low overlap), not evidence of an unreliable one (Bug #51).

#### `feedback(case_id, correct, add_as_reference, remark) -> stats`

The online-learning ("test-time calibration") loop:

1. Always records the feedback event (`correct`, `remark`, timestamp) on
   the profile — this happens regardless of what else below does or
   doesn't fire.
2. **Threshold nudge**: `FEEDBACK_THRESHOLD_STEP` (a small fixed step)
   applied to the live threshold, direction depending on whether the
   examiner confirmed or disputed the verdict — but bounded to within
   ±0.10 of the last STATISTICALLY CALIBRATED threshold
   (`calibrated_threshold` from `_relearn`), so repeated feedback can't
   walk the threshold arbitrarily far from what the actual enrolled
   evidence supports (Bug #42).
3. **Promotion to reference**: if the examiner confirms a GENUINE
   verdict AND opts in, the query image can be added as a new enrolled
   reference (up to `MAX_REFERENCES`) — but only if it's not a
   near-duplicate of an already-enrolled sample (`>= 0.97` similarity to
   an existing reference skips promotion, Bug #41, since a near-
   duplicate adds no real diversity and can actively corrupt the
   natural-variation estimates it's supposed to improve). A skipped
   promotion still records the feedback and returns
   `promotion_skipped_reason` explaining why. A real promotion triggers
   a fresh `_relearn()` — this is "training happening for this profile":
   every confirmed real-world verification can make the profile's own
   calibration (std, adaptive weights, threshold) measurably better,
   with guards against the specific ways that could go wrong.

### `app/core/shap_explain.py` — real Shapley-value attribution

`kernel_shap(query, mean, std, ...)` computes genuine KernelSHAP values
over the classical scalar features (not an approximation of a neural
network — there's no neural network in the decision path to
approximate). This answers "how much did each individual measurement
push the score toward or away from a match," which is what populates the
"SHAP attribution" table in the PDF report. `top_shap_findings` turns
the raw per-feature SHAP values into the human-readable ranked findings
list.

### `app/core/explain.py` — turn numbers into pictures

Pure rendering — every function here takes already-computed arrays/
scores and produces a saved PNG: `heatmap_overlay` (where the aligned
images diverge), `alignment_overlay`, `skeleton_compare`,
`annotated_regions` (draws the flagged-zone boxes with their labels),
`zone_grid_overlay`, `reference_agreement_chart` (the per-reference score
bar chart). None of these functions influence the verdict — they run
strictly after it's already decided, purely for human/report
consumption.

### `app/core/signet.py` — optional CNN-based signal (only when PyTorch is present)

A small Siamese-style CNN trained fresh, from scratch, on ONLY the
current profile's own enrolled references (`train_profile_signet`) — no
pretraining, no external dataset, consistent with the project's core
"no external dataset" invariant. `embedding_similarity` and `grad_cam`
provide an additional similarity signal and a saliency map WHEN PyTorch
is installed; `available()` lets every caller check this gracefully and
skip it entirely otherwise — this is enrichment, never a dependency the
core verdict relies on.

### `app/core/storage.py` — the entire filesystem contract

Every read/write of `data/profiles/<id>/profile.json`,
`data/reports/<id>/case.json`, and the associated images goes through
here — `create_profile`, `load_profile`, `save_profile`,
`find_profiles_by_name`, `delete_profile`, `case_dir`, `save_case`,
`load_case`, `list_cases`. No other file touches these paths directly;
this is intentional, so the on-disk format can change in one place.

---

## 2. The LLM layer (`app/llm/`) — narration only, never the verdict

**Load-bearing fact, worth repeating because it's been directly tested**:
the classical engine above produces a complete, final verdict with no
dependency on the LLM at all. The LLM only runs AFTER `verify()` has
already returned, when a report is requested, and its JSON output is
independently schema-validated (`report.py`'s pydantic `Examination`
model) before use — if the LLM is unavailable, times out, or returns
something that doesn't validate, `report.py` falls back to a
deterministic, template-based report using the exact same underlying
numbers, and the verdict shown is identical either way.

### `app/device.py` — decide if/which model this machine can run

`detect_device()` checks combined RAM+VRAM against `MIN_VIABLE_TOTAL_GB`
for the single supported `MODEL_ID`
(`google/gemma-4-E2B-it-qat-q4_0-gguf`, v8) — no tiers, exactly one
model or none. `model_present_on_disk()` checks the actual on-disk file
size against `MODEL_PRESENT_MIN_GB` so a manually-copied model is
recognized without needing to go through the app's own downloader.

### `app/llm/manager.py` — load the model, generate one completion

`VisionLLMManager._load()`: picks `n_gpu_layers` based on a VRAM-vs-
model-size estimate (`_KV_CACHE_OVERHEAD_GB` reserve + safety margin,
Bug #44) rather than just "is a GPU present" — skips straight to CPU
with a persisted, human-readable reason (`gpu_skip_reason`, Bug #46)
when the estimate says GPU offload would be doomed. The actual
construction call runs under a hard wall-clock timeout
(`_run_with_timeout`, Bug #43) so a genuine native-library hang produces
an actionable error instead of an indefinite silent "loading" state, with
a background reconciliation path in case the load finishes late anyway.
`generate(images, prompt, system)` converts image file PATHS to proper
`file://` URIs via `pathlib.Path.as_uri()` (Bug #48 — cross-platform;
naive string concatenation is not a valid URI on Windows) and runs one
multimodal chat completion.

### `app/llm/prompts.py` — what the LLM is actually told

`build_user_prompt(case)` assembles the EXACT structured JSON the LLM
sees: `engine_verdict`, `fused_score`, `decision_threshold`,
`component_similarities` (all 9, so the LLM can reference "the topology
count was the main driver," e.g.), `most_deviant_features` (from the
z-score reporting table), and enrollment metadata. `SYSTEM_PROMPT`
defines the persona ("Dr. Sigma," a conservative forensic examiner) and
the strict JSON-only, no-verdict-authority contract.

### `app/llm/report.py` — orchestrate, validate, and gracefully degrade

`generate_report(case_id, use_llm)`: tries the LLM up to a few times
(`LLM examination attempt N/3` in the logs), validates its JSON against
`Examination` (a pydantic model — malformed JSON, an out-of-range region
coordinate, a missing field, all get rejected here before ever reaching
a user), and falls back to `_template_examination()` (a fully
deterministic, non-LLM report built from the same case data) on any
failure. Either path produces the SAME `Examination` shape, so the PDF/
HTML rendering code downstream never needs to know which one it got.

### `install.py` — get the model and a working backend onto this machine

`_llm_backend()` decides `cpu` vs `cu121` vs a Metal/Vulkan variant based
on the detected OS/GPU. As of v8 (Bug #52), before committing to the
`cu121` GPU wheel, `_check_cuda_runtime_ready()` runs a real pre-flight
check — parses `nvidia-smi`'s own reported CUDA version against the
wheel's actual floor, and (on Windows) checks for the Microsoft Visual
C++ Redistributable's DLL — skipping straight to the CPU wheel with an
actionable message if either looks insufficient, rather than downloading
~450MB toward a wheel that's going to fail to load anyway.

---

## 3. Two complete request traces

### Enrolling a new profile with 5 reference photos

```
POST /api/profiles                     → storage.create_profile(name)
POST /api/profiles/{id}/enroll (5 imgs)→ VerificationEngine.enroll(id, [5 bytes])
    for each of the 5 images:
        preprocess.preprocess(img)     → Preprocessed (illumination-corrected,
                                          de-ruled, cropped, canvas-normalized)
        features.extract(pre)          → {"scalars": {...22 numbers...}, "vectors": {...}}
        → saved as a new reference record on the profile
    VerificationEngine._relearn(profile)
        → per-feature natural-variation std (from all 5 refs' own scalars)
        → C(5,2) = 10 pairwise intra-reference comparisons, each running
          the FULL _score_against_ref pipeline (align + all 9 components)
        → per-profile adaptive weights (from those 10 pairs' component
          breakdowns) -- blend factor min(1, 10/6) = 1.0, i.e. fully
          trusted at 5 references
        → threshold = mean(10 intra scores) - 2*std(10 intra scores)
    storage.save_profile(profile)      → data/profiles/{id}/profile.json
```

### Verifying one new signature against those 5 references

```
POST /api/verify (profile_id, 1 img)   → VerificationEngine.verify(id, bytes)
    preprocess.preprocess(query)       → Preprocessed
    features.extract(query_pre)        → query's own scalars/vectors
    for EACH of the 5 enrolled references (never a subset):
        align.align(query_pre, ref_pre)          → best warp, best_iou
        similarity.ssim/ink_iou/cosine_sim/...   → 9 raw component values
        zoning.compare_zones(...)                → zonal score + flagged regions
        weighted sum using profile's adaptive weights → this reference's score
    → 5 individual scores, 5 full component breakdowns (all kept, all
      returned in the case for transparency -- reference_scores[])
    fused_score = blend(best of the 5, mean of the 5)
    verdict = band(fused_score, threshold)        → GENUINE/FORGED/INCONCLUSIVE
    _edge_case_review(...)                        → possible downgrade + flags
    zoning/explain.* → heatmap.png, overlay.png, zone_grid.png, skeleton.png, agreement.png
    storage.save_case(case)            → data/reports/{case_id}/case.json
                                          (contains every single one of the
                                          5 references' full 9-component
                                          breakdown, not just the winner)
GET /api/reports/{case_id}             → app.llm.report.generate_report()
    tries LLM (Dr. Sigma persona, sees the fused case JSON + 3 evidence
    images) → validates JSON → PDF/HTML rendering
    (falls back to template on any LLM failure -- verdict unaffected)
```

---

## 4. Where to look for what

| You want to know... | Read |
|---|---|
| Why is a specific score what it is | `engine.py::_score_against_ref`, then the specific `similarity.py` function for the component in question |
| Why did the verdict get downgraded to INCONCLUSIVE | `engine.py::_edge_case_review` |
| What does a specific number in the report mean | `features.py`'s scalar catalogue (§1 above) |
| Why is the threshold what it is | `engine.py::_relearn`, the threshold-calibration step |
| Why does this profile trust `construction` more than `hu` | `profile["stats"]["component_reliability"]` (Bug #54) and `_relearn`'s adaptive-weighting step |
| Did the LLM decide this verdict | No. Read §2 above — it narrates, never decides |
| Why does the design use THIS algorithm and not another | `docs/ALGORITHM_DESIGN.md`, not this file |
| What bug was fixed, when, and why | `PROJECT_MEMORY_v8.md` |
