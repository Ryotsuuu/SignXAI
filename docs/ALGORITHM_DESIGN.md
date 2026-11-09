# SignXAI — Algorithm & Design Decisions

**What this document is:** a synchronous reference, organized by design
decision, for every algorithm and statistical choice in the classical
verification engine — what it is, why it was chosen over the obvious
alternatives, and what evidence (literature or this project's own
self-run tests) backs it. `PROJECT_MEMORY_v8.md` is the chronological
"what happened, in what order, because of what bug report" record; this
file is the "here is the whole design, as it stands today" one. Where a
decision was changed because a bug was found, this doc says so and
points at the PROJECT_MEMORY bug number for the full investigation —
it does not repeat that investigation here.

**What this document is not:** a promise that these are the *right*
numbers for every signer or every dataset, nor a step-by-step trace of
what the code does — for that, see `docs/SYSTEM_WALKTHROUGH.md`, which
names exact files, functions, and call order from photo upload to
verdict to report. This document explains *why* each choice was made;
that one explains *what happens and where*. Every empirically-chosen
constant below is marked as such, together with what it was actually
tested against (almost entirely the project's own 5-image, 1-signer
repro set plus the earlier CEDAR sweep — see `docs/TRAINING.md`). Real
multi-signer calibration remains the single highest-leverage way to
validate or correct any of this further (`PROJECT_MEMORY_v8.md` §5,
Phase 2).

---

## 1. Overall approach: why classical, interpretable features instead of a trained embedding network

**Decision:** every feature the engine computes (Hu moments, HOG, LBP,
stroke topology counts, ink density, etc.) is a classical, hand-defined
image/shape descriptor — not a learned embedding from a neural network
trained on signature data.

**Why:** three concrete reasons, all load-bearing for this specific
project's stated goal (explainable, per-signer verification with no
external dataset):

1. **No pretrained-model dependency.** A deep embedding network (e.g. a
   Siamese CNN) needs to have been trained on a large labelled
   signature dataset to produce a meaningful embedding space at all —
   this project has explicitly chosen not to depend on one (see
   `PROJECT_MEMORY_v8.md` §1's "no pretraining" invariant). Classical
   features are computed fresh from whatever samples a specific signer
   enrols, with no dependency on what any other dataset looked like.
2. **Genuine explainability, not post-hoc approximation.** Every
   component of the fused score (ssim, iou, hu, dtw, ...) is a named,
   independently-meaningful number a human reviewer can inspect and
   dispute ("the topology count is off because of these 3 extra
   junctions here") — this is what SHAP/Grad-CAM-style explanation
   normally has to reconstruct after the fact from an opaque embedding.
   Here, the "explanation" IS the score's own decomposition; the SHAP
   layer (`shap_explain.py`) computes real Shapley values over these
   same interpretable features rather than approximating a black box.
3. **Script/language independence.** None of these features do OCR or
   character recognition — a signature is treated purely as an ink
   shape (stroke curvature, topology, moments, density). This is
   architecturally robust to script (Latin, Devanagari, Han, Tamil,
   ...) in a way a network trained mostly on Latin-script handwriting
   would not automatically be — see `PROJECT_MEMORY_v8.md` §8 for the
   fuller discussion and its honest caveat (the *fusion weights* are
   only as validated as the data they were calibrated on, even though
   the features themselves are script-agnostic).

**Trade-off, stated honestly:** classical features are unlikely to ever
match a well-trained deep embedding's raw discriminative power on a
large, diverse dataset. This is a deliberate choice for THIS project's
constraints (no training data, explainability required as a first-class
requirement per the stated problem statement), not a claim that it's
the best approach unconditionally.

---

## 2. Preprocessing pipeline

### 2.1 Grayscale + Otsu binarization

Standard, well-established choice for ink/background separation on a
photographed (not scanned) document: Otsu's method picks a threshold
that minimizes intra-class variance, adapting automatically to lighting
differences across photos rather than needing a fixed threshold tuned
per photo.

### 2.2 Ruled-line removal (v7 Bug #40, redesigned in v7.1 Bug #47)

**Problem this solves:** a signature photographed on ruled/lined
notebook paper has page-rule lines that survive binarization as "ink,"
corrupting every downstream shape/topology measurement.

**Algorithm:** probabilistic Hough transform (`cv2.HoughLinesP`) over
the full-frame binary mask, before cropping, finds long straight line
segments. Two conditions must BOTH hold before any detected line is
removed:

1. **A genuine repeating pattern.** After deduplicating multiple Hough
   detections of the same physical line (grouped by matching angle AND
   position — see below for why this two-step dedup is necessary), at
   least `_MIN_PARALLEL_LINES = 2` distinct lines must share the same
   angle (within `_PARALLEL_ANGLE_TOL_DEG`). A single long straight
   line alone is never removed.
2. **Frame-spanning extent.** Each candidate line's actual measured
   extent must reach at least `_MIN_FRAME_SPAN_FRAC = 0.85` of the
   longest a straight line at that exact angle could possibly be while
   still fitting inside the photographed frame. This is the DECISIVE
   check (see below for why condition 1 alone isn't enough).

**Why angle is checked with NO axis assumption:** a first version
required lines to be near absolute horizontal/vertical (reasoning that
ordinary camera tilt is small). Tested directly against a real photo
rotated ~15-20 degrees and it missed the ruled lines entirely — a
rotated photo's page rules are rotated right along with everything
else, well outside a small axis tolerance. Dropping the axis
assumption and clustering by RELATIVE angle (lines parallel to each
other, at any absolute orientation) fixed this and is strictly more
general.

**Why a two-stage dedup-then-group process, not one grouping pass:**
`cv2.HoughLinesP` frequently returns several overlapping/adjacent
segment detections for a single physical thick line. A naive one-pass
grouping-by-angle miscounts these duplicate detections of ONE line as
multiple "parallel lines," triggering false removal. Verified directly
with a synthetic single straight stroke: a one-pass angle grouping
incorrectly removed it at every angle tested. Fixed by first merging
detections that share both angle AND position (a real second line, by
contrast, shares only the angle — it's spaced apart on the page) into
one physical line, THEN grouping by angle only across the deduplicated
lines.

**Why condition 2 (frame-span) exists, and why condition 1 alone is
provably not enough:** tested the angle+count signal against this
project's own real signature photos and found it falsely fired — this
specific user's signature style genuinely includes a deliberate
double-underline flourish, which is, by angle and count alone,
literally indistinguishable from 2 ruled lines. No amount of
angle-tolerance tuning resolves this, because the ambiguity is real,
not a measurement error: two real hand-drawn parallel strokes and two
real ruled lines can have the same angle and the same count. The
distinguishing fact that DOES separate them: a photographed page is
always bigger than the signature drawn on it, so real page rules
almost always span the full frame edge-to-edge (with only whatever
margin the paper itself has), while a signature's own flourish,
however long relative to its other strokes, essentially never reaches
the full frame's extent — there is always camera margin around a
photographed signature. This is an absolute geometric check against the
frame itself, not a relative comparison between candidate lines, which
is why it succeeds where tuning the relative signal further could not.
Verified: with this filter, this project's own repro images (including
the ones with the genuine double-underline) all show zero false
removal, while the real ruled-paper test image still has its lines
correctly detected and removed.

**Why `_LINE_LENGTH_FRAC` is 70% of the frame's shorter dimension for
Hough's own `minLineLength`** (a separate, earlier, more lenient gate
than the 85% frame-span check above): this just controls what Hough
considers "long enough to be a line candidate" at all, before either of
the two real decision criteria above are applied — kept generous so a
real ruled line is never missed by Hough itself, with the two
downstream criteria doing the actual discriminating work.

### 2.3 Speck removal and distant-outlier removal (Bug #26)

Two SEPARATE filters, deliberately not merged into one: `_remove_specks`
drops components below a minimum pixel-area (photographic noise,
dust, JPEG artifacts) regardless of position; `_remove_distant_outliers`
drops components that are spatially far from the main ink cluster
regardless of size (a stray pen mark, a staple hole, a fold shadow far
from the signature itself). A component that is BOTH large and distant
(like a ruled line running through the middle of the frame) is caught
by neither — which is exactly why ruled-line removal (2.2) had to be a
third, distinct mechanism rather than a tuning adjustment to either of
these two.

### 2.4 Crop-to-content + fixed-canvas normalization

The ink's bounding box (after the filters above) is cropped to, then
fitted onto a fixed `CANVAS_W x CANVAS_H = 512x256` canvas, aspect-ratio
preserved with padding. This is what allows two photographs taken at
different distances/crops/resolutions to be compared at all — every
downstream feature and comparison operates in this normalized space,
never on the original photo's raw pixel dimensions.

---

## 3. Alignment (`align.py`)

**Problem:** two photographs of the same signature are never pixel-
registered — different crop margins, slight rotation, and scale
differences from how the photo was framed. Pixel-level comparison
(SSIM, IoU) is meaningless without first finding the best available
geometric correspondence.

**Algorithm:** a small set of CANDIDATE transforms is tried
(identity, translation-only via centroid alignment, and a
rotation+scale+translation fit via PCA-axis matching) and the one that
maximizes ink IoU against the reference is kept. This is a deliberately
SIMPLE, bounded search — not a general optical-flow or feature-matching
registration (e.g. full ORB+RANSAC homography) — because signatures are
mostly rigid, small, and don't warp locally in ways that would justify
that complexity, and a bounded candidate set is easy to reason about
and debug (every candidate's resulting IoU is inspectable).

**What "alignment succeeded" means, and the bug in how that used to be
used (v7, Bug #36):** the boolean `aligned_ok` (did a non-identity
transform beat identity) answers "was a transform needed," not "is the
resulting overlap trustworthy." Those are different questions — a
query that's ALREADY perfectly registered needs no transform (identity
wins, `aligned_ok=False`) and that's the *best* case, not a failure.
v7 fixed this by keying the actual pixel-term-reliability decision off
the achieved overlap quality (`best_iou`, an absolute number) rather
than off which candidate won — see §5.4 below and
`PROJECT_MEMORY_v8.md` Bug #36 for the full story, including a real
regression an early fix attempt caused and how it was caught.

---

## 4. Per-feature similarity: the pairwise, natural-variation-scaled z-score (v7, Bug #34)

This is the single most consequential design decision fixed in v7 and
worth explaining in full, since it now underlies three of the nine
fusion components (`scalar_profile`, `construction`, `hu`).

**The question being asked:** "how different is this query from THIS
SPECIFIC reference, on this specific feature — relative to how much
this signer's own genuine samples naturally vary on that feature?"

**The formula:**
```
z = (query_value - reference_value) / natural_variation_scale
similarity = exp(-|z|)          # capped at |z| <= 6
```
averaged across the relevant feature keys.

**Why this exact formula, term by term:**

- **`query - reference`, not `query - profile_mean`.** The score is
  being computed once per enrolled reference (the engine compares a
  query against EVERY reference, then combines — see §6). Each of
  those per-reference comparisons should ask "how well does this match
  THIS ONE reference," not a population-average question that's
  identical across every reference — the latter defeats the entire
  point of comparing against every reference individually, and (see
  next point) has a serious independent problem with small samples.
  v6 used `query - profile_mean` here; fixing this was Bug #34.
- **Why the difference is SCALED by natural variation, not used raw.**
  A raw pixel-value or moment difference has no inherent sense of
  "big" or "small" — a difference of 0.5 might be enormous for a
  feature that's normally stable to 0.01, and trivial for one that
  swings by 3.0 between any two genuine signings of the same person.
  Dividing by an estimate of that person's own natural per-feature
  variation converts every feature onto a comparable "how many of this
  signer's own normal wobbles is this difference" scale — this is the
  same idea as a z-score in ordinary statistics, applied per-feature
  and per-signer rather than across a population of different people.
- **Why `exp(-|z|)` and not, say, a linear or step function.** A
  smoothly decaying function means a small deviation costs almost
  nothing (z=0.2 → similarity 0.82) while a large one costs a lot
  (z=3 → similarity 0.05) with no hard cliff at an arbitrary cutoff —
  appropriate when the "right" cutoff for any single feature isn't
  actually known precisely, which is true here (no large calibration
  dataset per signer, by design). The exponential's specific base
  (`e`, i.e. `exp`) is a convention, not a fitted choice — z=1
  (one unit of natural variation) is intentionally chosen to map to
  `1/e ≈ 0.37`, a soft "started to look different" signal rather than
  a sharp mid-point.
- **Why capped at `|z| <= 6`.** Purely numerical hygiene — without a
  cap, one wildly deviant feature (division by a near-zero natural-
  variation estimate) could produce an enormous z that, after `exp`,
  underflows to a value so small it behaves like a hard veto over the
  entire average. Capping bounds the worst any single feature can drag
  the average down by, consistent with the project's established
  "no single feature should veto everything" principle (first
  established for a different code path in an earlier bug fix,
  extended here — see `PROJECT_MEMORY_v8.md` Bug #35 for a concrete
  case, the Hu moments, where this mattered a great deal).

**Where "natural variation scale" itself comes from — the shrinkage
estimator (v7, Bug #34), and why it exists:**

With very few enrolled references (commonly just 2, per this project's
own onboarding flow), the OBVIOUS estimate — the raw sample standard
deviation across the enrolled references' own feature values — is
proven, not just suspected, to be unreliable: with exactly 2 samples,
`std([a, b])` is mathematically forced to equal exactly half `|a - b|`,
no matter what those two values actually are. This was directly
confirmed causing false-FORGED verdicts (`PROJECT_MEMORY_v8.md` Bug
#34's printed z=±1.000 evidence). The fix blends the raw sample
estimate toward a fixed relative prior, fading the prior out as more
real reference pairs accumulate:

```
blend = min(1, n_pairs / RELIABLE_PAIR_COUNT)     # RELIABLE_PAIR_COUNT = 6
std_used = blend * sample_std(ddof=1) + (1 - blend) * prior
prior = max(FEATURE_PRIOR_REL * |mean|, 0.01)      # FEATURE_PRIOR_REL = 0.22 (v7.2)
```

- `ddof=1` (Bessel's correction, i.e. dividing by n-1 not n) is the
  textbook-standard small-sample correction for variance estimation —
  used here because it's the correct thing to do, not a tuned choice.
- `FEATURE_PRIOR_REL = 0.22` (22% of a feature's own mean magnitude,
  raised from an original 0.18 in v7.2 — Bug #49, PROJECT_MEMORY_v8.md
  — after the original value proved too tight against a second real
  signer's data; a larger increase to 0.28 was tested and rejected
  after it measurably eroded discrimination against a known real
  forgery) is
  an EMPIRICALLY-CHOSEN constant — a documented, honest prior for "how
  much does an arbitrary geometric/shape feature normally wobble
  between two genuine signings of the same person," informed by the
  same repeated-genuine-score observations that originally justified
  an analogous (but separately-tuned) prior at the score-threshold
  level (`PROJECT_MEMORY_v8.md`, the older Bug #13). It has NOT been
  fit against a large multi-signer dataset — it's the best available
  evidence-grounded estimate given this project's actual test data,
  and is a clearly-labeled target for future recalibration once more
  signers' data is available.
- The blend construction itself (fade a prior out as `n_pairs` grows,
  rather than switching abruptly from "trust the prior" to "trust the
  data" at some cutoff) avoids a discontinuity at exactly
  `RELIABLE_PAIR_COUNT` pairs — a profile with 5 vs. 7 reference pairs
  shouldn't behave qualitatively differently.

---

## 5. The nine fusion components, and why each exists

The fused per-reference score is a fixed-weight sum of nine independently-
computed similarity components (`_HAND_GUESSED_WEIGHTS` in `engine.py`,
overridable by `training/calibrate_weights.py`'s data-fit weights if
present — see `docs/TRAINING.md`):

| component | weight | measures | why it's a SEPARATE component |
|---|---|---|---|
| `ssim` | 0.10 | structural similarity of aligned grayscale | tolerant of small local noise/lighting differences; complements pixel-exact IoU |
| `iou` | 0.16 | ink-mask overlap after alignment (tolerant, §5.4) | direct measure of "does the ink occupy the same space" |
| `hog` | 0.09 | gradient/edge-orientation histogram similarity | captures local stroke-direction texture independent of exact pixel position |
| `lbp` | 0.03 | local binary pattern texture similarity | fine-grained ink texture (pen pressure/type cues); low weight because it's the noisiest of the pixel-level cues |
| `hu` | 0.05 | shape-invariant moment similarity, 4 stable moments only (§5.3) | rotation/scale-invariant GLOBAL shape, independent of local texture or exact alignment — weight reduced in v7.2 (Bug #49), see below |
| `dtw` | 0.16 | dynamic-time-warped stroke-path similarity | sequence/ordering of the pen path, tolerant of speed/rhythm variation between signings |
| `scalar_profile` | 0.08 | pairwise z-score average over ~17 general scalar features (§4) | catches numeric geometric/stroke properties none of the pixel-level metrics directly measure (aspect ratio, density, slant, etc.) — weight reduced in v7.2 (Bug #49), see below |
| `zonal` | 0.11 | localized region-by-region comparison (§5.5) | catches a LOCAL difference a whole-signature average would dilute away |
| `construction` | 0.22 | pairwise z-score average over 5 topology counts (§5.2) | pen-lift/junction/loop PATTERN — a strong forgery signal, and empirically the most discriminative single family, hence the highest weight |

**v7.2 update (Bug #49, PROJECT_MEMORY_v8.md):** `hu` and
`scalar_profile` were found, across two independent real signers' data
(this project's own test signer AND a real user's own logged
measurements), to consistently underperform on confirmed-genuine
comparisons relative to the other seven components — dragging fused
scores down even when every pixel/topology signal unanimously agreed.
Weights were moderately reduced for these two and redistributed to
`dtw`, `zonal`, and `construction`, which have a consistent track
record across both datasets. This was DELIBERATELY a moderate change,
not an aggressive one: a larger reweighting draft was tested and
rejected after it was found to measurably hurt a DIFFERENT real
signer's data (this project's own repro set, where `construction` is
comparatively one of the weaker components) — see Bug #49's full
writeup for why a global weight vector cannot fully resolve this, and
Section 5's roadmap for the actual architectural fix (per-profile
adaptive weighting) this finding motivated.

**Why nine separate components instead of one combined score or a
smaller set:** each measures a genuinely different property (texture,
topology, global shape, pixel overlap, path sequence) that can fail
independently — a forgery can replicate overall shape while getting
pen-lift topology wrong, or vice versa. Keeping them separate, visible,
and independently weighted is also what makes the SHAP-based
explanation (`shap_explain.py`) meaningful: a reviewer can be told
*which* aspect of the signature drove a verdict, not just a single
opaque number.

**Where the weight VALUES come from:** originally hand-guessed (v1),
then optionally replaced by `training/calibrate_weights.py`'s logistic-
regression fit against a labelled dataset (CEDAR subset — see
`docs/TRAINING.md`) if that's been run and produced a valid file. The
hand-guessed values remain the default and the ones actually exercised
by every test in this document, since the calibration file was — until
v7's Bug #38 fix — never actually loadable due to a key-mismatch bug.

### 5.2 Construction (topology) features — why weighted highest (0.20)

Endpoints, junctions, connected components, holes, and outer contours
count structural properties of HOW the pen moved (how many times it
lifted, how many strokes crossed) — this is widely considered, in
forensic questioned-document literature, one of the harder things for a
forger to replicate under time pressure (a forger copies the visual
SHAPE but rarely reproduces the exact same pen-lift pattern). This
project's own empirical history agrees: `PROJECT_MEMORY_v8.md`'s Bug
#33 found construction counts to be the most reliable single family for
separating genuine from forged samples in the available test data,
which is why it carries the highest individual weight.

### 5.3 Hu moments — why only 4 of the 7 feed the fusion score (v7, Bug #35)

Hu's seven moment invariants are a classical (1962) family of rotation/
scale/translation-invariant shape descriptors. It's a well-documented
property of this family that higher-order moments (5, 6, and especially
7 — a designed skew/reflection indicator) are progressively more
sensitive to noise and small shape perturbations than the lower-order
ones. This project confirmed that sensitivity directly and concretely:
moments 5, 6, AND 7 each measured with a full SIGN INVERSION between
two of the user's own genuine, examiner-confirmed signature photos
(not a subtle drift — see `PROJECT_MEMORY_v8.md` Bug #35 for the
printed numbers). The 4 lower-order moments showed no such instability
and remained genuinely discriminative when tested against a real
forgery. Rather than trust literature alone, both claims (instability
of 5/6/7, stability and discriminative power of 1-4) were verified
against this project's own real test images before excluding 5/6/7 from
the automated score — they remain visible in the human-readable metrics
table for a reviewer's own judgment, just not weighted into the verdict.

### 5.4 Ink IoU tolerance — why dilated by 3px before comparison (v7, Bug #37)

Intersection-over-union on a raw, un-tolerant thin-stroke binary mask is
disproportionately sensitive to registration noise that has nothing to
do with signer identity: a 5-6px-wide pen stroke loses most of its
overlap from a 1-2px residual misalignment or a photograph-to-photograph
stroke-width difference (lighting, focus, JPEG compression) that
SSIM's local-neighborhood tolerance absorbs but pixel-exact IoU does
not. Dilating both masks by a small, fixed, symmetric radius before
computing IoU is a standard "tolerance band" technique for exactly this
class of problem in stroke/line-image comparison. `_IOU_TOLERANCE_PX=3`
is an empirically-chosen value (tested against the real repro set,
including confirming it does NOT rescue a real forgery's low IoU — see
Bug #37) rather than a value derived from a formula; a different
canvas resolution or stroke-width distribution could call for a
different number, which is why it's a single named constant in
`similarity.py`, not inlined.

The related question of WHEN a poor achieved IoU should count as
"unreliable evidence" (as opposed to real, correctly-scored dissimilarity)
uses a second, much lower, separately-justified constant
(`_MIN_SANE_IOU = 0.08` in `_edge_case_review`) — see
`PROJECT_MEMORY_v8.md` Bug #36 for why these had to be two different
numbers answering two different questions, after an early fix attempt
conflated them and caused a real regression.

### 5.5 Zonal (localized) comparison

Splits the aligned canvas into a grid of regions and compares each
region independently (`zoning.py`), rather than only ever comparing the
signature as one global blob. This exists specifically to catch a
LOCALIZED difference — a forger who nails the overall gestalt but
fumbles one specific letter or flourish — that a single whole-signature
average would dilute into invisibility. It's also the mechanism behind
the "point to where the anomaly is" requirement: `zoning.py`'s
`flagged_zones` output identifies WHICH region diverged and by how much,
surfaced in `case["zonal_analysis"]`, rather than a single undifferentiated
similarity number.

---

## 6. Multi-reference fusion: comparing against every enrolled reference, not just one

**Decision:** `verify()` scores the query against EVERY enrolled
reference independently (each producing its own 9-component fused
score, per §5), then combines those per-reference scores into one case-
level score (a blend of the best individual match and the average
across all references — see `engine.py::verify` for the exact
combination).

**Why not just compare against a single "prototype" signature (e.g. an
average of all enrolled samples):** a genuine signer's own samples
genuinely differ from each other (this project's own data confirms
real signatures can differ from a same-person reference by as much as
they differ from a different reference — see the self-check numbers in
`PROJECT_MEMORY_v8.md` §4). Averaging references into one prototype
BEFORE comparison would blur exactly the kind of natural stroke-level
detail (topology, exact moments) that the construction/hu components
depend on, and would make it impossible to ask "does this look like
ANY of this person's known signing styles" as opposed to only "does it
look like their average style." Comparing against each reference
individually, then combining, preserves that information.

---

## 7. Thresholding and the INCONCLUSIVE band

**Decision:** the case-level score is not compared to a single fixed
cutoff for GENUINE/FORGED. A calibrated threshold is computed per
profile (§7.1), and scores within a symmetric band around that
threshold (`INCONCLUSIVE_BAND = 0.07`) are reported as INCONCLUSIVE
rather than forced into GENUINE or FORGED.

**Why a third verdict exists at all, given the stated goal is "a
genuine correct remark":** a forced binary decision on a borderline
score, especially from a profile with only 2-3 enrolled references (the
realistic minimum this project's own onboarding asks for), is a
confident guess dressed up as certainty. `PROJECT_MEMORY_v8.md`'s own
history (the v3 "honest sweep") established this as the correct
philosophy, and v7's own findings reinforce it: a case that used to be
a confidently-WRONG FORGED verdict (Bug #34-#37's target case) is
better served by an honest "I'm not sure enough, please review" than by
tuning enough parameters to force a specific test image to a specific
answer with no other justification — see `PROJECT_MEMORY_v8.md` §4's
v7 status entry for the explicit statement that this was a deliberate
limit on how far parameter-tuning was pushed.

### 7.1 Threshold calibration — the shrinkage estimator, again

`threshold = mean(intra-reference scores) - 2.0 * std_used`, where
`std_used` is the SAME shrinkage-blended estimator described in §4
(originally introduced for this exact purpose — the older
`PROJECT_MEMORY` Bug #13 — and only later recognized as needing to
apply to the per-feature stds too, v7's Bug #34). `mean - 2*std`
approximates a ~95%-coverage lower bound under a rough normality
assumption — deliberately conservative (a wider net for "this still
counts as this person") given how few samples typically inform it.

### 7.2 Bounded feedback-driven adjustment (v7, Bug #42)

An examiner confirming or disputing a verdict nudges the live threshold
by a small fixed step (`FEEDBACK_THRESHOLD_STEP = 0.015`) — a simple
online-learning mechanism letting the system self-correct from real
usage. v7 found and fixed a real gap here: this nudge previously had no
bound relative to the profile's own statistically-calibrated value, so
enough repeated feedback events (even entirely reasonable, well-
intentioned ones — see `PROJECT_MEMORY_v8.md` Bug #42's simulation)
could walk the threshold arbitrarily far from what the actual enrolled
evidence supports. Fixed by bounding the cumulative nudge to within
`±0.10` of the last statistical calibration (`_relearn`'s freshly-
computed value) — a real correction within that band is still possible,
it just can't run away unboundedly between recalibrations.

---

## 8. Test-time calibration (TTC): promoting confirmed verdicts to new references

**Decision:** when an examiner confirms a GENUINE verdict and opts in,
that query image can be added to the profile as a new reference
(`engine.py::feedback`, `add_as_reference=True`), up to `MAX_REFERENCES
= 12`.

**Why:** more genuine reference samples directly improve every estimate
this document describes — the natural-variation std (§4), the fusion
score's evidence base (§6), and the threshold's own calibration (§7.1)
all get more reliable with more real samples, and this is the
project's own stated top lever for improving accuracy (echoed
throughout `PROJECT_MEMORY_v8.md`). Letting the system grow its own
evidence base from confirmed real usage, rather than requiring a
separate manual re-enrollment step, is the more usable version of that
same idea.

**Why a near-duplicate guard was needed (v7, Bug #41):** promoting a
sample that's already extremely close to an existing reference
(`>= 0.97` similarity, `_NEAR_DUPLICATE_SCORE`) doesn't add real
evidence about how this signer varies — repeated near-duplicates
compress the observed variation toward zero, which (via the very
shrinkage estimator in §4, whose whole PURPOSE is to trust real
diversity as it accumulates) actively corrupts the statistic it's
supposed to improve. Blocking near-duplicate promotion is the fix that
lets TTC keep its benefit (real new evidence still gets absorbed) while
removing the failure mode (redundant "evidence" that isn't).

---

## 9. What's deliberately NOT in this design (and why)

- **No deep embedding network in the core scoring path** — see §1.
- **No single combined "AI score"** — the nine components (§5) stay
  visible and separately weighted specifically so the system can be
  interrogated, not just trusted.
- **No forced binary verdict** — see §7's INCONCLUSIVE discussion.
- **No cross-signer population statistics inside a single profile's
  scoring** — every threshold/std estimate in this document is
  computed from THAT SIGNER'S OWN enrolled samples only (per-profile),
  never mixed with another signer's data, consistent with the stated
  problem (verify a specific person's own signature, not classify
  "genuine signatures in general").
- **The optional LLM narrative layer (`app/llm/`) never participates in
  the verdict.** It is a strictly downstream, natural-language
  explanation of a decision the classical engine already made — see
  `docs/ARCHITECTURE.md` §2 for the two-brain separation and
  `PROJECT_MEMORY_v8.md` §8 for direct evidence the false-FORGED bug
  occurred and was fixed with the LLM entirely absent, proving it was
  never the cause.
