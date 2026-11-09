# SignXAI — Project Memory v9 (continues PROJECT_MEMORY_v8.md)

> **Read `PROJECT_MEMORY_v8.md` first** (Sections 0-8, Bugs #1-#54) — this file does not repeat
> it. This file is the v9 continuation: new bugs #55+, found and fixed in the v8→v9 session,
> triggered by the user's own real production data (`profiles.zip` / `reports.zip` — a live
> profile named "Yash" with 5 enrolled samples and 3 verification cases, two of which the user
> had already marked wrong via feedback). Same rules apply: append, never silently overwrite;
> verify directly against real data before claiming a fix works; every bug entry states root
> cause, fix, and the actual before/after evidence, not just a description of the change.

---

## 1. What triggered this session

The user reported, in their own words, **"the BIGGEST AND MAJOR ISSUE of system"**:
enrolling 5 real samples for a new user ("Yash") caused the calibrated threshold to collapse
from a healthy ~0.65-0.82 range (with 2 samples) down to ~0.30-0.35 (with 5 samples) — and at
that collapsed threshold, a signature the user had explicitly marked **wrong** via feedback
(a false GENUINE) kept re-scoring GENUINE on re-run, as if the feedback had no effect. The user
also reported problems in every High-Level Visualization panel (heatmap, ink agreement,
skeleton, zonal grid) and asked for a full audit of the GENUINE/FORGED/INCONCLUSIVE decision
logic itself, specifically: *"the main INCONCLUSIVE will get when in result and why? Is it due
to signature inserted issue or other?"*

This session did **not** start from assumption. The user's uploaded `profiles.zip` and
`reports.zip` contain the actual enrolled images, actual profile.json, and actual case.json
files behind this exact complaint — every bug below was diagnosed by directly running this
codebase's own preprocessing functions against those real files and looking at the actual
intermediate images, not by guessing what might be wrong.

---

## 2. Bug #55 — preprocessing let large photographic contamination corrupt the ink bounding box, root-causing the threshold collapse

**Severity: critical.** This is the bug behind the user's "biggest and major issue" report.

**Direct evidence, not assumption.** Rendering `profiles/<yash>/ref_0.png` through
`ref_4.png`'s actual binary ink mask (post-existing Bug #50 illumination flattening) showed:

| image | ink bbox before fix | aspect ratio |
|---|---|---|
| ref_0 | (372,200)-(755,339) | 2.73 |
| ref_1 | (0,0)-(1099,638) | 1.72 |
| ref_2 | (0,0)-(1014,1199) | 0.85 |
| ref_3 | (662,489)-(1055,1199) | 0.55 |
| ref_4 | (541,423)-(947,602) | 2.27 |

Five photos of the literal same "Dhore" signature, aspect ratio swinging from 0.55 to 2.73.
Visually inspecting each binary mask found three distinct, real contamination sources, each
large enough to survive the existing size/distance-based outlier filters:

1. **ref_1**: a paper-fold/crease shadow (a long wavy band near the top of the photo) — sampled
  at saturation ≈11 (near-neutral grey) vs. the real ink's saturation of 57-117. Also a small
  patch of ghost bleed-through handwriting and printed "Page No:" text.
2. **ref_2**: a dark patterned fabric/blanket visible along the left frame edge (the photo's
  own background) — sampled at saturation ≈151 (**highly** saturated) but hue ≈10
  (red-orange), vs. the real ink's hue ≈126 (blue-violet). A real colour, just the wrong one —
  not a neutral shadow, so a pure "is it grey" check would have missed it.
3. **ref_3**: a small fragment of a *different, adjacent* enrolled sample bleeding in at the
  very bottom edge of the frame — same ink, same hue (it's genuinely the same pen), 559px away
  from a 132px-tall signature (over 4x its own height).

None of these are small or an isolated fringe — each survived directly into the "substantial,
never discard" bucket of the existing `_remove_distant_outliers` (Bug #21/v5), because that
filter only ever asked "is this big enough to be real content", never "is this the same
*colour* as the real ink" or "how far away, relative to the actual signature's own size, is
this — not relative to a flat height-only margin".

**Root cause, precisely:** intensity-only Otsu binarization (even after Bug #50's
illumination flat-fielding) fundamentally cannot distinguish "dark because it's ink" from
"dark/different because it's a shadow, a background object, or someone else's ink" — these are
a **colour** question a grayscale pipeline structurally cannot answer.

**Fix, three parts, in `app/core/preprocess.py`:**

1. **New `_remove_chromatic_outliers()`** — picks an *anchor* component (the largest one that
  isn't itself an edge-touching, thin/elongated strip — this matters because in ref_1 the
  contaminating fold-line was literally the single largest component in the frame, 24,911px
  vs. the real signature's 9,310px, and would have wrongly become the anchor under a naive
  "pick the biggest blob" rule). Any other component is rejected if it's a different colour
  (>30° hue distance) from the anchor **or** near-neutral while the anchor ink is clearly
  coloured (a shadow is never "a faint patch of the same coloured ink"). Skips entirely on a
  genuinely monochrome capture (anchor itself has near-zero saturation) — chroma carries no
  signal there, and applying it could wrongly strip real black ink.
2. **`_remove_distant_outliers()` refined**: a "substantial" component under 15% of the
  anchor's own area now ALSO has to fall within a margin scaled to the **anchor's own
  width/height** (not just a flat `typical_height × 1.5`), closing the exact gap that let
  ref_3's 559px-away bleed-through fragment through — it barely cleared the old area-only bar
  (321px vs. a ~301px floor) with nothing checking whether 321px of stray ink could plausibly
  be that far from a 132px-tall signature.
3. **Fallback-region leak, caught by this session's own regression test, not by the original
  real-data validation**: a component correctly excluded by check (2) above could still sneak
  back in if it happened to also fall inside the separate, looser region check meant for
  genuinely-never-substantial noise. Fixed by tracking `demoted` components explicitly and
  never re-considering them. This is exactly why `tests/test_regressions.py` (Bug #59, below)
  exists now — this specific leak did not show up against the 7 real photos on hand, only
  against a synthetic geometry the regression test constructed on purpose to probe it, and a
  future change could easily have reintroduced it silently without a codified test catching it.

**Verified, not assumed — before/after on the same 7 real images** (5 Yash + 2 pre-existing
Vinit references, used specifically as a regression guard for a *legitimately* multi-part,
full-width, ornate signature with several separated stroke groups):

| image | aspect before | aspect after | 
|---|---|---|
| yash ref_0 | 2.73 | 2.64 |
| yash ref_1 | 1.72 | 2.43 |
| yash ref_2 | 0.85 | 2.41 |
| yash ref_3 | 0.55 | 2.85 |
| yash ref_4 | 2.27 | 2.19 |
| vinit ref_0 | 1.95 (unchanged) | 1.95 |
| vinit ref_1 | 2.14 (unchanged) | 2.14 |

Vinit's two references are **byte-identical pixel counts before and after** — proof the fix
removes real contamination without touching legitimate multi-part ink.

**End-to-end impact, measured by actually re-enrolling and re-running the exact 3 cases the
user's own feedback had flagged:**

- Calibrated threshold on re-enrollment: **0.30 (the hard floor) → 0.443** — no longer pinned.
- The two cases the user had marked `correct: false` (system said GENUINE, user said wrong):
  score dropped from 0.577-0.619 (confidently, wrongly, GENUINE) to **0.472-0.493
  (INCONCLUSIVE)**. This is reported honestly, not oversold: the fix does not claim to now
  produce a confident, correct FORGED — it demonstrably stops the confident, wrong GENUINE,
  which is the failure mode the user actually reported ("re-run it provide same genuine
  remark" despite negative feedback). Whether the true answer is FORGED or a genuinely
  hard-to-call sample is something only a human examiner (or more enrolled samples) can settle
  from here — that is what INCONCLUSIVE is for, and is a categorically safer failure mode than
  confidently wrong.

---

## 3. Bug #56 — skeleton comparison used the UNALIGNED query skeleton, the only one of three pixel-level XAI views that skipped alignment

**User's words: "It works great, but I find all signature shows get break down."**

`engine.py`'s verify() aligns the query image to the reference (`align.align()`) before
computing the heatmap (`heat = heatmap_overlay(w_gray, r_pre.gray, ...)`) and the ink-agreement
overlay (`alignment_overlay(w_bin, r_pre.binary, ...)`) — both correctly use the *aligned*
query. But `skeleton_compare(q_pre.skeleton, r_pre.skeleton, ...)` used `q_pre.skeleton`, the
skeleton of the **original, unaligned** query — the only one of the three views that never
went through the alignment step the other two rely on. Whenever a real query photo needed
genuine rotation/translation correction (routine — that's the entire reason `align()` exists),
its skeleton would render visibly offset from the reference skeleton it's meant to overlay,
reading as broken/disconnected strokes even when the underlying signatures actually agree well.

**Fix:** skeletonize the same aligned binary (`w_bin`) already used for the other two views:
`query_skel_aligned = skeletonize(w_bin > 0).astype(np.uint8) * 255`, then compare that against
`r_pre.skeleton`, so all three pixel-level views are now geometrically consistent.

**Verified directly** — same genuine same-signer pair (ref_0 vs. ref_4), before/after:
before, the underline strokes render as two clearly-parallel, non-overlapping red/blue lines
for their entire length; after, the same underline overlaps in green for a visibly longer
stretch, and more of the letter strokes overlap as well. (`best_iou` for this pair: 0.27 — a
real, moderate alignment, exactly the kind of case where the bug used to matter and the fix
visibly helps.)

---

## 4. Bug #57 — divergence heatmap never named which enrolled reference it was compared against

**User's words: "It does not show which image is get based match [name is missing and side by
side comparison]."**

`heatmap_overlay()` now accepts an optional `ref_label` and burns it into the image as a
caption strip ("Compared against: sample 1 of 5 (ref_0.png)") via a new shared `_caption_strip()`
helper, so the saved PNG is self-explanatory even when viewed standalone (e.g. in the
downloaded PDF report, where surrounding JSON context isn't visible). `engine.py` now computes
the matched reference's position (`best_idx`) and passes a real label through.

---

## 5. Bug #58 — ink agreement view showed only the overlay, never the raw query/reference it came from

**User's words: "I want you show three things, side by side comparing and down below it
overlapping animation or clear image, and point out what it gets wrong."**

New `side_by_side_panels()` (shared helper, also usable elsewhere) stitches N captioned panels
horizontally. New `ink_agreement_strip()` builds: **Query (as processed) | Reference: <label> |
green/red/blue agreement map**, then computes the single largest contiguous disagreement
region's rough position (e.g. "largest mismatch concentration: lower-right area") and burns
that as a caption across the whole composite. Exposed as a new `agreement_strip` asset
alongside the existing `overlay` (kept unchanged, for backward compatibility with anything
already depending on it).

Note on the "dis-oriented and oversizing Question image" half of this complaint: verified this
was almost entirely downstream of Bug #55's crop-bbox corruption, not a separate rendering bug
— fixing the crop at the source fixes how this view (and the zonal grid, which showed the same
symptom) looks, without needing a second, independent fix here.

---

## 6. Bug #59 — verdict-decision logic audit: four real bugs in the GENUINE/FORGED/INCONCLUSIVE path, plus a transparency gap on "why INCONCLUSIVE"

The user asked directly: **"look for any logical bug or any wrong approach we are taking for 3
final remarks... the main INCONCLUSIVE will get when in result and why?"** This required
reading `_edge_case_review()` and the verdict-band decision in `verify()` end to end against
their own stated intent, not just skimming for obvious crashes. Four independent logic bugs
were found, plus one significant transparency gap:

**Bug #59.1 — opposing-evidence check gated at `n >= 3`, never running for 2-reference
profiles.** `DEFAULT_THRESHOLD`'s own comment names "1 reference" as a special case, and 2 is
just as common a real minimum enrollment size. With exactly 2 references split — one scoring
clearly GENUINE-range, the other clearly FORGED-range, a maximally contradictory 50/50 split —
this check silently never ran, and the fused verdict could confidently report whichever side
`best_score` happened to land on with **zero** downgrade. Fixed: gate lowered to `n >= 2`.
Verified: a synthetic n=2 case with scores `[0.75, 0.30]` around threshold 0.5 now correctly
triggers (`triggered: true`, `opposing_evidence_fraction: 0.5`).

**Bug #59.2 — alignment-reliability's "unanimous agreement waives poor alignment" (the v8 Bug
#51 fix) is vacuous at `n=1`.** With one enrolled reference, "unanimous agreement" is trivially
true (a single verdict always agrees with itself), so a genuine registration catastrophe (wrong
file, blank page, corrupt crop) against a single-reference profile could never be caught — the
exact opposite of Bug #51's own stated reasoning (no corroborating second reference to lean on
at n=1). Fixed: the waiver now requires `n >= 2`. Verified: n=1 with `best_iou=0.02` (far below
the 0.08 sanity floor) now correctly triggers.

**Bug #59.3 — dispersion check's `margin` used `mean_s` (plain average of per-reference
scores), not the actual fused `score` the verdict was really based on.** `score = 0.5*best +
0.5*prototype` can diverge meaningfully from a plain per-reference mean whenever `best_score`
pulls away from the pack — precisely when this sanity check matters most. Fixed: margin is now
computed from the real fused score, passed in explicitly as `fused_score`.

**Bug #59.4 — diagnostic flags were computed only when `verdict != "INCONCLUSIVE"`.** The
(more common) case of a verdict landing INCONCLUSIVE purely because the fused score fell inside
the threshold's own ±0.07 band produced **no diagnostic information at all** — reference
agreement, alignment reliability, and per-reference verdicts were computed internally then
discarded. There was no way to tell "a genuinely close-but-clean borderline match" apart from
"close AND internally inconsistent". Fixed: diagnostics are now always computed and returned;
only the automatic-downgrade behaviour stays conditional on there being something left to
downgrade.

**Regression guard for the original Bug #51 fix**, re-verified after this rewrite so fixing
#59.1/#59.2 didn't reintroduce what #51 fixed: n=2, both references unanimously FORGED despite
poor alignment (a real difference, not a registration failure) — confirmed still `triggered:
false`, alignment concern correctly demoted to informational-only.

**The transparency gap, and its fix — `verdict_reason`:** even with the four logic bugs fixed,
the system still only ever *explained itself* (in the web UI) when the edge-case pass
downgraded a verdict. A plain borderline-band INCONCLUSIVE — the far more common path — got no
explanation at all: the user's own direct question ("why? Is it due to signature inserted issue
or other") had no answer in the product itself. Added: every `verify()` call now returns a
`verdict_reason` object with an explicit `reason_code`:

- `clear_margin` — score is clear of the uncertainty band, sanity check found no disagreement.
- `borderline_band` — score genuinely falls within ±0.07 of the calibrated threshold. Stated
  plainly in the explanation text: *this is not necessarily a bad photo or a processing
  problem* — it means this signature sits between how confidently-genuine and
  confidently-different signatures have scored for this specific profile historically. The
  explanation points the user at the reference-agreement/alignment-reliability numbers to judge
  for themselves whether poor image quality is a likely contributor.
- `edge_case_downgrade` — score looked clear-cut but the per-reference evidence disagreed with
  itself; the specific flags are attached.

The web UI (`web/app.js` + `web/style.css`) renders this for **every** case now (previously,
only `edge.triggered` cases got any explanatory box at all), plus `docs/USAGE.md` gained a new
"What GENUINE / FORGED / INCONCLUSIVE actually mean" section defining the fused score formula,
the exact GENUINE/FORGED band conditions, and both INCONCLUSIVE pathways in one place — the
single canonical answer to the user's question, kept next to the feature itself rather than
only living in this memory file.

**One framing note, worth stating plainly rather than silently living with:** "FORGED" as a
label implies intent this system cannot actually detect — it can only say "statistically
inconsistent with the enrolled reference set." The existing report narrative (`app/llm/
report.py`) already frames this carefully ("not produced by the reference writer", explicit
"decision support, not a standalone legal determination" in every recommendation) — this was
checked and left as-is rather than renamed, since the enum value itself is load-bearing across
stored profile/case JSON and changing it is a breaking storage-format change that should be a
deliberate, separately-scoped decision, not a side effect of this audit.

---

## 7. New: `tests/test_regressions.py`

Per the user's explicit ask to "test things & validate the system is working properly", a
real, dependency-free (no pytest — matches this project's existing requirements.txt, which
depends on nothing test-related) regression suite now exists, run via:

```
python tests/test_regressions.py
```

7 tests, covering every bug above with either the diagnosing real data's *shape* (recreated as
a small synthetic case where the point is pure geometry/arithmetic, so the test doesn't depend
on the user's private photos being present in every environment) or direct unit calls into
`_edge_case_review()`. All 7 pass as of this writing. One of these tests (the anchor-footprint
distance check) caught **the Bug #55 fallback-region leak** during its own construction — a
real bug the original real-photo validation had not exposed, fixed as part of this same session
specifically because the test surfaced it. This is the argument for keeping this suite and
growing it, not a one-time exercise: it already paid for itself once.

---

## 8. Still open / not yet done in this session (see conversation for current status)

This session focused entirely on the preprocessing root-cause (#55) and the verdict-logic audit
(#59) the user asked about directly, plus the two other XAI bugs (#56, #57, #58) found along
the way. Not yet addressed from the user's full original do-list (tracked here so the next
session doesn't have to rediscover scope):

- The 3-way Analyze mode (parallel-AI / no-AI / Hybrid-default) is not yet implemented.
- Enroll/Verify preprocessing-preview UI (show each sample's processed form immediately, with a
  per-image re-preprocess action) is not yet built — `preprocess_traced()` (a new function
  capturing every intermediate stage) exists in `app/core/preprocess.py` ready for a UI to call,
  but no route/UI consumes it yet.
- Feedback-layer training transparency (what got trained, on what data) and a dedicated
  inconclusive/forgotten-case database section are not yet built.
- The zonal-grid visualization was not independently re-verified against real data this
  session (Bug #55's crop fix should substantially help it, by the same logic as Bug #58's
  ink-agreement case, but this was not directly confirmed with a rendered example the way the
  other views were).
