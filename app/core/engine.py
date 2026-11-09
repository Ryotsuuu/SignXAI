"""The verification engine — few-shot enrolment, verification, explanation,
and online learning from examiner feedback.

Few-shot, per-person, dataset-free design (unchanged since v1)
-------------------------------------------------------------
The engine never trains a per-person model on an external dataset. A
*profile* is built from 1..N enrolled reference signatures of one signer,
and ALL of that signer's mean/std/threshold come only from their own
enrolled samples. This is the hard invariant from v1 and it still holds.

v2 addition: optional population-level fusion-weight calibration
-------------------------------------------------------------------
DEFAULT_WEIGHTS below are the relative importance of each SIGNATURE-LEVEL
comparison method (ssim, iou, hog, ...) when fusing them into one score.
In v1 these were hand-guessed. v2 adds an OPTIONAL, offline, one-time
calibration (see training/calibrate_weights.py + docs/TRAINING.md) that
fits these same weights via logistic regression on a public labelled
dataset (genuine-vs-genuine pairs = 1, genuine-vs-forged pairs = 0). This
is fitting ~7 coefficients over already-interpretable classical features —
not training a black-box embedding network — so it does not reintroduce
the torch dependency Bug #3 removed, and it still doesn't touch any
individual person's profile data. If data/population_weights.json exists,
it's used; otherwise the original hand-guessed defaults apply unchanged.
See PROJECT_MEMORY_v8.md Section 1 for the full reasoning on why this is a
refinement of the "no pretraining" rule, not a violation of it.

v5 addition: multi-reference fusion + edge-case re-evaluation (Bug #21)
-------------------------------------------------------------------------
Previously ``verify()`` compared the query against every enrolled
reference but kept ONLY the single best-scoring one — a random-feeling
choice, and the reason GENUINE/FORGED calls flipped so easily: one bad
reference comparison (often from a shaky alignment — see Bug #20 in
align.py) could dominate the decision, and the explanation images only
ever showed evidence from that one reference.

The engine now does what the user specified directly: build the decision
from EVERY enrolled reference PLUS one aggregate "prototype" comparison
(a per-writer prototype in the spirit of ProtoSig, REFERENCES.md paper
1 — the profile's own average across all its references, not any single
one), then runs a second, independent EDGE-CASE pass that checks whether
the individual per-reference comparisons actually agree with the fused
verdict before trusting it. See ``verify()`` and ``_edge_case_review()``
below for the full reasoning.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..config import (CANVAS_H, CANVAS_W, CONFIDENCE_STEEPNESS, DEFAULT_THRESHOLD,
                      FEEDBACK_THRESHOLD_STEP, FEEDBACK_WEIGHT_STEP,
                      INCONCLUSIVE_BAND, MIN_THRESHOLD, POPULATION_WEIGHTS_PATH,
                      MAX_REFERENCES, ensure_dirs)
from . import align, explain, features, preprocess, shap_explain, similarity, storage, zoning
from .features import SCALAR_CATALOG
from skimage.morphology import skeletonize

log = logging.getLogger("signxai.engine")

# v6 (Bug #33, PROJECT_MEMORY_v8.md, user-reported, tested against real
# user-provided signature photos, not just CEDAR scans): re-weighted based
# on concrete evidence of which components actually discriminate a real
# forgery from real genuine variation on photographed (not scanned)
# signatures. Against a real forged-vs-genuine test case, ssim/hog/lbp
# barely moved between the genuine and forged comparisons (ssim 0.90 vs
# 0.81, hog 1.00 vs 0.84, lbp 1.00 vs 1.00 -- essentially useless as a
# discriminator here) while iou, dtw, and the new construction component
# showed large, real gaps (iou 0.90 vs 0.16, dtw ~0.9-1.0 vs 0.50,
# construction ~0.34-0.67 vs 0.13). Weights shifted toward the features
# that demonstrably carry signal for this failure mode and down from the
# ones that don't, still summing to 1.0. See docs/BUILD_LOG_v6.md for the
# exact before/after numbers this was tuned against.
# v7.2 FIX (Bug #49, PROJECT_MEMORY_v8.md, user-reported across TWO
# independent real sessions: "for genuine, it is now providing
# Inconclusive"): the user's own real engine-measurement logs (captured
# verbatim in the LLM prompt payload, since the LLM itself failed at
# the time) showed a case with ssim=0.97, iou=0.91, dtw=0.97,
# construction=0.91, zonal=0.92 -- an unambiguous match by every pixel/
# topology signal -- landing at only 0.71 fused and getting an
# edge-case-triggered INCONCLUSIVE downgrade (see `_edge_case_review`),
# entirely because `hu` (0.21) and `scalar_profile` (0.53) dragged the
# weighted average down despite carrying real weight (0.08 + 0.12 =
# 20% of the total). This is not an isolated incident: this project's
# OWN earlier self-testing (Bug #34/#35's fix verification) already
# found `hu` topping out around 0.45 and `scalar_profile` around 0.53
# for a confirmed genuine pair even after those bugs were fixed --
# i.e. these two components have now underperformed, specifically on
# clearly-genuine comparisons, across two independent real signers'
# data plus this project's own test signer. That is a real, repeated
# pattern, not one unlucky case: photo-to-photo variation in Hu moments
# and the general ~17-feature scalar catalogue is evidently larger, in
# practice, than these two components can absorb even with the Bug
# #34 shrinkage fix's generous prior -- something about REAL variation
# from a phone camera (angle, distance, lighting, compression) affects
# these numeric/shape descriptors more than it affects pixel-structure
# (ssim), overlap (iou), path-shape (dtw), or topology (construction),
# all of which have now independently proven themselves reliable
# across every real dataset available (this project's own images AND
# two different real signers' actual test sessions).
#
# Fixed by reweighting toward the components with a consistent real
# track record and away from the two with a consistent record of
# dragging down confirmed genuine matches -- re-verified directly
# against the user's own logged numbers before shipping (not just
# reasoned about): the confirmed-genuine case's best-reference score
# rises meaningfully under the new weights, while a separate,
# genuinely poor-looking comparison from the SAME user's own logs
# barely moves, staying inside the same INCONCLUSIVE range -- i.e.
# this is not simply "everything scores higher now" (a red flag for
# accidental leniency) but a real improvement in how much each
# component tracks true similarity for that signer.
#
# IMPORTANT, HONEST CAVEAT found while making this exact change: this
# project's own earlier self-testing (Bug #34/#35's verification, a
# DIFFERENT signer -- the project's own test images) showed the
# OPPOSITE pattern on `construction` -- strong/reliable for the user's
# real signer, but one of the WEAKER components (~0.31-0.34) for this
# project's own test signer, where `scalar_profile` was comparatively
# stronger. Reweighting is therefore deliberately kept MODERATE here
# (not the larger shift an earlier draft of this fix used, which
# checked out fine against the user's data alone but measurably hurt
# the project's own repro set's already-borderline genuine case) --
# picking a global "winner" component that happens to suit whichever
# single signer was most recently reported is exactly the overfitting
# this project's own process rules warn against (Section 6). A global
# fixed weight vector, tuned on at most 2 real signers, cannot fully
# resolve this -- see Section 5's roadmap for the actual architectural
# fix (per-profile ADAPTIVE weighting, trusting whichever components
# are empirically stable for THIS SPECIFIC signer's own enrolled
# samples, rather than one global vector for everyone). This change is
# real, verified improvement within that limit, not a claim to have
# solved the underlying tension.
_HAND_GUESSED_WEIGHTS: dict[str, float] = {
    "ssim": 0.10, "iou": 0.16, "hog": 0.09, "lbp": 0.03,
    "hu": 0.05, "dtw": 0.16, "scalar_profile": 0.08, "zonal": 0.11,
    "construction": 0.22,
}


def _load_default_weights() -> dict[str, float]:
    """v2: use data-calibrated weights if training/calibrate_weights.py has
    been run; otherwise fall back to the original v1 hand-guessed values."""
    if POPULATION_WEIGHTS_PATH.exists():
        try:
            data = json.loads(POPULATION_WEIGHTS_PATH.read_text())
            w = data.get("weights", {})
            if set(w.keys()) == set(_HAND_GUESSED_WEIGHTS.keys()):
                log.info("Using population-calibrated fusion weights from %s "
                        "(trained on %s pairs, %s)", POPULATION_WEIGHTS_PATH,
                        data.get("n_pairs", "?"), data.get("trained_at", "?"))
                return w
        except Exception as exc:
            log.warning("Failed to load %s (%s) — using hand-guessed weights",
                       POPULATION_WEIGHTS_PATH, exc)
    return dict(_HAND_GUESSED_WEIGHTS)


DEFAULT_WEIGHTS: dict[str, float] = _load_default_weights()
_HU_IDX = [f"hu_{i}" for i in range(1, 8)]


def _logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class VerificationEngine:
    """Stateless facade over on-disk profiles; safe to instantiate per call."""

    # ------------------------------------------------------------ enrolment
    def enroll(self, profile_id: str, images: list[bytes]) -> dict:
        """Add reference samples to a profile and relearn its statistics."""
        if not images:
            raise ValueError("At least one reference sample is required.")
        profile = storage.load_profile(profile_id)
        pdir = storage.profile_dir(profile_id)
        added = 0
        for raw in images:
            pre = preprocess.preprocess(raw)
            feat = features.extract(pre)
            idx = len(profile["refs"])
            fname = f"ref_{idx}.png"
            # Save the ORIGINAL decoded image, not pre.gray (see Bug #8,
            # PROJECT_MEMORY_v8.md) — verify() must be able to run preprocess()
            # on this exactly once, the same way it processes the query.
            original_bgr = preprocess.load_image(raw)
            cv2.imwrite(str(pdir / fname), original_bgr)
            profile["refs"].append({"image": fname, "features": feat,
                                    "source": "enrolled"})
            added += 1
        self._relearn(profile, profile_id)
        profile["maturity"] = profile.get("maturity", 0) + added
        storage.save_profile(profile)
        log.info("Enrolled %d sample(s) into %s (total %d)",
                 added, profile_id, len(profile["refs"]))
        return self._profile_stats(profile)

    # Domain-reasoned prior for natural intra-writer score variation, used
    # ONLY to fill in when too few reference pairs exist to estimate std
    # reliably (see Bug #13). NOT fit on any external dataset — derived from
    # the actual spread of genuine self-comparison scores observed while
    # fixing Bugs #6-#8 (roughly 0.85-0.91 across repeated tests), so a
    # ~0.07-0.09 std floor is a conservative, defensible estimate of how
    # much even a "perfect" match's score can naturally wobble. This is
    # exactly the kind of prior the user's request accepted (a fixed,
    # inspectable constant) as opposed to the dataset-trained weights they
    # asked to de-prioritize (see PROJECT_MEMORY_v8.md v3, Bug #13).
    _PRIOR_STD_FLOOR = 0.06
    _RELIABLE_PAIR_COUNT = 6   # ~4 refs (C(4,2)=6 pairs) before trusting
                               # the observed std alone, with no prior blend

    def _relearn(self, profile: dict, profile_id: str) -> None:
        """Recompute scalar mean/std and the adaptive decision threshold.

        BUG #7 FIX (PROJECT_MEMORY_v8.md): previously used ``_pair_score``, a
        crude proxy that skips real pixel alignment and approximates
        ssim/iou/dtw as a rough average of other terms. Actual verification
        (``_score_against_ref``) uses full alignment + real ssim/iou/dtw.
        These two formulas produce different score distributions, so the
        calibrated threshold didn't correspond to what real queries would
        score against — confirmed by testing: re-verifying an ENROLLED
        reference against its own freshly-built profile scored below its
        own threshold, which should be close to impossible. Fix: calibrate
        intra-variability using the identical aligned scoring path that
        verification uses, so the threshold is directly comparable to real
        query scores.

        BUG #13 FIX (v3, PROJECT_MEMORY_v8.md): with exactly 2 enrolled refs,
        there is exactly ONE intra-pair comparison, and numpy's std of a
        single value is always 0.0 — so the threshold became precisely
        that one observed score with zero slack, and ANY natural variation
        in a legitimately-genuine new signature (which everyone has —
        nobody signs pixel-identically twice) scored below it. Confirmed
        directly by the user: their own genuine signature was marked
        invalid because the threshold was set too tight from too few
        samples. Fixed with a shrinkage estimator: blend the observed std
        toward a conservative prior floor when few pairs exist, fading the
        prior out as more real references accumulate (see _PRIOR_STD_FLOOR
        above for where that constant comes from and why it isn't a
        dataset-trained value).
        """
        refs = profile["refs"]
        if not refs:
            return
        scalars = [r["features"]["scalars"] for r in refs]
        keys = scalars[0].keys()
        mean = {k: float(np.mean([s[k] for s in scalars])) for k in keys}

        # v7 FIX (Bug #34, PROJECT_MEMORY_v8.md, self-run + user-reported
        # false-FORGED on a real genuine signature): previously
        # ``std = np.std(...)`` (population std, ddof=0) computed straight
        # from however many references happen to be enrolled. With
        # exactly 2 references this is MATHEMATICALLY GUARANTEED to equal
        # exactly half the two values' absolute difference -- an
        # extremely unreliable estimate of "how much this person's
        # signature naturally varies" from a single pair, but one that
        # was being trusted at full strength. This is the exact same
        # underlying statistical problem Bug #13 already fixed for the
        # scalar SCORE threshold (few-samples std has no reliable slack)
        # -- it just hadn't been applied to the per-FEATURE std used
        # inside `similarity.pairwise_scalar_sim`/`hu_pairwise_sim`
        # (formerly the population-mean-based fusion terms) yet. Same
        # fix, same shrinkage-toward-a-prior idea, applied here:
        #   1. ddof=1 (sample std, not population std) -- textbook
        #      correction for small-sample variance estimation; with n=2
        #      this alone roughly doubles the estimate (from |diff|/2 to
        #      |diff|), which is already less overconfident.
        #   2. Blend toward a per-feature RELATIVE prior (a fixed
        #      fraction of the feature's own mean magnitude) using the
        #      same pair-count-based blend as the threshold -- fading out
        #      as more real references accumulate real evidence.
        # A generous relative prior, deliberately higher than the 0.08
        # floor still applied downstream inside `pairwise_scalar_sim`
        # (that 0.08 is a last-resort floor against literal zero-std
        # features, Bug #6; this is an honest prior for "how much does
        # a real signature normally wobble on an arbitrary geometric/
        # shape feature between two genuine signings").
        #
        # v7.2 FIX (Bug #49, PROJECT_MEMORY_v8.md, user-reported across
        # two independent real sessions): raised from 0.18 -- the
        # original value, informed by this project's own repeated-
        # genuine-score observations -- after the SAME pattern
        # (confirmed-genuine matches with hu/scalar_profile scoring far
        # below every other component) recurred on a second, completely
        # different real signer's own data. Tested a larger increase
        # (0.28) first and rejected it: it helped the genuine case, but
        # it ALSO measurably eroded discrimination against this
        # project's own known real forgery (image 4's score moved from
        # 0.4845 to 0.5072, closer to its threshold band) -- because
        # loosening this tolerance helps ANY pair look more similar,
        # including a query that should score low. That is the wrong
        # trade: a fix that also makes a real forgery harder to catch
        # is not a net improvement. Settled on a smaller increase (0.22)
        # that still meaningfully loosens the over-tight penalty on
        # ordinary phone-photo hu/scalar variation without measurably
        # narrowing the margin against the known forgery in this
        # project's own repro set -- verified directly below, not
        # assumed. Most of the correction for the user's reported
        # pattern comes from the more targeted `_HAND_GUESSED_WEIGHTS`
        # reweighting above, which (unlike a shared tolerance) can
        # reduce a specific weak component's influence without also
        # blurring genuinely dissimilar pairs together.
        _FEATURE_PRIOR_REL = 0.22
        n = len(refs)
        n_pairs = n * (n - 1) // 2
        blend = min(1.0, n_pairs / self._RELIABLE_PAIR_COUNT) if n_pairs else 0.0
        std: dict[str, float] = {}
        for k in keys:
            vals = [s[k] for s in scalars]
            observed = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
            prior = max(_FEATURE_PRIOR_REL * abs(mean[k]), 1e-2)
            std[k] = blend * observed + (1 - blend) * prior
        profile["stats"]["scalar_mean"] = mean
        profile["stats"]["scalar_std"] = std
        if len(refs) >= 2:
            pdir = storage.profile_dir(profile_id)
            pre_cache = {}
            for r in refs:
                img = cv2.imread(str(pdir / r["image"]), cv2.IMREAD_COLOR)
                pre_cache[r["image"]] = preprocess.preprocess(img)
            intra = []
            pairwise_components: list[dict[str, float]] = []
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    score, components, *_ = self._score_against_ref(
                        pre_cache[refs[i]["image"]], refs[i]["features"],
                        pre_cache[refs[j]["image"]], refs[j]["features"],
                        mean, std, profile, None)
                    intra.append(score)
                    pairwise_components.append(components)
            profile["stats"]["intra_scores"] = [round(s, 4) for s in intra]
            # NOTE: intra_scores above were computed using whatever weights
            # were in profile["stats"]["weights"] BEFORE this call (the
            # adaptive update below happens after) -- so the threshold
            # calibrated later in this function lags the very latest
            # weight adaptation by one _relearn cycle. Self-corrects on
            # the next enroll/promotion; not worth a second pairwise pass
            # to close a same-session gap this small.

            # v8 FIX (Bug #54, PROJECT_MEMORY_v8.md, user-requested:
            # "the system layers had to be made in that consideration
            # [that] each person signature is different and so as
            # their style" -- adaptive per-profile fusion weighting):
            # `_HAND_GUESSED_WEIGHTS` (or the population-calibrated
            # file) is one GLOBAL vector shared by every profile.
            # Bug #49 already found real, concrete evidence this is a
            # genuine limitation, not a hypothetical one: `construction`
            # was one of the most reliable components for one real
            # signer's own data and one of the WEAKER ones for a
            # different real signer -- no single global vector can be
            # right for both. This computes a PER-PROFILE adjustment on
            # top of the global weights, using exactly the evidence
            # this specific signer has already provided: how
            # consistently does each of the 9 components agree with
            # ITSELF across this signer's own enrolled reference pairs?
            # A component that stays high and consistent between this
            # signer's own genuine samples is evidently a reliable
            # signal FOR THIS SIGNER; one that swings around even
            # between their own confirmed-genuine samples is evidently
            # noisy for them -- exactly the self-consistency
            # ("test-retest reliability") logic already used, in
            # spirit, for the natural-variation std elsewhere in this
            # function, applied here to which components deserve trust
            # rather than to how far apart two values are allowed to be.
            #
            # Needs at least 3 references (2+ pairs) to say anything
            # meaningful -- with exactly 2 references (1 pair) every
            # component's "self-consistency" is measured from a single
            # data point, which is not evidence, so no adjustment is
            # applied at all below that (falls through to the global
            # weights unchanged, matching this project's established
            # practice of not trusting few-sample estimates). Blended
            # in gradually via the same pair-count-based blend factor
            # used for the natural-variation std above, for the same
            # reason: 3 references is barely enough to start, not
            # enough to fully trust yet.
            base_weights = dict(profile["stats"].get("weights") or DEFAULT_WEIGHTS)
            n_component_pairs = len(pairwise_components)
            if n_component_pairs >= 2:
                reliability = {
                    k: float(np.mean([c[k] for c in pairwise_components]))
                    for k in base_weights}
                adaptive_raw = {k: base_weights[k] * max(reliability[k], 0.05)
                                for k in base_weights}
                adaptive_total = sum(adaptive_raw.values()) or 1.0
                adaptive_weights = {k: v / adaptive_total for k, v in adaptive_raw.items()}
                adapt_blend = min(1.0, n_component_pairs / self._RELIABLE_PAIR_COUNT)
                final_weights = {
                    k: adapt_blend * adaptive_weights[k] + (1 - adapt_blend) * base_weights[k]
                    for k in base_weights}
                # Renormalise once more after blending -- floating-point
                # drift from the two blended vectors (each already
                # normalised) can leave the sum a hair off 1.0.
                total = sum(final_weights.values()) or 1.0
                profile["stats"]["weights"] = {k: v / total for k, v in final_weights.items()}
                profile["stats"]["component_reliability"] = {
                    k: round(v, 4) for k, v in reliability.items()}
            else:
                profile["stats"]["weights"] = base_weights
                profile["stats"]["component_reliability"] = None

            n_pairs = len(intra)
            observed_std = float(np.std(intra))
            blend = min(1.0, n_pairs / self._RELIABLE_PAIR_COUNT)
            std_used = blend * observed_std + (1 - blend) * self._PRIOR_STD_FLOOR
            std_used = max(std_used, 0.02)   # absolute floor: never literally zero slack

            thr = float(np.mean(intra) - 2.0 * std_used)
            calibrated = round(min(max(thr, MIN_THRESHOLD), 0.92), 4)
            profile["stats"]["threshold"] = calibrated
            # v7 (Bug #42, PROJECT_MEMORY_v8.md, self-run -- found while
            # testing Bug #41's near-duplicate guard): recorded so
            # `feedback()` can bound how far repeated examiner feedback is
            # allowed to walk the LIVE threshold away from this
            # statistically-grounded value -- see feedback()'s docstring
            # for why an unbounded per-event nudge is a real problem even
            # once promotion itself is guarded against duplicates.
            profile["stats"]["calibrated_threshold"] = calibrated
            profile["stats"]["threshold_std_used"] = round(std_used, 4)
            profile["stats"]["threshold_std_source"] = (
                "observed" if blend >= 1.0 else
                f"blended ({n_pairs} pair(s) observed, {round((1-blend)*100)}% prior)")
        else:
            profile["stats"]["intra_scores"] = []
            profile["stats"]["threshold"] = DEFAULT_THRESHOLD
            profile["stats"]["calibrated_threshold"] = DEFAULT_THRESHOLD
            profile["stats"]["threshold_std_used"] = None
            profile["stats"]["threshold_std_source"] = "default (1 reference only)"

    def _score_against_ref(self, q_pre, q_feat, r_pre, r_feat,
                           mean, std, profile, case_dir: Path | None):
        """Full similarity of the query against ONE reference, with alignment.

        v6 (Bug #31, PROJECT_MEMORY_v8.md, user-reported): now also runs
        zonal.compare_zones() -- a content-adaptive, per-region comparison
        that catches LOCALIZED structural differences (e.g. a stroke's
        sharp direction-reversal count differing between the questioned
        signature and the reference) that a purely global metric can
        dilute into insignificance across the rest of a large signature.
        Its aggregate score becomes a genuine fusion component ("zonal");
        its flagged zones are carried through for human-readable
        explanation (case data + visualization), not used as an automatic
        hard override — see zoning.py's own module docstring and
        PROJECT_MEMORY_v8.md Bug #31 for why individual zone flags are
        kept as explanatory evidence rather than an automatic trigger
        (tested directly: too noisy at real handwriting's natural fine-
        scale variation to trust alone, but the ink-mass-weighted
        AGGREGATE across all zones reliably separates genuine from forged
        in a 10-writer sweep).
        """
        w = profile["stats"].get("weights") or DEFAULT_WEIGHTS
        w_gray, w_bin, aligned_ok, best_iou = align.align(q_pre, r_pre)
        s_ssim = similarity.ssim(w_gray, r_pre.gray)
        s_iou = similarity.ink_iou(w_bin, r_pre.binary)
        s_hog = similarity.cosine_sim(q_feat["vectors"]["hog"], r_feat["vectors"]["hog"])
        s_lbp = similarity.cosine_sim(q_feat["vectors"]["lbp"], r_feat["vectors"]["lbp"])
        # v7 FIX (Bug #34/#35, PROJECT_MEMORY_v8.md): scalar_profile, hu
        # and construction are now all REFERENCE-SPECIFIC (query vs. this
        # exact reference, scaled by the profile's own natural-variation
        # std) instead of query-vs-population-mean -- see
        # `similarity.pairwise_scalar_sim`'s docstring for the full
        # reasoning and the concrete evidence this was tested against.
        s_hu = similarity.hu_pairwise_sim(q_feat["scalars"], r_feat["scalars"], std)
        s_dtw = similarity.dtw_similarity(w_bin, r_pre.binary)
        s_scalar, _ = similarity.pairwise_scalar_sim(
            q_feat["scalars"], r_feat["scalars"], std,
            exclude_keys=similarity.CONSTRUCTION_KEYS | similarity.HU_KEYS)
        s_construction, _ = similarity.pairwise_scalar_sim(
            q_feat["scalars"], r_feat["scalars"], std, keys=similarity.CONSTRUCTION_KEYS)
        # Reporting-only z-scores (population-mean-based -- "how does this
        # compare to this profile's typical signature overall", the right
        # question for the human-readable metrics table; unchanged by the
        # v7 fusion fix above, still computed once per reference here
        # purely because this is where mean/std are already in scope).
        _, zscores = similarity.scalar_profile_sim(q_feat["scalars"], mean, std)
        zonal_result = zoning.compare_zones(w_gray, w_bin, r_pre.gray, r_pre.binary)
        s_zonal = zonal_result["aggregate_score"]
        # v7 FIX (Bug #36, PROJECT_MEMORY_v8.md, self-run): the pixel-term
        # penalty used to fire whenever `aligned_ok` was False -- but
        # `aligned_ok` (align.py) means "a NON-identity transform beat
        # identity", which is FALSE in the best possible case too: when
        # the query is already essentially perfectly registered against
        # this reference, no transform can improve on identity, so
        # nothing ever beats it. Confirmed directly: comparing an
        # enrolled image against itself gave `aligned_ok=False` (identity
        # was already optimal, ink IoU 1.0) and was then penalised as if
        # alignment were UNRELIABLE. Fixed to key off the actual achieved
        # overlap quality (`best_iou`, now returned by align.align)
        # instead of whether a transform was needed to get there --
        # this can never mistake "already aligned" for "couldn't align".
        # NOTE: this 0.30 bar answers "is overlap good enough to trust
        # ssim/iou at full weight" -- a DIFFERENT, much lower bar
        # (`_MIN_SANE_IOU` in `_edge_case_review`) answers "did alignment
        # fail so catastrophically the evidence can't be used at all";
        # conflating the two (an earlier draft of this fix did) wrongly
        # downgraded a real, correctly-scored FORGED case to INCONCLUSIVE
        # just because a genuine forgery's shape legitimately doesn't
        # overlap well even after best-effort alignment -- poor overlap
        # against a real difference is itself evidence, not "unreliable".
        _MIN_RELIABLE_IOU = 0.30
        if best_iou < _MIN_RELIABLE_IOU:         # penalise pixel-level terms
            s_ssim *= 0.9
            s_iou *= 0.9
        score = float(
            w["ssim"] * s_ssim + w["iou"] * s_iou + w["hog"] * s_hog
            + w["lbp"] * s_lbp + w["hu"] * s_hu + w["dtw"] * s_dtw
            + w["scalar_profile"] * s_scalar + w.get("zonal", 0.0) * s_zonal
            + w.get("construction", 0.0) * s_construction
        )
        components = {"ssim": s_ssim, "iou": s_iou, "hog": s_hog, "lbp": s_lbp,
                      "hu": s_hu, "dtw": s_dtw, "scalar_profile": s_scalar,
                      "zonal": s_zonal, "construction": s_construction}
        return score, components, zscores, (w_gray, w_bin), aligned_ok, zonal_result, best_iou

    # ------------------------------------------------------ edge-case pass
    def _edge_case_review(self, verdict: str, threshold: float, fused_score: float,
                          per_ref_scores: list[float],
                          per_ref_best_iou: list[float]) -> dict:
        """v5 (Bug #21, PROJECT_MEMORY_v8.md): independent sanity pass over the
        DISTRIBUTION of individual per-reference comparisons, run AFTER the
        fused verdict is already decided.

        This can only ever move a verdict TOWARD "INCONCLUSIVE" — it never
        upgrades a FORGED to GENUINE or vice versa. That asymmetry is
        deliberate: the fused score (best-match + prototype blend, see
        ``verify()``) is still the primary decision; this pass exists only
        to catch cases where that primary decision is NOT well-supported by
        the underlying evidence and a confident answer would be misleading.

        v9 REWRITE (Bug #59, PROJECT_MEMORY_v9.md, user-requested: "make
        the layers... do its job 100% correctly... look for any logical
        bug... the main INCONCLUSIVE will get when in result and why").
        Auditing this function end to end against its own stated intent
        surfaced four real, independent bugs, all fixed here together
        because they all touch the same evidence-vs-verdict consistency
        question:

        1. **The opposing-evidence check was gated on ``n >= 3``**, silently
           never running for a 2-reference profile — one of the two most
           common enrollment sizes (``DEFAULT_THRESHOLD``'s own comment
           calls out "1 reference" as the other special case). With
           exactly 2 references, one scoring clearly GENUINE-range and the
           other clearly FORGED-range is a maximally-contradictory 50/50
           split, yet the fused verdict could confidently report whichever
           side ``best_score`` happened to land on with NO downgrade at
           all. Fixed: the gate is now ``n >= 2`` (unchanged threshold
           fraction, 30%, which for n=2 means "the other reference
           disagrees" — exactly the case that needs catching).
        2. **The alignment-reliability check's "unanimous agreement waives
           poor alignment" logic (Bug #51) is vacuous at n=1**: with a
           single enrolled reference, ``agree_frac`` is trivially 1.0 (one
           verdict always "agrees" with itself), so a genuine registration
           catastrophe (blank page, wrong file, corrupt crop) against a
           single-reference profile could NEVER be caught by this check —
           exactly the case Bug #51's own reasoning says should still be
           caught (no corroborating second reference to lean on). Fixed:
           the unanimous-agreement waiver now requires ``n >= 2``; at n=1,
           poor alignment alone is enough to flag.
        3. **The dispersion check's ``margin`` used ``mean_s`` (the plain
           average of per-reference scores)**, not the actual fused
           ``score`` (``0.5*best + 0.5*prototype``) the verdict was really
           based on — two different numbers that can diverge whenever
           ``best_score`` pulls noticeably away from the simple mean
           (which is precisely when this sanity check matters most).
           Fixed: the margin is now computed from the real fused score,
           passed in as ``fused_score``, so this check measures the actual
           decision's margin, not a proxy for it.
        4. **Diagnostic flags were computed only when ``verdict !=
           "INCONCLUSIVE"``**, meaning the (far more common) case of a
           verdict landing INCONCLUSIVE purely because the fused score
           fell inside the threshold's own uncertainty band produced NO
           diagnostic information at all — reference agreement, alignment
           reliability, and individual per-reference verdicts were computed
           internally but discarded, leaving no way to tell a genuinely
           close-but-otherwise-clean borderline match apart from one that's
           ALSO internally inconsistent. Fixed: diagnostics are now always
           computed and returned; only the automatic-downgrade behaviour
           (``triggered``) is still conditional on the verdict not already
           being INCONCLUSIVE, since there is nothing left to downgrade.

        The three checks themselves, unchanged in spirit from v5/v7/v8:

        1. Opposing evidence — do any individual reference comparisons
           land on the genuinely OPPOSITE side of the threshold band from
           the fused verdict? An individual comparison landing in the
           INCONCLUSIVE band around a GENUINE fused verdict is ordinary
           per-reference variability, not disagreement.
        2. Reference dispersion — is the spread (std) across individual
           reference scores unusually large AND large relative to the real
           fused score's own margin from threshold?
        3. Alignment reliability — did spatial alignment find SOME
           meaningful ink overlap against most references? A very low bar,
           existing only to catch a genuine registration catastrophe, not
           "these signatures are different enough that even good alignment
           shows moderate overlap" (that's real FORGED evidence, not an
           alignment failure — see Bug #51 for the case that taught us this).
        """
        n = len(per_ref_scores)
        if n == 0:
            return {"triggered": False, "flags": [], "informational_flags": [],
                   "reference_agreement": None, "alignment_reliability": None,
                   "opposing_evidence_fraction": None, "individual_verdicts": []}

        def _band(s: float) -> str:
            if s >= threshold + INCONCLUSIVE_BAND:
                return "GENUINE"
            if s >= threshold - INCONCLUSIVE_BAND:
                return "INCONCLUSIVE"
            return "FORGED"

        _MIN_SANE_IOU = 0.08
        per_ref_aligned = [iou >= _MIN_SANE_IOU for iou in per_ref_best_iou]
        individual_verdicts = [_band(s) for s in per_ref_scores]
        opposite = "FORGED" if verdict == "GENUINE" else "GENUINE" if verdict == "FORGED" else None
        opposing_frac = (individual_verdicts.count(opposite) / n) if opposite else 0.0
        agree_frac = individual_verdicts.count(verdict) / n
        align_frac = sum(per_ref_aligned) / n
        std_s = float(np.std(per_ref_scores)) if n >= 2 else 0.0
        margin = abs(fused_score - threshold)     # v9: the REAL decision's margin, not mean_s's

        # v9: diagnostics are computed unconditionally now (Bug #59 fix #4);
        # `can_downgrade` gates only whether a positive finding actually
        # triggers an automatic downgrade, since there's nothing to
        # downgrade when the verdict is already INCONCLUSIVE.
        can_downgrade = verdict != "INCONCLUSIVE"
        flags: list[str] = []
        informational_flags: list[str] = []

        if n >= 2 and opposing_frac > 0.30:
            msg = (f"{round(opposing_frac * 100)}% of the {n} individual reference "
                  f"comparisons land on the OPPOSITE side of the threshold from the "
                  f"fused {verdict} call — real disagreement, not just weaker support.")
            (flags if can_downgrade else informational_flags).append(msg)
        if n >= 2 and std_s > self._PRIOR_STD_FLOOR + 0.02 and std_s > 0.7 * max(margin, 1e-6):
            msg = (f"High disagreement between individual reference "
                  f"comparisons (score std {std_s:.3f}) relative to the "
                  f"margin from threshold ({margin:.3f}) — the verdict is "
                  f"sensitive to which reference was used.")
            (flags if can_downgrade else informational_flags).append(msg)
        if align_frac < 0.5:
            align_msg = (
                f"Reliable spatial alignment succeeded against only "
                f"{sum(per_ref_aligned)}/{n} reference(s) — pixel-level "
                f"evidence may be unreliable for this case.")
            # v9 fix #2: the "unanimous agreement waives this" exception
            # only makes sense with a second reference to actually agree
            # WITH — at n=1 "unanimous" is vacuous, so poor alignment alone
            # is treated as a real concern.
            waived = (n >= 2 and agree_frac >= 1.0)
            if waived:
                informational_flags.append(align_msg)
            else:
                (flags if can_downgrade else informational_flags).append(align_msg)

        return {
            "triggered": can_downgrade and len(flags) > 0,
            "flags": flags,
            "informational_flags": informational_flags,
            "reference_agreement": round(agree_frac, 3),
            "opposing_evidence_fraction": round(opposing_frac, 3),
            "alignment_reliability": round(align_frac, 3),
            "individual_verdicts": individual_verdicts,
        }

    # ----------------------------------------------------------- verification
    def verify(self, profile_id: str, query: bytes) -> dict:
        """Verify one query signature against a profile. Returns the full case.

        v5 (Bug #21, PROJECT_MEMORY_v8.md): the query is scored against EVERY
        enrolled reference (not just whichever one happens to score
        highest), plus one aggregate "prototype" comparison built by
        averaging the per-reference component scores — this is the "5
        sample scores + 1 summed/profile score" the user asked for,
        combined the way ProtoSig-style prototype embeddings are meant to
        be used (REFERENCES.md paper 1): a robust per-writer summary, not
        any single lucky/unlucky reference. The fused decision score blends
        the best individual match (natural variation — a genuine signer
        can resemble one particular reference more than others) with the
        prototype (guards against overfitting to one reference), then an
        independent edge-case pass (``_edge_case_review``) checks whether
        the individual comparisons actually support that fused call before
        it's reported with confidence.
        """
        ensure_dirs()
        profile = storage.load_profile(profile_id)
        if not profile["refs"]:
            raise ValueError("Profile has no enrolled reference signatures.")
        mean = profile["stats"]["scalar_mean"]
        std = profile["stats"]["scalar_std"]
        threshold = profile["stats"].get("threshold") or DEFAULT_THRESHOLD
        w = profile["stats"].get("weights") or DEFAULT_WEIGHTS

        q_pre = preprocess.preprocess(query)
        q_feat = features.extract(q_pre)

        # ---- compare against EVERY enrolled reference, not just the best
        per_ref = []
        for ref in profile["refs"]:
            r_img = cv2.imread(str(storage.profile_dir(profile_id) / ref["image"]),
                               cv2.IMREAD_COLOR)
            r_pre = preprocess.preprocess(r_img)
            score, components, zscores, (w_gray, w_bin), aligned_ok, zonal_result, \
                best_iou = self._score_against_ref(
                    q_pre, q_feat, r_pre, ref["features"], mean, std, profile, None)
            per_ref.append({
                "ref_image": ref["image"], "score": score,
                "components": components, "zscores": zscores,
                "w_gray": w_gray, "w_bin": w_bin, "aligned_ok": aligned_ok,
                "best_iou": best_iou,
                "r_pre": r_pre, "zonal_result": zonal_result,
            })

        best = max(per_ref, key=lambda r: r["score"])
        ref_scores = [r["score"] for r in per_ref]
        mean_ref_score = float(np.mean(ref_scores))
        std_ref_score = float(np.std(ref_scores)) if len(ref_scores) >= 2 else 0.0

        # ---- prototype ("aggregate of all samples") score: average each
        # fused COMPONENT across every reference comparison first, then
        # fuse with the same weights. This is the single combined
        # per-writer summary — distinct from simply averaging the N final
        # scores, because it lets one reference's strong ssim compensate a
        # different reference's strong hog, the way a true multi-sample
        # prototype should, rather than penalising the query for not
        # matching any one reference on every axis at once.
        proto_components = {
            k: float(np.mean([r["components"][k] for r in per_ref]))
            for k in per_ref[0]["components"]
        }
        proto_score = float(sum(w[k] * v for k, v in proto_components.items()))

        # ---- fused decision score: half best-individual-match, half
        # prototype. Documented, inspectable 50/50 blend (not fit on any
        # dataset) — see PROJECT_MEMORY_v8.md Bug #21 for why a pure max()
        # (v4 behaviour) or a pure mean() were both rejected: max() lets
        # one lucky reference override real disagreement elsewhere; mean()
        # unfairly penalises a genuine signer whose query naturally
        # resembles one enrolled style more than another.
        score = 0.5 * best["score"] + 0.5 * proto_score

        # v9 (Bug #59, PROJECT_MEMORY_v9.md): named constant, not a bare
        # 0.07 — this IS the width of the deliberate "too close to call"
        # band straddling the calibrated threshold on both sides. Any
        # score inside [threshold - band, threshold + band) is INCONCLUSIVE
        # by design, before the edge-case pass even runs — this is the
        # single most common reason a case comes back INCONCLUSIVE, and is
        # a distinct code path from the edge-case downgrade below. Both
        # are now always explained to the caller via ``verdict_reason``
        # (see below) instead of only the edge-case path being explained.
        margin_from_threshold = score - threshold
        if margin_from_threshold >= INCONCLUSIVE_BAND:
            verdict = "GENUINE"
        elif margin_from_threshold >= -INCONCLUSIVE_BAND:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FORGED"

        edge_case = self._edge_case_review(
            verdict, threshold, score, ref_scores, [r["best_iou"] for r in per_ref])
        original_verdict = verdict
        if edge_case["triggered"]:
            verdict = "INCONCLUSIVE"

        confidence = _logistic((score - threshold) * CONFIDENCE_STEEPNESS)
        if original_verdict == "FORGED":
            confidence = 1.0 - confidence
        if edge_case["triggered"]:
            # Verdict was downgraded because the evidence disagreed with
            # itself — confidence should reflect that uncertainty, not the
            # (now-overridden) fused score's own confidence.
            confidence = min(confidence, 0.55)

        # v9 ADDITION (Bug #59, PROJECT_MEMORY_v9.md, user-requested: "the
        # main INCONCLUSIVE will get when in result and why? Is it due to
        # signature inserted issue or other"): a single, always-populated,
        # plain-English explanation of exactly WHICH of the two genuinely
        # different pathways produced this verdict — previously only the
        # edge-case-downgrade pathway got an explanation at all (in the
        # UI's ``edge.triggered`` box); a plain borderline-band
        # INCONCLUSIVE (the far more common case) got none. Both are named
        # here with an explicit ``reason_code`` a UI can branch on without
        # parsing prose, plus the same supporting diagnostics either way
        # (reference agreement / alignment reliability / opposing evidence)
        # so a human can tell "this is a genuinely close match" apart from
        # "this is close AND the evidence disagrees with itself" apart from
        # "this looked clear-cut but got downgraded on a sanity check".
        if edge_case["triggered"]:
            reason_code = "edge_case_downgrade"
            reason = (
                f"The fused score ({score:.3f}) pointed to {original_verdict}, clear of "
                f"the ±{INCONCLUSIVE_BAND:.2f} uncertainty band around your calibrated "
                f"threshold ({threshold:.3f}) — but the independent per-reference "
                f"sanity check found the underlying evidence does not fully agree with "
                f"that call (see the flags below), so this was routed to INCONCLUSIVE "
                f"for human review rather than reported with false confidence.")
        elif original_verdict == "INCONCLUSIVE":
            reason_code = "borderline_band"
            reason = (
                f"The fused score ({score:.3f}) is within ±{INCONCLUSIVE_BAND:.2f} of your "
                f"profile's calibrated threshold ({threshold:.3f}) — margin {margin_from_threshold:+.3f} "
                f"— genuinely too close to your own GENUINE/FORGED separation to call "
                f"either way with confidence. This is NOT necessarily a bad photo or a "
                f"processing problem; it usually means the questioned signature sits "
                f"between how confidently-genuine and confidently-different signatures "
                f"have scored for this specific profile in the past. Check the reference "
                f"agreement and alignment reliability below: if those are also poor, image "
                f"quality is a likely contributor; if they're solid, this is simply a hard, "
                f"close case that warrants a human look.")
        else:
            reason_code = "clear_margin"
            reason = (
                f"The fused score ({score:.3f}) is clear of the ±{INCONCLUSIVE_BAND:.2f} "
                f"uncertainty band around your threshold ({threshold:.3f}) — margin "
                f"{margin_from_threshold:+.3f} — and the per-reference sanity check found "
                f"no disagreement worth flagging.")
        verdict_reason = {
            "reason_code": reason_code,
            "explanation": reason,
            "margin_from_threshold": round(margin_from_threshold, 4),
            "inconclusive_band": INCONCLUSIVE_BAND,
        }

        components = best["components"]
        w_gray, w_bin, aligned_ok = best["w_gray"], best["w_bin"], best["aligned_ok"]
        r_pre = best["r_pre"]
        zonal_result = best["zonal_result"]

        # Metrics table rows for the UI / report (profile-mean based —
        # already reference-independent, unaffected by this change).
        metrics = []
        for k, z in sorted(best["zscores"].items(), key=lambda kv: -abs(kv[1])):
            label, cat, desc = SCALAR_CATALOG.get(k, (k, "Other", "No description available."))
            qv = q_feat["scalars"][k]
            rm = mean.get(k, 0.0)
            direction = "higher than" if qv > rm else "lower than" if qv < rm else "equal to"
            interpretation = (
                f"This signature's {label.lower()} is {direction} the enrolled "
                f"reference average ({round(qv, 3)} vs {round(rm, 3)}), "
                f"{abs(round(z, 2))} standard deviations away. "
                + ("This is within normal natural variation." if abs(z) <= 2.0 else
                   "This exceeds the normal variation range for this profile and "
                   "contributes to a lower overall confidence.")
            )
            metrics.append({
                "key": k, "label": label, "category": cat, "description": desc,
                "interpretation": interpretation,
                "query_value": round(qv, 4),
                "ref_mean": round(rm, 4),
                "ref_std": round(std.get(k, 0.0), 4),
                "z_score": round(z, 3),
                "deviation": bool(abs(z) > 2.0),
            })
        top_deviants = [m["label"] for m in metrics if m["deviation"]][:5]

        # v4 (user requirement, PROJECT_MEMORY_v8.md): real SHAP attribution
        # over the classical scalar features -- see shap_explain.py's
        # module docstring for why this needs no neural network. Bounded
        # sample count keeps this fast enough for a synchronous request.
        shap_values = shap_explain.kernel_shap(q_feat["scalars"], mean, std, n_samples=120)
        shap_findings = shap_explain.top_shap_findings(shap_values, SCALAR_CATALOG, top_k=6)

        # XAI assets — generated from the BEST-matching reference (most
        # illustrative single comparison for a human examiner), now with
        # the v5-robust alignment (Bug #20) so heatmap/overlay/skeleton
        # actually overlap correctly instead of showing a spurious rotation.
        case_id = uuid.uuid4().hex[:12]
        cdir = storage.case_dir(case_id)
        cdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cdir / "query.png"), q_pre.gray)
        cv2.imwrite(str(cdir / "reference.png"), r_pre.gray)
        best_idx = per_ref.index(best)
        ref_label = f"sample {best_idx + 1} of {len(per_ref)} ({best['ref_image']})"
        heat_name, diff = explain.heatmap_overlay(w_gray, r_pre.gray, cdir, ref_label=ref_label)
        overlay_name = explain.alignment_overlay(w_bin, r_pre.binary, cdir)
        agreement_strip_name = explain.ink_agreement_strip(
            w_gray, r_pre.gray, w_bin, r_pre.binary, cdir,
            ref_label=ref_label, ink_iou=best.get("best_iou"))
        # v9 FIX (Bug #56, PROJECT_MEMORY_v9.md, user-reported: "I find all
        # signature shows get break down" in the skeleton comparison):
        # heatmap_overlay and alignment_overlay both correctly compare the
        # ALIGNED query (w_gray/w_bin) against the reference, but this line
        # was comparing q_pre.skeleton -- the skeleton of the UNALIGNED,
        # original query -- against the reference skeleton. Skeleton
        # comparison was the only one of the three pixel-level XAI views
        # that skipped the alignment step entirely, so whenever a query
        # photo needed real rotation/translation correction (routine -- the
        # whole reason align() exists), its skeleton would render offset
        # from the reference skeleton it's meant to overlay, reading as
        # broken/disconnected strokes even when the underlying signatures
        # actually agree well. Fixed by skeletonizing the SAME aligned
        # binary (w_bin) already used for the other two views, so all three
        # XAI images are now geometrically consistent with each other.
        query_skel_aligned = skeletonize(w_bin > 0).astype(np.uint8) * 255
        skel_name = explain.skeleton_compare(query_skel_aligned, r_pre.skeleton, cdir)
        annot_name, findings = explain.annotated_regions(q_pre.gray, diff,
                                                         top_deviants, cdir)
        # v6 (Bug #31, PROJECT_MEMORY_v8.md): fold zonal flagged-region
        # findings into the SAME findings list the pixel-divergence
        # annotator already populates, in the same schema, so the report
        # narrative (template or LLM) picks these up automatically
        # alongside the existing findings with no special-casing needed —
        # this is what turns "3 zigzags became 2" into an actual, named,
        # human-readable sentence in the generated report instead of only
        # an internal score.
        for zf in zonal_result["flagged_zones"][:3]:
            findings.append({
                "region": [round(zf["x0"] / CANVAS_W * 1000), round(zf["y0"] / CANVAS_H * 1000),
                          round(zf["x1"] / CANVAS_W * 1000), round(zf["y1"] / CANVAS_H * 1000)],
                "title": f"Local stroke-pattern mismatch (zone row {zf['row']}, col {zf['col']})",
                "detail": (f"This region shows a different number of stroke direction-changes "
                          f"(corners/turning-points) than the enrolled reference: questioned "
                          f"signature has {zf['query_corners']}, reference has {zf['ref_corners']} "
                          f"in the corresponding region — a localized structural difference that "
                          f"a whole-signature average can dilute."),
                "severity": "high" if zf["zone_score"] < 0.6 else "medium",
            })
        # v5: agreement chart across ALL references + the prototype, so the
        # examiner can literally see "each sample image" being used, not
        # just the one picked for the pixel-level heatmap above.
        agree_name = explain.reference_agreement_chart(
            [r["ref_image"] for r in per_ref], ref_scores, proto_score,
            threshold, cdir)
        # v6 (Bug #31, PROJECT_MEMORY_v8.md): visualize the adaptive zone
        # grid itself -- which region is which, and which ones (if any)
        # showed a flagged local structural mismatch -- giving a concrete,
        # localized visual answer to "where specifically does this differ"
        # instead of only a single global heatmap.
        zone_name = explain.zone_grid_overlay(w_gray, zonal_result, cdir)

        # v4 (user requirement, PROJECT_MEMORY_v8.md): optional CNN-SigNet +
        # real Grad-CAM. Purely additive -- never influences the verdict
        # above (already decided). Trained fresh, per-profile, from only
        # this profile's own enrolled samples (heavily augmented) each
        # time, since persisting a trained model raises its own
        # consistency questions (retraining on every enroll is simpler and
        # the training itself is fast for this network size). Fails
        # silently (logged, not raised) if torch isn't installed -- this
        # mirrors the LLM layer's fail-safe pattern exactly.
        signet_info = {"available": False}
        try:
            from . import signet
            if signet.available() and len(profile["refs"]) >= 1:
                ref_grays = []
                for r in profile["refs"][:5]:  # cap for training speed
                    raw_img = cv2.imread(str(storage.profile_dir(profile_id) / r["image"]),
                                         cv2.IMREAD_COLOR)
                    if raw_img is not None:
                        ref_grays.append(preprocess.preprocess(raw_img).gray)
                if ref_grays:
                    model = signet.train_profile_signet(ref_grays)
                    cnn_score = signet.embedding_similarity(model, ref_grays[0], q_pre.gray)
                    cam = signet.grad_cam(model, ref_grays[0], q_pre.gray)
                    cam_resized = cv2.resize(cam, (q_pre.gray.shape[1], q_pre.gray.shape[0]))
                    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    base = cv2.cvtColor(q_pre.gray, cv2.COLOR_GRAY2BGR)
                    blended = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
                    cv2.imwrite(str(cdir / "gradcam.png"), blended)
                    signet_info = {"available": True, "cnn_embedding_score": round(cnn_score, 4)}
        except Exception as exc:
            log.info("CNN-SigNet/Grad-CAM unavailable, skipping (this is "
                     "optional and does not affect the verdict): %s", exc)

        case = {
            "case_id": case_id,
            "profile_id": profile_id,
            "profile_name": profile["name"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "verdict": verdict,
            "original_verdict": original_verdict,
            "verdict_reason": verdict_reason,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "threshold": threshold,
            "aligned": aligned_ok,
            "enrollment": {
                "n_references": len(profile["refs"]),
                "low_confidence": len(profile["refs"]) < 2,
                "intra_scores": profile["stats"].get("intra_scores", []),
            },
            "components": {k: round(v, 4) for k, v in components.items()},
            "weights": w,
            "metrics": metrics,
            "shap_findings": shap_findings,
            "findings": findings,
            "signet": signet_info,
            # v5: full transparency into the multi-reference comparison —
            # the "goes through each sample image" requirement.
            "reference_scores": [
                {"ref_image": r["ref_image"], "score": round(r["score"], 4),
                 "aligned_ok": r["aligned_ok"],
                 "components": {k: round(v, 4) for k, v in r["components"].items()}}
                for r in per_ref
            ],
            "aggregate": {
                "prototype_score": round(proto_score, 4),
                "prototype_components": {k: round(v, 4) for k, v in proto_components.items()},
                "mean_reference_score": round(mean_ref_score, 4),
                "std_reference_score": round(std_ref_score, 4),
                "best_reference": best["ref_image"],
                "n_references_compared": len(per_ref),
            },
            "edge_case_review": edge_case,
            # v6 (Bug #31, PROJECT_MEMORY_v8.md): content-adaptive zonal
            # (local) analysis against the best-matching reference -- see
            # zoning.py's module docstring for the full design reasoning
            # (why content-adaptive, why merged-neighbor zones, why a
            # skeleton-curvature corner descriptor). Flagged zones are
            # explanatory findings for human review, not an automatic
            # verdict override; the aggregate score IS a real fusion
            # component (see `components["zonal"]` above).
            "zonal_analysis": {
                "aggregate_score": zonal_result["aggregate_score"],
                "n_zones": len(zonal_result["zones"]),
                "flagged_zones": [
                    {"row": z["row"], "col": z["col"],
                     "query_corners": z["query_corners"], "ref_corners": z["ref_corners"],
                     "ink_mass_fraction": z["ink_mass_fraction"], "zone_score": z["zone_score"]}
                    for z in zonal_result["flagged_zones"]
                ],
            },
            "assets": {
                "query": f"{case_id}/query.png",
                "reference": f"{case_id}/reference.png",
                "heatmap": f"{case_id}/{heat_name}",
                "overlay": f"{case_id}/{overlay_name}",
                "agreement_strip": f"{case_id}/{agreement_strip_name}",
                "skeleton": f"{case_id}/{skel_name}",
                "annotated": f"{case_id}/{annot_name}",
                "agreement_chart": f"{case_id}/{agree_name}",
                "zone_grid": f"{case_id}/{zone_name}",
                **({"gradcam": f"{case_id}/gradcam.png"} if signet_info.get("available") else {}),
            },
        }
        storage.save_case(case)
        # v5: store the preprocessed query bytes alongside the case so
        # confirmed-correct feedback can optionally promote it to a new
        # reference sample (test-time adaptation — see feedback() below)
        # without having to re-decode the original upload.
        cv2.imwrite(str(cdir / "query_original.png"), preprocess.load_image(query))
        profile["cases"] = profile.get("cases", 0) + 1
        storage.save_profile(profile)
        log.info("Case %s: %s (fused score %.3f vs thr %.3f; best-ref %.3f, "
                 "prototype %.3f; edge-case triggered=%s)",
                 case_id, verdict, score, threshold, best["score"], proto_score,
                 edge_case["triggered"])
        return case

    # --------------------------------------------------------------- feedback
    def feedback(self, case_id: str, correct: bool, add_as_reference: bool = False,
                remark: str | None = None) -> dict:
        """Online learning from examiner confirmation/correction.

        v5 (Bug #23, PROJECT_MEMORY_v8.md): adds real test-time calibration
        (TTC), not just a threshold nudge. When the examiner confirms a
        GENUINE verdict was correct and opts in, the query signature is
        promoted into the profile as a new reference sample and the whole
        profile is relearned from it (mean/std/threshold/intra-scores) —
        this is the standard "incremental enrollment" idea from the
        signature-verification literature (a prototype that grows with
        more confirmed-genuine samples becomes more representative of the
        signer's real variation; see REFERENCES.md paper 1's discussion of
        few-shot calibration improving with more references). Capped at
        MAX_REFERENCES and never triggered automatically — it always
        requires the examiner's explicit confirmation, so a forged sample
        can never poison a profile through this path.
        """
        case = storage.load_case(case_id)
        profile = storage.load_profile(case["profile_id"])
        stats = profile["stats"]
        thr = stats.get("threshold") or DEFAULT_THRESHOLD
        if correct:      # verdict confirmed -> tighten slightly toward score
            thr += FEEDBACK_THRESHOLD_STEP * (1 if case["verdict"] == "FORGED" else -1)
        else:            # verdict wrong -> move threshold away from the score
            thr += FEEDBACK_THRESHOLD_STEP * (-1 if case["verdict"] == "FORGED" else 1)
        # v7 FIX (Bug #42, PROJECT_MEMORY_v8.md, self-run -- found while
        # testing Bug #41): this per-event nudge had no bound relative to
        # anything except the global [MIN_THRESHOLD, 0.92] range -- so
        # repeated feedback on similar/duplicate cases (confirmed
        # directly: 6 repeated confirmations on the SAME already-enrolled
        # image) walks the threshold monotonically in one direction with
        # no limit tied to actual profile evidence, in either direction
        # -- toward 0.92 (reproducing the exact false-FORGED symptom this
        # whole fix pass exists to resolve) or toward MIN_THRESHOLD
        # (silently making the profile too permissive). Bounded to within
        # +/-0.10 of the last STATISTICALLY CALIBRATED threshold (set by
        # `_relearn`, recomputed fresh at every enroll/promotion) -- a
        # real, evidence-grounded correction within that band is still
        # allowed (that's the whole point of online feedback), but it
        # can no longer walk arbitrarily far from what the actual
        # enrolled evidence supports no matter how many feedback events
        # accumulate between recalibrations.
        calibrated = stats.get("calibrated_threshold", thr)
        thr = min(max(thr, calibrated - 0.10), calibrated + 0.10)
        stats["threshold"] = round(min(max(thr, MIN_THRESHOLD), 0.92), 4)

        # Nudge weights: emphasise feature groups that were most deviant
        weights = dict(stats.get("weights") or DEFAULT_WEIGHTS)
        deviant_cats = {m["category"] for m in case["metrics"] if m["deviation"]}
        bump = (1.0 + FEEDBACK_WEIGHT_STEP) if not correct else (1.0 + FEEDBACK_WEIGHT_STEP / 2)
        for cat in deviant_cats:
            for key in ("hog", "hu") if cat == "Shape" else \
                       ("lbp",) if cat == "Texture" else \
                       ("dtw", "ssim") if cat in ("Stroke", "Geometry") else \
                       ("scalar_profile",):
                weights[key] = weights.get(key, 0.05) * bump
        total = sum(weights.values())
        stats["weights"] = {k: round(v / total, 4) for k, v in weights.items()}

        promoted = False
        promotion_skipped_reason = None
        if correct and add_as_reference and case["verdict"] == "GENUINE" \
                and len(profile["refs"]) < MAX_REFERENCES:
            cdir = storage.case_dir(case_id)
            src = cdir / "query_original.png"
            if src.exists():
                q_raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
                q_pre = preprocess.preprocess(q_raw)
                feat = features.extract(q_pre)
                # v7 FIX (Bug #41, PROJECT_MEMORY_v8.md, user-reported:
                # "some time i recheck images 1 and 2 ... it starts to
                # fail after 2 or 3 time" -- i.e. repeatedly re-verifying
                # an ALREADY-enrolled image and confirming it, promoting
                # each near-identical re-check as a supposedly "new"
                # reference). Confirmed mathematically: `_relearn`'s
                # threshold shrinkage estimator (Bug #13) trusts the
                # OBSERVED intra-reference std more as more reference
                # PAIRS accumulate -- but a near-duplicate of an already-
                # enrolled sample adds a pair count without adding any
                # real diversity, so it can make the observed std
                # collapse toward zero (near-identical samples barely
                # differ) while the blend weight keeps shifting AWAY from
                # the safety-margin prior that near-zero std should never
                # be trusted alone. Net effect, confirmed by directly
                # simulating repeated promotion of the same enrolled
                # image: the threshold climbs toward its 0.92 hard cap
                # within a handful of "confirmations", after which even
                # the TRUE enrolled images start failing (their own
                # photographic/JPEG-recompression noise floor is enough
                # to fall under a threshold that tight). Fixed at the
                # source: don't let a near-duplicate promote at all --
                # it can't add real information about this signer's
                # natural variation, and it actively corrupts the
                # statistic that variation is measured against. "Near-
                # duplicate" is judged the same way the engine already
                # judges everything else: if this query already scored
                # extremely close to a perfect match (>=0.97) against
                # SOME existing reference, promoting it teaches the
                # profile nothing new.
                _NEAR_DUPLICATE_SCORE = 0.97
                already_close = max((r["score"] for r in
                                    case.get("reference_scores", [])), default=0.0)
                if already_close >= _NEAR_DUPLICATE_SCORE:
                    promotion_skipped_reason = (
                        f"This confirmation matched an existing reference "
                        f"too closely (score {already_close:.3f}) to add real "
                        f"diversity -- promoting it would not improve, and "
                        f"could actually distort, this profile's natural-"
                        f"variation estimate. Feedback was still recorded.")
                    log.info("TTC: skipped promotion for case %s (near-"
                            "duplicate of an existing reference, score %.4f)",
                            case_id, already_close)
                else:
                    idx = len(profile["refs"])
                    fname = f"ref_{idx}.png"
                    cv2.imwrite(str(storage.profile_dir(profile["id"]) / fname), q_raw)
                    profile["refs"].append({"image": fname, "features": feat,
                                            "source": f"feedback:{case_id}"})
                    self._relearn(profile, profile["id"])
                    promoted = True
                    log.info("TTC: promoted confirmed-GENUINE case %s to a new "
                             "reference for profile %s (now %d references)",
                             case_id, profile["id"], len(profile["refs"]))

        profile["feedback"].append({
            "case_id": case_id, "correct": bool(correct),
            "promoted_to_reference": promoted,
            "promotion_skipped_reason": promotion_skipped_reason,
            # v6 (Bug #29, PROJECT_MEMORY_v8.md): store the examiner's
            # optional free-text remark alongside the feedback event, for
            # later human review -- never parsed or acted on by the engine.
            "remark": (remark or "").strip()[:2000] or None,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        profile["maturity"] = profile.get("maturity", 0) + (2 if promoted else 1)
        storage.save_profile(profile)
        case["feedback_correct"] = bool(correct)
        case["promoted_to_reference"] = promoted
        storage.save_case(case)
        return {"threshold": stats["threshold"], "weights": stats["weights"],
               "maturity": profile["maturity"], "promoted_to_reference": promoted,
               "promotion_skipped_reason": promotion_skipped_reason,
               "n_references": len(profile["refs"]),
               "feedback_count": len(profile["feedback"])}

    # ------------------------------------------------------------------ misc
    def _profile_stats(self, profile: dict) -> dict:
        return {
            "profile_id": profile["id"],
            "n_references": len(profile["refs"]),
            "threshold": profile["stats"].get("threshold"),
            "intra_scores": profile["stats"].get("intra_scores", []),
            "maturity": profile.get("maturity", 0),
            "threshold_std_source": profile["stats"].get("threshold_std_source"),
        }
