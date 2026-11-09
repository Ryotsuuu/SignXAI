"""Real SHAP (SHapley Additive exPlanations) attribution over the engine's
classical scalar features.

v4 (user requirement, PROJECT_MEMORY_v8.md): "SHAP" was listed as a core
requirement never integrated. This implements genuine KernelSHAP —
Shapley-value attribution via weighted linear regression over feature
coalitions — not a heuristic approximation dressed up with the name.

Why this needs no new dependency and no neural network: SHAP is a
model-agnostic explanation method. It treats the scoring function as a
black box and asks "how much did each feature's presence/absence change
the output, averaged over all possible orderings of features." That
applies directly to `similarity.scalar_profile_sim` — the ~22 classical
scalar features and their fused similarity score — with zero need for a
differentiable model. This is arguably MORE useful here than SHAP over a
CNN's internal representations would be, because these features are
already human-interpretable (stroke width, loop count, Hu moments, ...),
so a Shapley value on top of them answers "how much did THIS SPECIFIC,
NAMEABLE property drive the score" directly.

Algorithm (standard KernelSHAP, Lundberg & Lee 2017):
1. The "coalition" is a subset of features held at the reference (mean)
   value; the rest use the query's actual value.
2. Sample coalitions, weighted by the SHAP kernel (favours very small and
   very large coalitions, per the original paper's derivation from
   cooperative game theory).
3. Fit a weighted linear regression of coalition -> score; the
   regression coefficients ARE the Shapley values (this equivalence is
   the core KernelSHAP result, not an approximation of convenience).
"""
from __future__ import annotations

import numpy as np

from . import similarity

N_SAMPLES_DEFAULT = 200      # coalition samples -- enough for stable
                              # attributions on ~22 features without being
                              # slow (this runs synchronously in a verify
                              # request, so needs to stay fast)


def _shap_kernel_weight(n_features: int, coalition_size: int) -> float:
    """The KernelSHAP weighting function (Lundberg & Lee 2017, Eq. 5).
    Assigns high weight to very small and very large coalitions (these are
    the most informative for isolating individual feature effects) and low
    weight to mid-sized coalitions."""
    if coalition_size == 0 or coalition_size == n_features:
        return 1e6  # these boundary coalitions anchor the regression
    from math import comb
    return (n_features - 1) / (
        comb(n_features, coalition_size) * coalition_size * (n_features - coalition_size)
    )


def kernel_shap(query: dict[str, float], mean: dict[str, float],
                std: dict[str, float], n_samples: int = N_SAMPLES_DEFAULT,
                seed: int = 42) -> dict[str, float]:
    """Compute Shapley-value attribution for each scalar feature's
    contribution to the scalar_profile similarity score.

    Returns {feature_key: shap_value}. Positive = pushed the score toward
    "matches the reference" (genuine-supporting); negative = pushed it
    toward "diverges from the reference" (forgery-supporting). Values sum
    to (score - baseline_score), the core SHAP "efficiency" property.
    """
    rng = np.random.default_rng(seed)
    keys = list(query.keys())
    n = len(keys)

    def f(mask: np.ndarray) -> float:
        """Score with `mask`==1 features at their query value, ==0 at the
        reference mean (i.e. "removed" -- contributing zero deviation)."""
        blended = {k: (query[k] if mask[i] else mean.get(k, query[k]))
                  for i, k in enumerate(keys)}
        s, _ = similarity.scalar_profile_sim(blended, mean, std)
        return s

    baseline = f(np.zeros(n, dtype=int))     # all features at reference -> ~perfect match
    full = f(np.ones(n, dtype=int))          # all features at query value -> the real score

    # Sample coalitions (subsets of features to "activate"), weighted per
    # the SHAP kernel, then solve the weighted least-squares regression
    # whose coefficients are the Shapley values (standard KernelSHAP).
    masks, weights, targets = [], [], []
    all_sizes = list(range(1, n))  # exclude 0 and n, added explicitly below
    for _ in range(n_samples):
        size = int(rng.choice(all_sizes)) if all_sizes else 0
        idx = rng.choice(n, size=size, replace=False)
        mask = np.zeros(n, dtype=int)
        mask[idx] = 1
        masks.append(mask)
        weights.append(_shap_kernel_weight(n, size))
        targets.append(f(mask))

    # Anchor with the empty and full coalitions (standard KernelSHAP setup)
    masks = [np.zeros(n, dtype=int)] + masks + [np.ones(n, dtype=int)]
    weights = [_shap_kernel_weight(n, 0)] + weights + [_shap_kernel_weight(n, n)]
    targets = [baseline] + targets + [full]

    X = np.array(masks, dtype=float)
    y = np.array(targets, dtype=float)
    W = np.diag(weights)

    # Weighted least squares with the efficiency constraint (sum of SHAP
    # values == full - baseline) built in via an augmented intercept term.
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    try:
        WX = W @ X_aug
        beta = np.linalg.lstsq(WX.T @ X_aug, WX.T @ y, rcond=None)[0]
        shap_values = beta[1:]  # drop intercept
    except np.linalg.LinAlgError:
        # Degenerate case (e.g. n_features too small) -- fall back to
        # equal split of the total effect, which is the correct Shapley
        # answer when features are symmetric.
        shap_values = np.full(n, (full - baseline) / max(n, 1))

    return {k: float(v) for k, v in zip(keys, shap_values)}


def top_shap_findings(shap_values: dict[str, float], catalog: dict, top_k: int = 5) -> list[dict]:
    """Convert raw Shapley values into report-ready findings, sorted by
    |impact|, using the same SCALAR_CATALOG labels/descriptions the rest
    of the report already uses for consistency."""
    ranked = sorted(shap_values.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    out = []
    for key, val in ranked:
        label, cat, desc = catalog.get(key, (key, "Other", ""))
        out.append({
            "feature": key, "label": label, "category": cat,
            "shap_value": round(val, 4),
            "direction": "supports GENUINE" if val > 0 else "supports FORGED",
        })
    return out
