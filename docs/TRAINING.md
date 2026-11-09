# Training / Dataset Calibration (v2)

> Read `PROJECT_MEMORY_v8.md` Section 1 first — this document explains *how*
> to use the calibration pipeline; that section explains *why* it exists
> without violating the "no pretrained per-person model" rule.

## What this is (and isn't)

**Is:** an optional, one-time, offline calibration of 7 numbers — the
relative weight given to each of the engine's classical signature-comparison
methods (SSIM, IoU, HOG-cosine, LBP-cosine, Hu-moment-cosine, DTW,
scalar-feature-profile) when they're fused into one similarity score. In
v1 these 7 weights were hand-guessed. This script fits them properly using
logistic regression on a real labelled dataset of genuine/forged signature
pairs.

**Isn't:** a neural network, an embedding model, or anything that changes
how an individual person's profile is calibrated. Every profile still
learns 100% from its own 1-5 enrolled samples, exactly as in v1. This
script never sees or touches profile data — it only produces better
defaults for the fusion weights that apply to everyone, before any
per-person calibration happens.

## How to run it

1. Get a Kaggle-style signature dataset with this exact structure:
   ```
   001/          <- genuine signatures for user 001 (any image count)
   001_forg/     <- forged signatures for user 001
   002/ 002_forg/
   ...
   ```
2. Place the zip file at the project root, named exactly `kaggle-dataset-sign.zip`.
3. Install the training-only dependency (not needed for the app itself):
   ```bash
   .venv/bin/python -m pip install -r requirements-train.txt
   ```
4. Run:
   ```bash
   .venv/bin/python training/calibrate_weights.py
   ```
   For a quick test on a subset first (recommended before committing to a
   full run on a large dataset):
   ```bash
   .venv/bin/python training/calibrate_weights.py --max-users 15
   ```

What it does, step by step:
1. Auto-extracts the zip into `Sign Dataset/` (skips if already extracted;
   `--force-extract` to redo it).
2. Scans for `NNN` / `NNN_forg` folder pairs.
3. For each user, builds genuine-vs-genuine pairs (label 1) and
   genuine-vs-forged pairs (label 0), capped at 8 pairs/user so no single
   user's folder dominates the fit.
4. Runs every pair through the **exact same** preprocess → align → extract
   → compare pipeline `engine.py` uses for real verification (not a
   simplified/approximate version — this matters for the fit to actually
   transfer to real usage).
5. Fits a logistic regression, evaluates it on a held-out 25% split
   (prints AUC + accuracy — always read this number before trusting the
   result), and writes `data/population_weights.json`.
6. `engine.py` picks this file up automatically on next run. Delete it to
   revert to the v1 hand-guessed weights.

## What's shipped in this delivery vs. what you should do

> **v7 note:** the validation-run numbers below were produced by an
> OLDER version of `calibrate_weights.py` that only computed 7 of the
> 9 features the real engine actually fuses (`zonal` and `construction`
> were missing — see `PROJECT_MEMORY_v8.md` Bug #38). Because of that
> mismatch, a weights file from that older script could never actually
> be loaded by `engine.py` in the first place (a silent, permanent
> no-op — also Bug #38), so this historical run was never live in any
> shipped version regardless. The script now computes all 9 components
> using the same reference-specific similarity functions the engine
> itself uses post-v7 (see `docs/ALGORITHM_DESIGN.md` §4-5) — **the
> numbers below are kept for historical record only; re-run the script
> to get a real, loadable 9-weight file** before trusting any
> calibrated weights in production.

The `data/population_weights.json` included in this delivery came from a
**validation run**, not your real dataset — the build sandbox didn't have
your `kaggle-dataset-sign.zip`, so a CEDAR-shaped 15-writer subset (same
folder pattern) was used to prove the pipeline actually works end to end.
That run's real, honestly-reported result:

```
240 pairs (120 genuine, 120 forged) from 15 users
Held-out AUC: 0.813   Accuracy: 0.750
Calibrated weights:
    hog              0.3012
    iou              0.2307
    scalar_profile   0.1870
    dtw              0.1428
    lbp              0.1009
    hu               0.0375
    ssim             0.0000
```

**You should re-run this with your actual Kaggle dataset** — the weights
above are a proof-of-concept from a small subset, not a final calibration.
The script will overwrite `population_weights.json` and the app picks up
the new file automatically.

## Reading the results honestly

- **AUC (Area Under Curve)**: how well the fused score separates genuine
  from forged pairs, independent of any specific threshold. 0.5 = random
  guessing, 1.0 = perfect separation. 0.81 (the validation run above) is
  decent but not exceptional — expect it to improve with more users/images
  in your real dataset, since 15 users / 240 pairs is a small sample.
- **A weight of 0.0000 (like `ssim` above) doesn't mean SSIM is useless**
  — it means, once the other 6 features are accounted for, SSIM didn't add
  independent discriminative power in THIS dataset. It may behave
  differently on your dataset, especially if your images have more/less
  photographic noise than CEDAR's (which is scanned, not photographed).
- If AUC comes back low (<0.65) on your real dataset, don't trust the
  weights — the script already handles the degenerate all-zero case by
  falling back to the hand-guessed defaults, but a low-but-nonzero AUC
  fit could still be *worse* than the hand-guessed defaults. Compare
  before/after on a few real verifications, don't assume newer = better.

## Multi-language / multi-script signatures

The classical features this calibrates ARE script-agnostic (see
`PROJECT_MEMORY_v8.md` Section 7 for the full reasoning) — but the **weights**
this script fits are only as validated as the dataset they're fit on. If
your dataset is Latin-script signatures only, the weights are proven for
Latin-script signatures only; the underlying features should generalize,
but that's an architectural argument, not a measured result. If you have
access to non-Latin-script signature samples, running a second calibration
pass (or at minimum a manual verification test) against them is the
honest way to confirm generalization rather than assume it.
