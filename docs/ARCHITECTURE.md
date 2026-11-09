# Architecture

> Companion to `PROJECT_MEMORY_v8.md` — that file is the source of truth for
> *history* (bugs, decisions, status). This file is the source of truth for
> *how the system is shaped right now*. Update both when you change the shape.

## 1. System diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ENTRY POINTS (pick one)                      │
│   main.py  → FastAPI + web UI (web/)          cli.py → terminal (rich)│
└───────────────────────────┬────────────────────────┬─────────────────┘
                             │                        │
                             ▼                        ▼
                    app/api/routes.py          (calls engine directly,
                    (HTTP <-> engine glue)       same functions)
                             │
                             ▼
              ┌──────────────────────────────┐
              │   app/core/engine.py          │   <- THE DECISION LAYER
              │   VerificationEngine          │      (no pretraining, ever)
              │   .enroll()  .verify()        │      v5: verify() scores vs.
              │   .feedback()                 │      EVERY reference + a
              └───┬────────┬────────┬─────────┘      prototype, then an
                  │        │        │                edge-case re-check
      ┌───────────┘        │        └───────────┐    (Bug #21)
      ▼                    ▼                    ▼
preprocess.py          features.py          align.py
(binarize, crop,       (stroke geometry:    (v5: scored multi-candidate
 deskew, denoise)       curvature, pressure  alignment -- identity /
                        proxy, stroke width,  centroid / PCA-moments /
                        density, ink ratio,   ORB+ECC, rotation-clipped,
                        ~20+ features)        best-by-IoU wins; see
                                              PROJECT_MEMORY_v8.md Bug #20)
      │                    │                    │
      └───────────┬────────┴────────┬───────────┘
                   ▼                 ▼
           similarity.py       explain.py
           (per-writer          (heatmap / overlay /
            z-scored distance    skeleton-diff images —
            -> score)            the VISUAL evidence, not
                                  the LLM's job to invent)
                   │                 │
                   └────────┬────────┘
                             ▼
                     storage.py (JSON on disk:
                     data/profiles/*.json,
                     data/reports/<case_id>/*)
                             │
                             ▼  (case dict: verdict, score, metrics, asset paths)
              ┌──────────────────────────────┐
              │   app/llm/report.py           │   <- THE EXPLANATION LAYER
              │   generate_report(case_id)    │      (narrates only, never
              └───┬────────────────────┬──────┘       re-decides validity)
                  │ if LLM ready        │ always available
                  ▼                     ▼
          app/llm/manager.py      template_examination()
          (llama-cpp-python,      (deterministic, metric->
           GGUF, local, no        sentence rules — this is
           network needed once    what runs with ZERO extra
           model is downloaded)   install, always correct
                  │               about what it says because
                  ▼               it only restates numbers)
          app/llm/prompts.py
          (strict JSON contract,
           forensic-examiner
           system prompt,
           retry-on-malformed-
           output loop)
```

## 2. Why the decision/explanation split is a hard rule, not a style choice

This is the single most important architectural decision in the codebase,
so it's justified here with evidence, not just asserted.

A May 2026 study (arXiv:2605.14845) tested current-generation vision-LLMs
(GPT-5.2, Gemini 2.5 Pro) doing signature verification **zero-shot** —
i.e., asking the LLM to look at two signatures and decide genuine/forged
itself, the way a naive "just use a smart AI" design would. Results:

- On easy cases (a different person's signature entirely — "random
  forgery"), the VLM was excellent, even beating specialized trained models.
- On **skilled forgery** — the actual fraud case, where someone practiced
  copying the signature — VLM zero-shot error rates were 35-48%, roughly
  7-8x worse than a small specialized model trained for the task.
- Worse: adding chain-of-thought reasoning made skilled-forgery detection
  *worse*, not better (35%→48% error). The researchers call this the
  **"Rationalization Trap"** — the model talks itself into dismissing real
  forgery artifacts as "natural variation," while reporting 88-92%
  confidence in the wrong answer.

**Conclusion for this codebase:** never let the LLM decide. `engine.py`
computes the verdict from classical, auditable, per-writer-calibrated
features (`features.py`, `similarity.py`) — the same numbers every time,
reproducible, no temperature/sampling randomness. `report.py` then asks
the LLM to narrate *only* the metrics `engine.py` already computed, via a
strict JSON schema (`prompts.py`) that structurally cannot contain a
verdict field the engine didn't provide. If the LLM's output doesn't
validate, `report.py` retries up to 3 times, then falls back to
`template_examination()` — a deterministic, rule-based sentence generator
that is *always correct* about what it says, because it only restates
numbers, never invents an interpretation.

## 3. Why no pretraining, ever

Per the project brief: the engine must self-calibrate per person from 1-5
enrolled samples, not rely on any external signature dataset or pretrained
embedding model. This is implemented literally — `app/core/*` has zero
imports of torch/transformers/any ML framework. Feature extraction
(`features.py`) is classical computer vision (stroke curvature, pressure
proxy from ink density, stroke width variance, aspect ratio, etc. — see
that file's docstring for the full list). The "learning" that happens is:

1. **Per-profile calibration** (`engine.py::_relearn`): mean/std of each
   feature across a profile's own enrolled samples, producing a
   personalized similarity threshold. Two people can have very different
   natural signature variability, and this is why the threshold is
   per-person, not global.
2. **Online feedback correction** (`engine.py::feedback`): when an examiner
   confirms or corrects a verdict, the profile's threshold nudges toward
   that feedback. This is the "learns through past & new data" mechanism
   from the brief — implemented as a lightweight statistical update, not
   gradient descent on a neural net.

This means accuracy is fundamentally bounded by how many genuine samples a
profile has (see `PROJECT_MEMORY_v8.md` Section 2 for the ProtoSig paper's
measured EER-vs-sample-count numbers, which is the best available
reference point even though it's a different, deep-learning-based system —
the qualitative trend "more samples = lower error" applies to any
few-shot verification approach, including this one).

## 4. Cross-platform design decisions

| Constraint | Decision | Why |
|---|---|---|
| No bash/.bat scripts | `install.py`, `main.py`, `cli.py` — plain Python | Same command on every OS |
| GPU acceleration without requiring a compiler | `install.py` auto-detects NVIDIA via `nvidia-smi` and installs from the official prebuilt-wheel index for the matching backend (`cu121` for NVIDIA, `metal` for Apple Silicon, `cpu` otherwise — see Bug #25) | Avoids requiring Visual Studio Build Tools / Xcode CLI tools for most users |
| Modern Linux blocks bare `pip install` (PEP 668) | `install.py` auto-creates and re-execs into `.venv/` | See Bug #4 in PROJECT_MEMORY_v8.md — found by actually running the installer, not assumed |
| GGUF model format | `llama-cpp-python` instead of `transformers`+`bitsandbytes` | One inference code path for CPU/CUDA/Metal/Vulkan — see Bug #1 |

## 5. Data layout on disk

```
data/
├── profiles/
│   └── <profile_id>.json      # name, enrolled feature vectors, threshold, history
└── reports/
    └── <case_id>/
        ├── case.json           # full verdict + metrics for this verification
        ├── query.png           # the questioned signature, preprocessed
        ├── reference.png       # composite of enrolled references used
        ├── heatmap.png         # divergence heatmap (XAI evidence)
        ├── overlay.png         # ink-agreement overlay (green/red/blue)
        ├── skeleton.png        # stroke skeleton comparison
        ├── report.json         # LLM or template examination, structured
        ├── report.html         # human-readable report
        └── report.pdf          # downloadable forensic report
```

## 6. v2 additions and lessons

### The false-FORGED bug was a methodology mismatch, not one bug
Bugs #6, #7, #8 (see `PROJECT_MEMORY_v8.md`) compounded: a z-score floor too
tight for low-variance integer features, a threshold calibrated with a
different formula than real verification used, and references being
preprocessed twice (corrupting the binary mask). The lesson generalized
into process rule #7: **calibration and verification must use the
identical scoring path**, or the threshold doesn't mean what it claims to.
Any future change to `_score_against_ref` must be mirrored in `_relearn`'s
calibration call, or this class of bug reappears.

### Dataset calibration (`training/calibrate_weights.py`)
See `docs/TRAINING.md` for full usage. In short: the engine's 7 fusion
weights (how much SSIM vs. HOG vs. stroke-profile etc. matter) can now be
fit via logistic regression on a real labelled dataset instead of
hand-guessed, without changing the per-person zero-pretraining design —
see `PROJECT_MEMORY_v8.md` Section 1 for why this is a refinement, not a
violation, of that rule.

### Multi-script signatures
The feature set (`features.py`) never does OCR or script/character
recognition — it treats a signature as pure ink geometry (stroke
curvature, topology, Hu moments, density). This makes it architecturally
script-agnostic in a way a Latin-handwriting-trained embedding network
would not be. The *fusion weights* from dataset calibration are only as
validated as the dataset they're fit on, though — see `docs/TRAINING.md`
"Multi-language / multi-script signatures" for the honest caveat.

## 7. v4 additions

### Single-model simplification
The multi-tier model system (v1-v3) was removed entirely per explicit user
request, after a real bug where hardware-tier logic downloaded a much
larger model than intended. `app/device.py` now has exactly one model —
`google/gemma-4-E2B-it-qat-q4_0-gguf` — with a simple viable/not-viable
check, nothing more.

### Real KernelSHAP (`app/core/shap_explain.py`)
Genuine Shapley-value attribution over the classical scalar features, via
standard KernelSHAP (weighted linear regression over feature coalitions).
Needs no neural network — SHAP is model-agnostic by design, so it applies
directly to the existing `scalar_profile_sim` scoring function. Verified
via the Shapley "efficiency" property (attributions sum to exactly
full-score minus baseline-score) to floating-point precision.

### CNN-SigNet + Grad-CAM (`app/core/signet.py`) — optional, torch-based
A small Siamese CNN, trained fresh per profile from only that profile's
own enrolled samples (heavily augmented — rotation, elastic-style warp,
stroke-thickness jitter, noise), never external data. Provides a genuine
gradient-based Grad-CAM heatmap, distinct from `explain.py`'s classical
pixel-divergence heatmap (which remains the primary, always-available
evidence image). Follows the exact same optional/lazy-import pattern as
the LLM layer: if torch isn't installed, `signet.available()` returns
`False` and the rest of the pipeline is completely unaffected. **This
module's torch-enabled path could not be verified end-to-end during
development** — see `PROJECT_MEMORY_v8.md` v4 status for the concrete
disk-space and CUDA-linkage evidence uncovered while trying. Treat it as
a first implementation to verify, not a tested component.

### LLM install — history (Bug #18, then corrected properly by Bug #25)
Four attempts across v2-v5 to fix LLM install reliability:
- v2-v3: version pins that turned out to not exist on any wheel index.
- v4: switched to a hand-built direct GitHub Releases wheel URL,
  reasoning that `--extra-index-url` resolution was unreliable. This was
  itself wrong in a way not caught until v5: it assumed a universal
  `py3-none-<platform>` wheel filename, but llama-cpp-python ships
  compiled C extensions, so real wheels are tagged per-Python-version
  (`cp310`/`cp311`/`cp312`/...) — the v4 URLs were very likely 404s that
  were never actually exercised against a real qualifying machine.
- v5 (current, Bug #25): uses the project's own officially documented
  install path — `pip install llama-cpp-python --extra-index-url
  https://abetlen.github.io/llama-cpp-python/whl/<backend>` — verified
  fresh against the upstream README, PyPI page, and ReadTheDocs together
  (all agree) before switching to it. pip resolves the correct wheel for
  the current Python version/platform automatically via this index, the
  same mechanism PyPI itself uses — no filename or version-tag guessing
  by this project's own code, so it keeps working as new Python versions
  and llama-cpp-python releases ship. See `install.py::_llm_backend()`
  and `install.py::install_llm_backend()`.

## 8. v5 additions

### Multi-reference recognition + edge-case engine (`app/core/engine.py`)
`verify()` no longer collapses the query-vs-every-reference comparison
down to a single best match. It keeps every individual comparison, builds
a ProtoSig-style aggregate prototype (average each fused component across
references, then fuse), blends `0.5 × best + 0.5 × prototype` into the
decision score, and runs `_edge_case_review()` — an independent pass over
the per-reference distribution that can only ever downgrade a verdict
toward INCONCLUSIVE (never invert GENUINE<->FORGED) when the individual
comparisons don't actually support the fused call. See
`PROJECT_MEMORY_v8.md` Bug #21 for the full reasoning and a real dev-time
retune caught by testing.

### Alignment rewrite (`app/core/align.py`)
Replaced "trust ORB unconditionally" with "generate several candidate
transforms (identity / centroid-translation / PCA-moments-rotation /
ORB+ECC, each rotation-clipped to ±25°), score every one by actual ink
IoU against the reference, keep the winner." Alignment is now
structurally guaranteed to match-or-beat doing nothing. See
`PROJECT_MEMORY_v8.md` Bug #20.

### LLM auto-detection and autostart (`app/device.py`, `app/llm/manager.py`, `main.py`)
`device.model_present_on_disk()` does a real filesystem check (total
bytes under the model's cache directory, ≥2.5GB) instead of relying on an
in-memory status flag that never looked at disk. `manager.autostart()`,
called from a FastAPI startup hook, checks model presence + GPU
viability once at process start and loads the model in a background
thread if both hold — no manual "download" click required for a model
that's already present. See `PROJECT_MEMORY_v8.md` Bug #22.

### Real test-time calibration (`app/core/engine.py::feedback`)
`feedback()` now accepts `add_as_reference`; when a GENUINE verdict is
confirmed correct and the examiner opts in, the query is promoted into
the profile as a new reference sample and the whole profile is relearned.
Capped (`MAX_REFERENCES`), never automatic. See `PROJECT_MEMORY_v8.md`
Bug #23.

## 9. v6 additions

### Preprocessing robustness (`app/core/preprocess.py`)
Two complementary fixes to the crop bounding box: `_remove_distant_outliers()`
discards ink components that are both small (relative to the largest
stroke and to total ink mass) AND spatially isolated from every
substantial component; the crop boundary itself now uses the 0.5-99.5
percentile of ink pixel coordinates rather than literal min/max, so a
long thin flourish physically attached to the real signature can't
distort the whole normalization. See `PROJECT_MEMORY_v8.md` Bug #26.

### Real heatmap + content-adaptive region boxes (`app/core/explain.py`)
`heatmap_overlay()` alpha-modulates its blend by actual divergence
magnitude instead of a flat blend ratio (near-zero-divergence stays
visually neutral); `annotated_regions()` finds real divergence blobs via
contour detection instead of a rigid grid, with non-max suppression and
label collision avoidance. See `PROJECT_MEMORY_v8.md` Bug #27.

### Windows LLM install hardening (`install.py`, `app/llm/manager.py`)
Wheel-only fail-fast installs (Bug #28), plus a post-install import
verification (`_verify_llama_cpp_importable()`) that catches a package
that installed successfully but whose native library can't actually load
— falls back to a more broadly-compatible backend automatically instead
of leaving a broken install in place. See `PROJECT_MEMORY_v8.md` Bugs
#28, #30.

### Web UI restyle + feedback/remark flow (`web/`)
Restyled against the Browserbase design reference; results split into
three labeled sections (engine analysis / high-level visualization /
local AI explanation); real remove control on enrollment thumbnails;
explicit remark + submit feedback flow with the remark persisted
server-side. See `PROJECT_MEMORY_v8.md` Bug #29.

### Zonal (local) signature analysis (`app/core/zoning.py`, new module)
Content-adaptive grid (ink-mass quantile splits, not a fixed pixel grid)
plus merged-neighbor zones plus a skeleton-curvature corner descriptor,
specifically designed to catch a localized structural difference (e.g. a
stroke's direction-reversal count) that global metrics can dilute across
a whole signature. The aggregate zonal score is a real fusion component;
individual zone flags are explanatory findings for human review, not an
automatic verdict override. Read `zoning.py`'s own module docstring in
full before modifying it — it documents three real design iterations
caught by testing, not just the final shipped version. See
`PROJECT_MEMORY_v8.md` Bug #31.

## 10. v6.2 additions

### GPU offload honesty (`app/llm/manager.py`)
`n_gpu_layers=-1` is silently ignored by a non-GPU-capable
`llama-cpp-python` build with no Python-level error. `_capture_native_output()`
+ `_parse_gpu_offload()` capture llama.cpp's own native load log (via
file-descriptor-level redirection, since it bypasses `sys.stdout`
entirely) to get a real, confirmed answer instead of trusting hardware
presence. See `PROJECT_MEMORY_v8.md` Bug #32.

### Construction-consistency fusion component (`similarity.py`, `engine.py`, `features.py`)
Discrete pen-lift/loop/junction COUNT features are split into their own
top-level fusion component (`construction`) instead of being averaged
together with ~17 continuous shape features, which was diluting even
maximally-extreme count mismatches into a moderate-looking overall
score — the same dilution mechanism already fixed once locally for
`zonal` (Bug #31), now fixed globally too. `features.py` also computes
these specific counts from a morphologically-closed ink mask, since raw
counts are vulnerable to lighting/exposure noise on photographed (not
scanned) signatures flipping a discrete count by one. See
`PROJECT_MEMORY_v8.md` Bug #33 for the full reasoning and real evidence
this was tuned against.

## 11. What to read before modifying each part
  `engine.py`, `features.py`, `similarity.py` fully, and re-read Section 2
  above before touching anything the LLM might see — don't let a "smarter"
  change accidentally hand the LLM more decision-making power.
- Changing the LLM layer? Read `prompts.py`'s docstring (the JSON contract
  and retry logic) and `report.py`'s fallback logic — never remove the
  template fallback, it's the reliability floor.
- Changing the installer? Read `PROJECT_MEMORY_v8.md` Bug #4 first — the venv
  bootstrap exists for a real, tested reason, don't simplify it away.
- Changing the classical scoring pipeline (`similarity.py`,
  `engine.py`'s `_score_against_ref`/`_relearn`, `align.py`,
  `zoning.py`)? Read `docs/ALGORITHM_DESIGN.md` first — every
  statistical/algorithmic choice there has a specific reason and, in
  several cases, a specific real bug (`PROJECT_MEMORY_v8.md` Bugs
  #34-#42) it was fixed in response to. Then follow
  `PROJECT_MEMORY_v8.md` §6 rule 10: re-run the full 5-image repro set
  (every enrolled reference, the held-out genuine, and the known
  forgery) after the change, not just the one case motivating it.
