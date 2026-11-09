# SignXAI — Project Memory (Single Source of Truth)

> **Read this file fully before making any change.** Update it after every phase.
> Append, never silently overwrite. Move stale/wrong info to "Superseded", don't delete it —
> wrong turns stay visible so nobody repeats them.
>
> **If you are an LLM picking up this project**: read sections 1-6 in order before touching
> code. Section 5 (roadmap) tells you what phase we're in and what NOT to redo. Section 3
> (bugs) tells you what's already broken and fixed — don't reintroduce those bugs. Section 7
> (open questions) tells you what's genuinely undecided — don't silently assume an answer.
> Also read every `docs/*.md` file linked from Section 8 before editing that subsystem.
>
> **Multi-agent note:** this project has already been worked on by more than one AI agent —
> this file's edit history is from one continuous session, but a separate session (a different
> AI coding agent, "Antigravity", working directly on the user's machine) independently found
> and partially fixed Bug #12 first (see that bug entry). If you're a different agent/session
> picking this up: (1) check whether local, uncommitted changes exist that aren't reflected
> here yet — this file describes the state as of its last update, not necessarily every local
> edit the user has made since; (2) if you find evidence of a fix that contradicts what's
> written here, don't assume this file is right and the code is wrong — verify which is
> current, and update this file to match reality, not the other way around; (3) always add
> your own bug/decision entries here before finishing your session, using the same format, so
> the next agent (human or AI) has the same continuity you're relying on right now.

---

## 0. What this project is (one paragraph)

**SignXAI** — Handwritten Signature Verification Using Explainable AI. A signer enrolls
1-5 genuine signature samples. The system verifies a new signature against that person's
enrolled samples using classical ML/pattern-recognition (no pretrained model, no external
training dataset — every profile is self-calibrated from its own enrolled samples and
improves via examiner feedback). A local, optional vision-LLM narrates *why* the engine
decided GENUINE/FORGED/INCONCLUSIVE, grounded strictly in the engine's pre-computed metrics
(the LLM never makes the decision itself — see Section 6, Rule 1). Runs on Windows/macOS/Linux.

---

## 1. How to reason about this system (read this before editing anything)

- **Two-brain separation is the core architectural invariant.** The classical engine
  (`app/core/*`) makes the GENUINE/FORGED/INCONCLUSIVE decision and computes every metric.
  The vision-LLM (`app/llm/*`) is ONLY allowed to narrate metrics that already exist — it
  never independently re-judges validity. This exists because zero-shot VLM reasoning about
  forgery validity is empirically unreliable and prone to a "rationalization trap" (see
  `docs/ARCHITECTURE.md` §LLM boundary for the evidence). **Do not blur this line** — if you
  find yourself asking the LLM "is this genuine?", stop, that's the engine's job.
- **No pretraining of a per-person model, ever, on external signature datasets.** Every
  profile's mean/std/threshold is derived only from that profile's own enrolled samples
  (`app/core/engine.py::_relearn`). This is intentional (privacy, per-user calibration, no
  data-scarcity dependency) — don't "fix" it by adding a pretrained embedding model without
  updating this file first.
  **v2 refinement (not a violation):** `training/calibrate_weights.py` fits the ~7 fusion
  weights (how much ssim vs. hog vs. stroke-width-profile etc. matter when combined into one
  score) using logistic regression on a public labelled dataset. This is fitting a handful of
  coefficients over already-interpretable classical features — not training a black-box
  embedding network, and it never touches any individual person's profile data. See Section 3
  Bug/decision log below and `docs/TRAINING.md` for the full reasoning and a real validation
  run's results.
- **This file's "append, don't delete" rule is not decorative.** During the v2 update, an edit
  meant to append Bugs #6-#12 after Bug #5 accidentally replaced Bug #5's text instead — caught
  immediately by re-reading the diff before moving on, and fixed by restoring Bug #5 verbatim.
  Left here as a concrete example of why every edit to this file should be re-read, not assumed
  correct.
- **The engine is stateless; profiles/cases are the only state**, persisted as JSON under
  `data/profiles/` and `data/reports/` (see `app/core/storage.py`). Any new subsystem should
  read/write through `storage.py`, not invent a second persistence path.
- **Everything in `app/core/` must run with only `requirements.txt`** (numpy/opencv/scikit-image/
  scipy/fastapi — no torch, no LLM deps). `app/llm/` is the only place heavy/optional deps
  (`requirements-llm.txt`) are imported, and only inside function bodies (lazy import) so the
  core app runs even if the LLM stack isn't installed. Don't move a torch import to module
  level anywhere outside `app/llm/`.
- **Cross-platform is a hard constraint.** No bash-only scripts, no CUDA-only code paths
  without a non-CUDA fallback. The installer and launcher are Python (`install.py`, `main.py`),
  not `.sh`/`.bat`. This is why the LLM backend is llama.cpp/GGUF (works identically on
  Win/Mac/Linux, CPU or GPU) and not `transformers`+`bitsandbytes` (CUDA-only for 4-bit,
  breaks gracefully-degrading cross-platform story) — see Bug #1.

---

## 2. Verified facts (primary-source or self-run/user-confirmed only)

- **[self-run, 2026-07-20]** The uploaded prototype zip (`app/core/*`, `app/llm/*`) has ZERO
  entry point: no `main.py`, no FastAPI `app = FastAPI()` instance, no route file wired to
  `engine.py`, no web frontend, no CLI, no install script. This is the actual reason the user
  could not run it on Windows — there was nothing to launch, not a bash-portability issue per se.
- **[self-run, 2026-07-20]** `app/core/*` has no import of `torch`/`transformers` anywhere —
  confirmed clean separation, the "no pretrained model" requirement is already satisfied by
  the existing engine design.
- **[web search, 2026-07-20, ggml-org/llama.cpp + Google AI docs]** Gemma 4 (family: E2B, E4B,
  12B "Unified" encoder-free, 26B-A4B MoE, 31B dense) exists and is natively multimodal
  (text+image, audio on E2B/E4B/12B), with official QAT GGUF releases from `google/` on HF
  and community Dynamic-quant GGUFs from `unsloth/`. This is a real, current model family —
  confirms the user's requested default (`google/gemma-4-E4B-it-qat-q4_0-gguf` family) is valid.
  Per Unsloth docs: E2B QAT runs in ~3GB RAM, E4B ~5GB, 12B ~7GB, 26B-A4B ~15GB, 31B ~18GB
  (RAM+VRAM combined, unified or split). GGUF vision requires a matching `mmproj` file loaded
  alongside the main weights via `llama-cpp-python` or `llama.cpp` server.
- **[self-run, 2026-07-20]** ProtoSig paper (Pattern Recognition 180, 2026, uploaded by user)
  real numbers for CEDAR, global threshold: 2 genuine refs -> ~13.5% EER, 3 refs -> ~11.5% EER,
  5 refs -> ~9% EER, 10 refs -> ~6.7% EER. This sets our own honest accuracy expectations for
  the "1-5 enrolled samples" requirement — do not claim 90%+ accuracy at 2-3 samples anywhere
  in docs/UI copy; ~85-90% is the honest range per this evidence.
- **[web fetch, 2026-07-16, arXiv:2605.14845]** Zero-shot VLM signature verification (GPT-5.2 /
  Gemini 2.5 Pro tested): near-perfect on random forgery (different-person) detection, but
  35-48% EER on skilled forgery, and chain-of-thought reasoning made skilled-forgery detection
  *worse* (35%→48% EER) via a "Rationalization Trap" — model dismisses real forgery artifacts
  as natural variation while reporting 88-92% false confidence. This is the primary evidence
  for the two-brain separation rule in Section 1.

---

## 3. Bugs found and fixed (chronological, append-only, never delete)

### Bug #1 — LLM backend architecturally incompatible with stated model choice
- **Found:** 2026-07-20, self-audit of `app/llm/manager.py` and `app/device.py`.
- **Symptom:** `requirements-llm.txt` pins `torch`+`transformers`+`bitsandbytes`+`qwen-vl-utils`,
  and `device.py`/`manager.py` hardcode `Qwen/Qwen2.5-VL-*-Instruct` loaded via `transformers`
  with `BitsAndBytesConfig` 4-bit. `bitsandbytes` 4-bit quantization is CUDA-only — on macOS
  or Windows-without-NVIDIA it silently falls back to bf16 full-precision (much higher RAM/VRAM
  need than advertised), and the user explicitly asked for `google/gemma-4-*-qat-*-gguf` as
  the default, which is a **GGUF** model — GGUF is not loadable via `transformers` at all, it
  requires `llama.cpp`/`llama-cpp-python`.
- **Fix:** Replaced the LLM backend with `llama-cpp-python` (`Llama` + `--mmproj` multimodal
  handle), which runs identically (CPU or GPU) on Windows/macOS/Linux with no CUDA-only code
  path. Rewrote `app/device.py` model tier table to the Gemma 4 GGUF family (E2B/E4B/12B/
  26B-A4B/31B) with RAM+VRAM thresholds from the Unsloth QAT docs (see Section 2). Rewrote
  `app/llm/manager.py::_load`/`generate` accordingly. Updated `requirements-llm.txt` to
  `llama-cpp-python` + `huggingface_hub` only (removed torch/transformers/bitsandbytes/
  qwen-vl-utils — no longer needed).
- **Status:** Fixed in v1 (this delivery). `docs/ARCHITECTURE.md` updated to match.

### Bug #2 — No application entry point at all
- **Found:** 2026-07-20, self-audit (`find . -iname "main*.py" -o -iname "run*.py"` → empty).
- **Symptom:** `app/api/schemas.py` defines Pydantic request/response models but no file
  creates a `FastAPI()` instance or registers routes against them. `app/core/engine.py` and
  `app/llm/report.py` are fully-implemented but nothing calls them. No web frontend, no CLI.
- **Fix:** Added `app/api/routes.py` (wires `VerificationEngine` + report generator to REST
  endpoints), `main.py` (FastAPI app + static web frontend mount + uvicorn launcher),
  `cli.py` (Rich-based terminal interface, works standalone without the web server),
  `install.py` (one-command cross-platform dependency installer + hardware detection).
- **Status:** Fixed in v1 (this delivery).

### Bug #3 — Dead config flag implied a future core-engine torch dependency
- **Found:** 2026-07-20, self-audit (`grep -rn "USE_DEEP_EMBEDDING" app/` → only
  the declaration in `config.py`, zero usages anywhere else).
- **Symptom:** `app/config.py` declared `USE_DEEP_EMBEDDING` (env-toggle, default
  on) with a comment "Optional deep embedding (torchvision)" but it was never
  read by `engine.py`, `features.py`, or anywhere else. Dead code that implies
  a torch dependency was planned for the core engine — directly conflicts with
  process rule #5 (heavy deps stay in `app/llm/` only) and Section 1's "no
  pretraining, ever" invariant.
- **Fix:** Removed the flag, replaced with an explanatory comment pointing back
  here so nobody silently re-adds a torch import to `app/core/` without an
  explicit decision recorded in this file first.
- **Status:** Fixed in v1 (this delivery).

### Bug #4 — `install.py` failed outright on a real, common setup (PEP 668)
- **Found:** 2026-07-20, self-run (`python3 install.py` in this sandbox —
  Debian-based Python) → `error: externally-managed-environment`.
- **Symptom:** Modern Linux distros (Debian 12+/Ubuntu 23.04+) ship Python
  with PEP 668 protection that refuses `pip install` outside a virtual
  environment. The first version of `install.py` called pip directly against
  the system interpreter — this is exactly the class of "gets stuck, can't
  install" problem the user reported for Windows, except this instance hit
  it on Linux during self-testing, which is good evidence the fix needed to
  be general, not OS-specific.
- **Fix:** `install.py` now auto-creates a local `.venv/` on first run and
  re-execs itself inside it (`_ensure_venv_and_reexec`), before touching pip
  at all. Idempotent — safe to re-run. Final instructions print the exact
  venv-python path to use for `main.py`/`cli.py` afterward so dependencies
  stay isolated and the "one command" promise holds on Windows, macOS, AND
  modern Linux.
- **Status:** Fixed in v1 (this delivery), verified by re-running the
  installer end-to-end after the fix (see Section 4 for the test transcript).

### Bug #5 — `verify()` crashed on every real request (grayscale ndarray into a BGR-only preprocessor)
- **Found:** 2026-07-20, self-run end-to-end test: enrolled a real CEDAR
  signature via the live server, then called `/api/verify` with a real
  forged CEDAR signature → `cv2.error: Bad number of channels ... scn is 1`.
- **Root cause:** `preprocess.load_image()` returns any `np.ndarray` input
  unchanged, assuming it is 3-channel BGR. But `engine.py::verify()` reloads
  each saved reference with `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` — a
  2D single-channel array — before calling `preprocess.preprocess(r_img)`.
  `preprocess()` unconditionally calls `cv2.cvtColor(bgr, COLOR_BGR2GRAY)`,
  which requires 3 channels and crashed on every single verification call.
  This means **verification never worked in the original prototype**, even
  once an entry point existed — this was caught only by actually running
  the enroll→verify flow with real images, not by reading the code (the
  bug is in the interaction between two files, invisible file-by-file).
- **Fix:** `load_image()` now detects a 2D (grayscale) ndarray and converts
  it to 3-channel BGR before returning, so any caller passing a
  already-grayscale array (however it got that way) is handled correctly,
  not just this one call site.
- **Status:** Fixed in v1. Verified by re-running the exact same
  enroll→verify request that originally crashed — see Section 4.
- **Lesson for future sessions:** file-by-file code review (which is how
  Bugs #1-4 were found) does NOT catch cross-file contract mismatches like
  this. Process rule #7 (Section 6) exists because of this bug: **before
  shipping any version, run at least one real enroll→verify round-trip
  with real image data**, not just import/syntax checks. (v2 note: Bugs
  #6-#8 below were found by following this exact rule, using real
  user-submitted images and reports rather than synthetic test data —
  the rule earned its keep.)

### Bug #6 — Fixed 1e-3 z-score floor caused +1000 blown-up z-scores, biasing genuine signatures toward FORGED
- **Found:** 2026-07-23, from a REAL user-generated PDF report showing
  "Enclosed loops ... z = +1000.00" — an impossible value that was the
  smoking gun. Root cause in `similarity.py::scalar_profile_sim`: std floor
  was a fixed `1e-3`. Integer topology features (enclosed loop count,
  connected components) are often IDENTICAL across a person's own few
  enrolled samples (std=0), and 1e-3 is far too tight a floor — a 1-unit
  difference produced z=1000, and `exp(-1000) == 0.0`, so that one feature
  alone could veto an otherwise-genuine signature.
- **Fix:** relative std floor (8% of the feature's own mean magnitude,
  scale-aware instead of a fixed tiny constant) AND a hard z-score cap at
  ±6.0, so no single feature can ever zero out the score outright. Same
  `1e-3` floor also removed from `engine.py::_relearn`'s std storage (was
  cosmetically inflating the displayed reference std in reports for no
  benefit, now removed).
- **Status:** Fixed in v2. Verified: max observed z-score is now exactly
  6.0 (the cap), not 1000+.

### Bug #7 — Threshold calibration used a DIFFERENT formula than actual verification, so the threshold didn't mean what it claimed to mean
- **Found:** 2026-07-23, while investigating Bug #6: even after fixing the
  z-score blowup, re-verifying an ENROLLED reference against its own
  freshly-built profile still scored *below* its own calibrated threshold
  — which should be close to impossible.
- **Root cause:** `engine.py::_relearn` computed the profile's threshold
  using `_pair_score()`, a crude proxy that skips real pixel alignment and
  approximates ssim/iou/dtw as a rough average of two other terms. Actual
  verification (`_score_against_ref`) computes REAL ssim/iou/dtw from
  properly aligned images. Two different formulas producing two different
  score distributions — the threshold, calibrated on one formula, was
  being compared against scores from a different formula.
- **Fix:** `_relearn` now calibrates the threshold using the EXACT SAME
  `_score_against_ref` function real verification uses (reloading and
  aligning each pair of enrolled references), so threshold and query
  scores are directly, honestly comparable. `_pair_score` (the old proxy)
  was deleted entirely rather than left as dead/misleading code.
- **Status:** Fixed in v2.

### Bug #8 — References were preprocessed TWICE, corrupting the binary mask (the biggest single contributor to false-FORGED results)
- **Found:** 2026-07-23, isolated by directly testing self-comparison
  (an image against itself) through the exact disk round-trip
  `enroll()`/`verify()` use: IoU dropped from a perfect 1.0 to 0.126, and
  DTW similarity from 1.0 to 0.26, on two copies of the literal same image.
- **Root cause:** `enroll()` called `preprocess.preprocess(raw)` once,
  producing `pre.gray` (already deskewed/cropped/Otsu-binarized/normalised),
  and saved THAT to disk. `verify()` and `_relearn()` then reloaded that
  already-processed image and called `preprocess.preprocess()` AGAIN —
  running Otsu binarization a second time on an image whose intensity
  histogram is now mostly flat white background + already-normalised ink,
  which is a very different histogram than the original raw photo. Otsu's
  automatic threshold selection produced a materially different (much
  worse) binary mask the second time.
- **Fix:** `enroll()` now saves the ORIGINAL decoded image (before any
  processing) instead of the post-preprocess gray. Every reload site
  (`verify()`, `_relearn()`) now loads as color and runs `preprocess()`
  exactly once on genuinely-raw-equivalent data — methodologically
  identical to how the query is processed. Verified: the same
  self-comparison test now gives a perfect ssim/iou/hog/dtw = 1.0.
- **Status:** Fixed in v2. This was very likely the single largest
  contributor to the "genuine signature marked forged" behavior the user
  reported, since it affected EVERY verification (every reference reload
  went through this corruption), not just edge cases.
- **Combined impact of Bugs #6+#7+#8, verified on real user-uploaded images:**
  before fixes, the user's own enrolled genuine signature images scored
  FORGED. After all three fixes, re-tested with the same real images:
  enrolled genuine images now correctly score GENUINE (0.90 vs 0.71
  threshold), and a held-out different-style signature and independent
  CEDAR forgery samples both still correctly score FORGED — confirming the
  fixes resolved the false-positive bias without breaking real forgery
  detection.

### Bug #9 — LLM status permanently showed "error" instead of "not installed" on a perfectly normal machine
- **Found:** 2026-07-22, from the user's real Windows screenshot showing
  "LLM: error" in the UI pill on a machine that simply hadn't installed the
  optional `llama-cpp-python` dependency yet — not an actual error.
- **Root cause:** `manager.py::_load()` caught ALL exceptions (including
  `ModuleNotFoundError` for the simply-not-installed case) and set
  `status = "error"` unconditionally. Since every `verify()` call triggers
  a report generation attempt, this got set on the very first verification
  and stayed "error" forever after.
- **Fix:** `ModuleNotFoundError`/`FileNotFoundError` (dependency or weights
  missing — both expected, actionable states) now set `status =
  "not_installed"`, distinct from genuine unexpected failures which still
  set `status = "error"`. Also added a short-circuit in `report.py` so it
  doesn't waste 3 retry attempts calling an import that will fail
  identically every time once the status is already known.
- **Status:** Fixed in v2. Verified: status now correctly reports
  `"not_installed"` with a helpful message, not `"error"`.

### Bug #10 — `install.py --llm` failed on Windows even with the GPU-wheel logic from v1
- **Found:** 2026-07-22, from the user's real Windows install log: the
  CUDA wheel index (`cu121`) didn't have a prebuilt wheel for the specific
  `llama-cpp-python` version pip resolved to (open `>=0.3.4` constraint
  resolved to `0.3.34`, which wasn't published on that index), so pip
  silently fell back to a SOURCE build. The v1 "CPU fallback" was even
  worse: it used **plain PyPI**, which has no prebuilt wheels for
  llama-cpp-python at all (source-only) — reproducing the exact
  `CMAKE_C_COMPILER not set` crash the user hit, since this sandbox
  environment (like most Windows machines) has no C++ compiler installed.
- **Fix:** pinned to a specific version (`0.3.16`) known to be published on
  abetlen's prebuilt wheel indices, and the installer now tries, in order:
  CUDA wheel indices (cu121/cu122/cu124) if an NVIDIA GPU is detected, then
  the **prebuilt CPU wheel index** (`.../whl/cpu` — still no compiler
  needed, works on any OS) — and only as an absolute last resort, a
  from-source build with an explicit compiler-requirement warning. Never
  silently falls through to plain PyPI anymore.
- **Status:** Fixed in v2's `install.py`. **Not yet re-verified on an
  actual Windows machine** (this sandbox has no Windows target) — this is
  the top item for the user to re-test and confirm.

### Bug #11 — Overlapping text labels on the annotated evidence image
- **Found:** 2026-07-22, from the user's real PDF report screenshot: two
  finding labels ("Ref: enclosed loops", "Ref: contour perimeter...")
  rendered directly on top of each other, unreadable.
- **Root cause:** `explain.py::annotate_regions` placed each finding's text
  label at a fixed offset from its own bounding box with no collision
  detection against previously-placed labels. When multiple findings land
  in the same region (common — the template groups several deviant
  features per pixel region), labels stacked directly on top of each other.
- **Fix:** added simple collision detection — each new label is pushed
  down past any previously-placed label it would overlap, bounded to 20
  attempts and clamped to stay on-canvas. Also added a small white backing
  rectangle behind each label so it stays legible over ink or other boxes.
- **Status:** Fixed in v2.

### Bug #12 — Model tier recommendation ignored what was actually already downloaded
- **Found:** 2026-07-23/24, via a parallel debugging session the user ran
  independently using a different AI coding agent ("Antigravity") directly
  on their Windows machine (see `antigravity_edit.md`, incorporated here —
  re-implemented independently rather than pasted verbatim, but the root
  cause and fix approach are the same, and credit belongs to that session
  for finding it first).
- **Symptom:** `device.py::choose_model()` picked a model tier purely from
  hardware specs (RAM+VRAM), with no awareness of what was actually sitting
  in `data/models/`. A machine with enough RAM/VRAM to qualify for a bigger
  tier than what the user had already downloaded would get recommended (and
  `manager.py` would try to load) a model that was never fetched —
  `FileNotFoundError`, even though a perfectly good smaller model was
  already on disk and ready.
- **Fix:** `choose_model()` now checks `MODELS_DIR` for an already-downloaded
  model FIRST, before falling back to a fresh hardware-tier recommendation.
- **Status:** Fixed in v3.

### Bug #13 — Small-sample threshold had ZERO slack with exactly 2 references (the core v3 report)
- **Found:** 2026-07-24, directly from the user's report: their own genuine
  signature was marked invalid, and they correctly diagnosed the mechanism
  themselves — "system makes scoring threshold high" — before I even
  looked at the code.
- **Root cause:** with exactly 2 enrolled references there is exactly ONE
  intra-reference pairwise comparison. `numpy.std()` of a single value is
  always exactly 0.0. The threshold formula (`mean(intra) - 2*std(intra)`)
  therefore became precisely that one observed score, with NO slack for
  natural variation at all — any legitimately-genuine second signing that
  differed even slightly from the two enrolled samples' mutual similarity
  scored below threshold.
- **Fix:** shrinkage estimator — blend the observed std toward a
  conservative, hand-set prior floor (`_PRIOR_STD_FLOOR = 0.06`, NOT fit on
  any external dataset — derived from the actual spread of genuine
  self-comparison scores observed while fixing Bugs #6-#8) when few
  reference pairs exist, fading the prior out as more real references
  accumulate (fully trusts observed std once ~4 references / 6 pairs are
  enrolled). This is exactly the kind of fixed, inspectable constant the
  user's request implicitly endorsed, as opposed to the dataset-trained
  weights they asked to de-prioritize (see the "De-prioritizing dataset
  training" decision below).
- **Verified on a 10-writer CEDAR sweep** (held-out genuine sample + a
  forgery sample per writer, none of which were enrolled): 0 false
  negatives on genuine signatures (was the core reported bug — now zero
  across the sweep), forgery detection holds with some borderline forgeries
  correctly routed to INCONCLUSIVE rather than either confident verdict.
  Full honest numbers in Section 4.
- **Status:** Fixed in v3, on real data.

### Bug #14 — The "uncertain zone" only existed below threshold, not above
- **Found:** 2026-07-24, during the Bug #13 sweep test: a CEDAR forgery
  scored only 0.036 above a loosely-calibrated 2-reference threshold and
  was labeled confidently GENUINE — because the INCONCLUSIVE band only
  ever applied to scores just BELOW threshold, never just above.
- **Fix:** made the band symmetric — scores within `INCONCLUSIVE_BAND` on
  EITHER side of threshold route to human review, not just the below side.
  This matters more now that Bug #13 intentionally widens thresholds for
  low-sample profiles, since more scores land close to threshold on both
  sides.
- **Status:** Fixed in v3. Confirmed this exact forgery case now correctly
  lands INCONCLUSIVE instead of wrongly GENUINE.

### Bug #15 — EXIF photo orientation was never honored
- **Found:** 2026-07-24, from the user's ask #3 ("not get mirror or in any
  other angle"). Not reproduced with a specific failing image in this
  session (the test images available didn't carry EXIF orientation data),
  but this is a well-established real-world bug class for any photo-upload
  pipeline: `cv2.imdecode` ignores EXIF orientation tags entirely, so a
  phone photo whose pixel data is stored sideways/upside-down (common —
  many phones store the sensor orientation as metadata rather than
  rotating pixels) would silently be processed in the wrong orientation
  with no error.
- **Fix:** `load_image()` now always decodes through PIL first and applies
  `ImageOps.exif_transpose()` before handing off to OpenCV, falling back to
  the old raw-decode path only if PIL fails.
- **Deliberately NOT done:** full deskew-to-horizontal for every signature.
  A signature's natural cursive slant is a real discriminative feature the
  engine measures directly (`pca_orientation_deg`) — auto-rotating every
  signature to canonical horizontal would destroy that signal. Only
  genuine orientation errors (EXIF-tagged sideways photos) are corrected,
  not each person's natural handwriting angle. This is a considered
  architectural decision, not an oversight — documented here so it isn't
  "fixed" into a regression later.
- **Status:** Fixed defensively in v3; not directly reproduced with a
  failing test case in this session (no EXIF-carrying test image was
  available) — worth the user confirming with an actual phone photo that
  needed rotation.

### Bug #17 — `install.py --llm` had a race condition that could silently abort the download
- **Found:** 2026-07-24, from the user's report that `install.py --llm`
  "still has an issue" while the web page's Download button works fine —
  investigated why those two paths, which call the same underlying
  `download()` method, would behave differently.
- **Root cause:** `manager.py::download()` starts a background **daemon**
  thread and returns immediately (correct design for the web server, which
  stays running indefinitely, so the thread has all the time it needs).
  `install.py` then polled `mgr.status()` in a loop, waiting for it to stop
  being `"downloading"`. Race condition: if the poll loop's first check ran
  before the daemon thread had actually started and set status to
  `"downloading"` (a real possibility — thread scheduling isn't
  instantaneous), the loop would see the pre-download status, treat it as
  "already finished," exit the loop immediately, print `[OK] Model ready`,
  and `install.py`'s `main()` would return — terminating the Python
  process. Daemon threads are killed the instant their parent process
  exits, so the download could be silently aborted moments after starting,
  with the script having already claimed success. The web UI never hits
  this because the server process (and thus the thread) stays alive for as
  long as the user keeps the app running.
- **Fix:** added `VisionLLMManager.download_blocking()` — no thread, runs
  the download synchronously in the calling process. `install.py --llm`
  and `cli.py`'s interactive download option both switched to this method.
  huggingface_hub's own progress output prints naturally since nothing is
  backgrounded. The web UI's download button still correctly uses the
  async `download()` (appropriate there, since the server process
  legitimately outlives the request).
- **Verified:** confirmed the model repo (`google/gemma-4-E2B-it-qat-q4_0-gguf`)
  is real and `download_blocking()` reaches it correctly — this sandbox's
  network policy blocks `huggingface.co` specifically (not a code bug),
  producing a clean, immediate, non-hanging error rather than a silent
  abort. **Full download-to-completion not verified** (no network access
  to huggingface.co in this sandbox) — top item for the user to confirm.
- **Status:** Fixed in v3.

### Bug #18 — LLM install STILL failed on Windows despite two previous "fixes" (v2 Bug #10, v3 wheel-index retry) — root cause finally found and verified
- **Found:** 2026-07-25, from the user's real Windows install log, showing
  every single wheel-index attempt (cu121, cu122, cu124, cpu) falling back
  to a from-source build, and the web UI simultaneously trying to download
  the wrong model tier (31B instead of the intended default).
- **Investigation (this time done properly — empirically verified, not
  assumed):**
  1. Fetched abetlen's actual wheel index pages directly. Found the
     previously-pinned version (`0.3.16`) simply isn't published on ANY of
     them — it never existed as a prebuilt wheel, so every attempt was
     guaranteed to fail regardless of which index was tried. The v2 "fix"
     that introduced this pin was never verified against the real index —
     an assumption presented as a fact. This is the second time this
     project shipped an unverified version pin as if it were checked (see
     Bug #10's postmortem) — see process rule addition below.
  2. Checked the CUDA index specifically: **official CUDA wheels have been
     Linux-only since v0.3.20** — no Windows CUDA wheel exists for ANY
     current version. Every previous Windows CUDA attempt (this session
     and prior ones) was structurally impossible to succeed, independent
     of version pinning.
     **[v5 correction, Bug #25]: this specific claim was re-checked and
     could not be confirmed against the project's current, live
     documentation (README/PyPI/ReadTheDocs list `cu118`-`cu132` backends
     without stating an OS restriction) — it may have been an
     over-generalization from what pip happened to resolve in one earlier
     session, not a verified platform limitation. Treat this "Linux-only"
     line as superseded; see Bug #25 for what's used now.**
  3. Even after correcting the version pin to `0.3.34` (verified present
     on the index pages) and testing `pip install --extra-index-url
     .../whl/cpu llama-cpp-python==0.3.34`, pip STILL resolved to the
     71MB source distribution instead of the real 23MB prebuilt wheel that
     genuinely exists at that index. Root cause of *this* layer: unreliable
     behavior in how pip resolves packages against a GitHub-Pages-hosted
     PEP 503 index in this environment — confirmed by testing a DIRECT
     wheel URL (bypassing index resolution entirely), which downloaded and
     installed the correct 23MB wheel immediately and correctly.
- **Fix:** `install.py` now builds a direct GitHub Releases wheel URL for
  the current OS/arch (verified against the real release asset naming
  pattern) and installs from that URL directly — no index resolution
  involved at all, nothing for pip to get ambiguous about. Verified
  end-to-end in this session: real 23.2MB wheel downloaded, installed,
  `import llama_cpp` succeeds, `Llava15ChatHandler` imports, and
  `VisionLLMManager` correctly recognizes and reports on it.
- **Also fixed the wrong-model-tier download** (web UI trying to fetch the
  31B model): per explicit user instruction, the entire multi-tier model
  system was removed. There is now exactly ONE model this project ever
  downloads (`google/gemma-4-E4B-it-qat-q4_0-gguf`, see `app/device.py`).
  "Which model is recommended" and "which model is installed" are now the
  same question by construction — eliminates this entire bug class rather
  than patching around it.
- **Status:** Fixed and verified in v4, in THIS sandbox (Linux, CPU-only
  path). **Windows path uses the same direct-URL mechanism with a
  different filename** (`win_amd64` instead of `manylinux2014_x86_64`) —
  logically identical and should work the same way, but only the Linux
  path was directly testable here. Top item for the user to confirm.

### Tested and rejected (v4) — consensus-penalty for majority-divergent components
- **Motivation:** user's real INCONCLUSIVE report (score 0.6493 vs threshold
  0.6324, case d5972492d296) showed 4+ features with severe/capped
  z-scores (several Hu moments at the ±6.0 cap, enclosed loops z=+6.0,
  stroke width z=-4.99) yet the fused score barely cleared into
  INCONCLUSIVE. Signature-verification ensemble literature (searched this
  session) consistently uses majority-voting specifically because plain
  linear-weighted averaging can dilute a real consensus signal when several
  independent methods agree but a few (typically the pixel-level ones) stay
  moderate.
- **Implemented:** a bounded penalty in `_score_against_ref` that only
  engaged when a majority (4+ of 7) of independent comparison components
  scored below 0.5, scaled by how far past majority.
- **Tested honestly, on the same 10-writer CEDAR sweep used for v3's
  baseline** (direct before/after comparison, not a new/cherry-picked test):
  ```
  v3 baseline:      genuine_correct=6  forged_correct=3  forged_wrong=2  inconclusive=9
  v4 w/ consensus:  genuine_correct=5  forged_correct=2  forged_wrong=2  inconclusive=11
  ```
  **No improvement** — genuine detection got slightly worse, forgery
  detection did not improve, and MORE cases became INCONCLUSIVE (the
  opposite of what the user asked for: fewer, more confident decisions,
  not more hedging). `forged_wrong` — the specific failure mode this was
  meant to fix — was unchanged.
- **Decision: reverted.** Literature-motivated reasoning is not a
  substitute for empirical validation on this specific system — this is a
  concrete example of the process rule "verify, don't assume" applying to
  an algorithm change, not just a dependency pin. Left here, not deleted,
  so nobody re-tries the same idea without seeing it already failed here.
- **What the user's specific report case actually was:** re-examined
  honestly — score 0.6493 vs threshold 0.6324 is a genuine ±0.017 margin,
  comfortably inside the ±0.07 INCONCLUSIVE band by design. The system
  behaved as designed, not incorrectly. The underlying tension is real
  though: a profile calibrated from few (2-3) references has a
  wide-enough threshold uncertainty that even fairly divergent-looking
  cases can land inside the band. The most effective actual lever remains
  what's already documented: more enrolled references narrow the band
  (see Bug #13's shrinkage fix — the prior floor's influence fades as
  reference count grows).

### Bug #19 — (placeholder for next bug found; keep appending here, do not delete prior bugs)

### Bug #20 — Alignment trusted ORB unconditionally; a spurious rotation could pass with no check that it actually improved overlap

- **User's report:** "I get to see that images get rotated and not able to
  properly overlap in sample image & heatmap." — the divergence
  heatmap/overlay/skeleton images looked wrong, with the query visibly
  rotated relative to the reference.
- **Root cause (confirmed by reading `align.py`):** the v4 aligner ran ORB
  keypoint matching, and if RANSAC found >=8 inlier matches, used the
  resulting affine AS-IS — no check that the transform actually made the
  two ink masks overlap better than doing nothing at all. ORB is a
  corner/blob detector built for richly-textured natural photos; a
  pen-stroke signature on white paper has almost none of that texture, so
  the handful of "keypoints" ORB finds are weak, ambiguous stroke-edge
  points. With RANSAC fitting a full affine (rotation+scale+shear+
  translation) to ~8 such points, it can converge on a plausible-looking
  but SPURIOUS transform, including large, wrong rotations. The alignment
  "succeeded" by the old definition (enough matches found) while producing
  a bad transform — exactly what the user saw.
- **Fix (`align.py`, fully rewritten):** generate several candidate
  transforms — identity, centroid-translation-only, an image-moments/PCA
  rotation (uses every ink pixel, not a handful of ambiguous corners), and
  the original ORB+ECC — clip every candidate's implied rotation to
  ±25° (signatures are captured near-upright by design; a bigger implied
  rotation is far more likely a bad fit than a real photo angle), then
  actually SCORE every candidate by ink IoU against the reference and keep
  the highest-scoring one. This makes alignment monotonic: the chosen
  transform is structurally guaranteed to match-or-beat doing nothing.
  `aligned_ok` now means "the winner is non-identity AND beat identity IoU
  by a real margin," not merely "ORB found enough points."
- **Also fixed as part of the same investigation:** DTW similarity
  (`engine.py::_score_against_ref`) was comparing the UNALIGNED query
  binary against the reference, while every other pixel-level term (ssim,
  iou) used the ALIGNED version — an inconsistency that meant DTW's
  contribution to the fused score didn't benefit from alignment at all.
  Now uses the same aligned binary as ssim/iou.
- **Verified:** a genuine CEDAR image rotated 12° synthetically (to
  simulate a skewed photo) went from 0.085 raw ink IoU against its
  reference to 0.167 after alignment, with a visual overlay check showing
  the strokes correctly overlapping (green = matched ink) rather than
  offset — see `docs/BUILD_LOG_v6.md` v5 section for the test script and
  images.

### Bug #21 — Recognition only ever used ONE reference for both the decision and every explanation asset, and the decision could flip on which one was picked

- **User's report:** "compare each sample signature and combining each
  one generates a metric data to verify new images... it only provide
  feedback for sample images which we upload not on new & other upload
  Questioned signature... Questioned signature have to goes through each
  sample image."
- **Root cause (confirmed by reading `engine.py::verify()`):** the v4
  loop DID compare the query against every enrolled reference internally,
  but then kept only `max(results, key=lambda r: r[0])` — the single
  best-scoring reference — and used ONLY that one comparison for the
  verdict, the score, and every explanation asset (heatmap, overlay,
  skeleton). One unusually good or unusually bad reference match could
  single-handedly flip GENUINE/FORGED, and the examiner had no visibility
  into how the other references scored at all.
- **Fix (`engine.py::verify()`, rewritten):**
  1. Score the query against EVERY enrolled reference (unchanged from v4)
     — but now KEEP every individual result, not just the best.
  2. Build an aggregate **prototype** score in the spirit of ProtoSig
     (`REFERENCES.md` paper 1): average each fused COMPONENT (ssim, hog,
     ...) across every reference comparison first, then fuse with the
     same weights. This lets one reference's strong ssim compensate a
     different reference's strong hog, the way a real multi-sample
     prototype should — this is the aggregate "sum of all sample images"
     6th data point the user asked for (5 refs + 1 combined profile
     score).
  3. Fused decision score = `0.5 × best-individual-match + 0.5 ×
     prototype`. A documented, inspectable 50/50 blend, not fit on any
     dataset. Pure `max()` (v4 behaviour) lets one lucky reference
     override real disagreement elsewhere; pure `mean()` unfairly
     penalises a genuine signer whose query naturally resembles one
     enrolled style more than another — the blend was chosen to avoid
     both failure modes.
  4. Added `_edge_case_review()`: an INDEPENDENT second pass over the
     distribution of the N individual reference comparisons, run after
     the fused verdict, that can only ever downgrade toward INCONCLUSIVE
     (never invert GENUINE<->FORGED). Flags real opposing evidence (some
     individual comparisons land on the genuinely opposite side of the
     threshold band from the fused call) and unusually high dispersion
     relative to the margin from threshold, or unreliable alignment
     across most references.
  5. `case["reference_scores"]` now reports every individual reference's
     score/alignment/components, `case["aggregate"]` reports the
     prototype and summary stats, and a new asset
     (`explain.reference_agreement_chart`) renders a bar chart of the
     query's score against every reference plus the prototype — direct,
     visible evidence that the questioned signature really was compared
     against every sample image, addressing "it only provides feedback
     for sample images... not the query" at its root: previously no
     artifact showed the query compared to more than one reference.
- **Dev-time self-correction, logged honestly (not glossed over):** the
  first version of `_edge_case_review()` counted ANY individual verdict
  that didn't exactly match the fused label as disagreement — including a
  reference landing in the merely-INCONCLUSIVE band around a clearly
  GENUINE fused call, which is normal per-reference variability, not
  contradiction. A 10-writer/5-reference-each CEDAR sweep caught this
  directly: genuine, well-separated cases were being downgraded to
  INCONCLUSIVE at a high rate (writer 1's held-out genuine sample scored
  0.679 against a 0.587 threshold — clearly above the band — yet was
  downgraded because 3 of 5 individual comparisons merely fell in the
  INCONCLUSIVE band rather than the GENUINE band). Retuned to only count
  OPPOSING evidence (an individual comparison on the literal opposite
  side of the threshold from the fused verdict) as real disagreement.
  Re-swept after the fix: genuine-correct went from 3/10 to 5/10 with
  genuine-wrong still 0/10 (see Section 4 for the full before/after
  numbers). This kind of self-caught, re-tested correction is exactly
  the process this file exists to make visible.
- **Honest limitation, found in the same sweep, not fixed (needs a real
  decision, not a threshold hack):** skilled-forgery detection remains
  weak — 0/10 forgeries were confidently called FORGED in the same sweep,
  and 3/10 scored high enough to be called GENUINE outright (the rest
  landed INCONCLUSIVE, which is the safe outcome but not a confident
  catch). This is a known, well-documented limitation of classical/
  handcrafted features (HOG/LBP/Hu/SSIM/DTW) against SKILLED forgeries
  specifically — by definition a skilled forgery closely mimics overall
  shape, which is exactly what these features are least sensitive to
  distinguishing; it's the literature's own stated reason papers 3-5
  (`REFERENCES.md`) moved to trained Siamese/CNN embeddings. Retuning the
  fusion weights or threshold against this one 10-writer sweep would be
  overfitting to a single test set, not a real robustness fix — logged
  here as an open item for Section 7, not silently patched around.

### Bug #22 — LLM manager never checked the filesystem for an already-present model; a manually-copied model was permanently invisible

- **User's report:** "I had LLM already downloaded the model in previous
  version codebase & so i copy it and pastes inside that particular path
  - but the system is not able to locate and recognize that LLM is
  already in system... when web page is launch - it check two things [1]
  model path & size [2] gpu is available or not - and start to load model
  on gpu."
- **Root cause, a genuine deadlock (confirmed by reading `manager.py` and
  `report.py` together):** `report.py`'s gate only attempted the LLM when
  `status` was already `"ready"` or `"loading"` — but NOTHING ever set
  status to `"ready"` except a load that only happened INSIDE
  `generate()`, which itself is only called from behind that exact same
  gate. A model correctly sitting on disk, on perfectly viable hardware,
  could never transition out of `"not_installed"` through this path at
  all — no matter how long it sat there. Separately, `download_blocking()`
  set status back to `"not_installed"` even after a SUCCESSFUL download,
  which was simply incorrect.
- **Fix:**
  1. `device.py::model_present_on_disk()` — a real, direct filesystem
     check (sums bytes under the model's cache directory, no cached
     in-memory flag) against `MODEL_PRESENT_MIN_GB = 4.5` (slightly below
     the user's stated 5GB so minor version/quantization file-size
     differences don't cause a false negative on a genuinely-present
     model).
  2. `manager.py::refresh_device()` now calls this check every time and
     sets a new `"downloaded"` status (weights present, not yet loaded
     into memory) instead of leaving status stuck.
  3. `manager.py::autostart()` — the exact two checks the user asked for,
     run once at process start: model path+size (`model_present_on_disk`)
     and GPU availability (`detect_device`'s existing `nvidia-smi` probe).
     If both are satisfied, kicks off the actual GGUF load in a
     background thread so server startup itself is never blocked.
  4. `main.py` — a FastAPI startup hook calling `autostart()`.
  5. `report.py`'s gate now also accepts `"downloaded"` status (triggers
     a lazy load on first use if autostart's background thread hasn't
     finished yet), and a manual `POST /api/llm/load` endpoint + UI
     button were added as an explicit retry path.
- **Verified (this build sandbox has no real 5GB+ model or viable
  hardware, so this was tested with a synthetic fake-sized model
  directory and a monkeypatched viable device profile — see
  `docs/BUILD_LOG_v6.md` v5 section for the exact test script):**
  `model_present_on_disk()` correctly returns `False` for both a missing
  directory and a near-empty one, and `True` (with the right GB reading)
  once a >=4.5GB file is actually present; `refresh_device()` correctly
  transitions status to `"downloaded"` the moment that's true;
  `autostart()` correctly attempts a load and fails SAFELY (logged,
  actionable error, no crash) when the optional `llama-cpp-python`
  package isn't installed — exactly the failure mode this sandbox has,
  and exactly the failure mode that must never take the server down.

### Bug #23 — Feedback was a threshold nudge and a popup; no real training/adaptation signal existed

- **User's report:** "There is no visible TTC & Training sign in system.
  When we provide feedback - so that model has to make use of it and
  train the system... Right now - it is just providing message of pop up
  nothing more."
- **Root cause:** confirmed in `web/app.js::sendFeedback()` — a single
  `alert()` call with a canned string, nothing persisted to the visible
  UI. The backend (`engine.py::feedback()`) DID already nudge the
  threshold/weights (this part was real, just invisible), but had no
  mechanism to grow or otherwise meaningfully adapt the profile itself.
- **Fix:**
  1. **Real test-time calibration (TTC), not just a nudge:**
     `engine.py::feedback()` now accepts `add_as_reference` — when the
     examiner confirms a GENUINE verdict was correct and opts in
     (checkbox in the UI, defaulted OFF), the query signature is promoted
     into the profile as a new reference sample and the whole profile is
     relearned (`_relearn()`: mean/std/threshold/intra-scores) from it.
     This is the standard "incremental enrollment" idea from the
     signature-verification literature — a prototype that grows with more
     confirmed-genuine samples becomes more representative of the
     signer's real variation (see `REFERENCES.md` paper 1's discussion of
     few-shot calibration improving with more references). Capped at
     `MAX_REFERENCES = 12` and NEVER automatic — always requires the
     examiner's explicit confirmation AND opt-in, so a forged sample can
     never poison a profile through this path.
  2. **Visible, persistent training panel** (`web/app.js`) replaces the
     alert-only popup: shows the updated adaptive threshold, profile
     maturity, feedback count, reference count, and whether this specific
     feedback event was promoted to a new reference — stays on screen
     after the click, doesn't disappear like a toast.
  3. `api/schemas.py::FeedbackRequest` gained `add_as_reference: bool =
     False` (backward compatible — existing callers get v4 behaviour
     unless they opt in).
- **Verified end-to-end** (live server, real CEDAR image): enrolled 5
  references, verified a held-out genuine query (correctly GENUINE),
  submitted feedback with `add_as_reference: true` — profile went from 5
  to 6 references, threshold and weights visibly updated, `maturity`
  incremented by 2 (vs 1 for ordinary feedback) to reflect the larger
  adaptation. See `docs/BUILD_LOG_v6.md` v5 section for the full request/
  response transcript.


### Bug #24 — `install.py`'s venv creation could silently produce a venv with no pip, and `run.bat`/`run.sh` broke if not run from inside the project folder

- **User's report (with real terminal output):** `python install.py` on
  Windows created `.venv`, printed no error, then immediately failed
  re-running itself inside it:
  ```
  >>> Re-running installer inside the virtual environment...
  D:\signxai\.venv\Scripts\python.exe: No module named pip
      [FAILED] Installing core engine dependencies ... (exit code 1)
  [FATAL] Core dependency install failed. See errors above.
  ```
- **Root cause:** `venv.EnvBuilder(with_pip=True).create(VENV_DIR)` is
  *supposed* to guarantee pip exists in the new environment (it runs
  `ensurepip` internally), but it does not reliably surface or raise when
  that internal bootstrap is silently incomplete. This is a real,
  documented category of Windows failure -- common causes are antivirus/
  security software quarantining pip's bundled wheel files during
  creation, or a base Python installation whose own `ensurepip` module is
  itself incomplete (seen on some minimal/corporate Windows Python
  installs). `install.py` trusted `with_pip=True` unconditionally and
  went straight to `pip install -r requirements.txt` on the next line --
  guaranteed to fail with a confusing low-level error instead of a clear
  diagnosis.
- **Fix (`install.py`):** added `_venv_has_pip()` (a real, direct check --
  runs `python -m pip --version` inside the venv, doesn't trust the
  creation call) and `_bootstrap_pip()`, called both right after venv
  creation AND defensively every time the installer finds itself already
  inside a venv (covers a pre-existing broken `.venv/` left over from a
  prior failed run, exactly the state the user's own machine was in after
  hitting this bug once). `_bootstrap_pip()` tries two independent
  repairs before giving up: (1) `python -m ensurepip --upgrade
  --default-pip` run explicitly; (2) if that doesn't work -- e.g. the
  BASE Python's own `ensurepip` module is what's broken, so repair (1)
  can't fix it either -- downloads the official `get-pip.py` bootstrapper
  directly from `bootstrap.pypa.io` and runs it with the venv's
  interpreter. If both fail, it stops with a clear, actionable message
  (antivirus exclusion suggestion, manual `ensurepip` fallback command)
  instead of proceeding to the same confusing pip error.
- **Verified (this build sandbox, no Windows machine available, so
  reproduced the identical failure class on Linux and confirmed the
  repair path fixes it):** created a venv with `python3 -m venv
  --without-pip` (the same end state as the user's broken venv -- a venv
  that exists but has no pip), confirmed `_venv_has_pip()` correctly
  reports `False`, then confirmed `_bootstrap_pip()` repairs it via the
  `ensurepip` path (repair attempt 1) and `_venv_has_pip()` then reports
  `True`. Full `python install.py` (no `--check`) also verified end to
  end from a clean checkout in this sandbox: creates `.venv`, verifies
  pip, installs all of `requirements.txt` successfully (37 packages),
  and is idempotent -- running it again reuses the existing, working venv
  without re-downloading anything.
- **Second issue found while testing "run from anywhere," not user-
  reported but directly relevant to it:** `run.bat`/`run.sh`/
  `run_cli.bat`/`run_cli.sh` used bare relative paths
  (`.venv\Scripts\python.exe`), which only resolved correctly if the
  caller's CURRENT DIRECTORY already happened to be the project folder --
  e.g. a desktop shortcut with a different "Start in" folder, or running
  `bash /some/path/run.sh` from elsewhere, would silently fail to find
  `.venv`. Fixed by having each script `cd` into its OWN directory first
  (`%~dp0` on Windows, `dirname "${BASH_SOURCE[0]}"` on bash) before
  touching any relative path -- the same principle `main.py`/`cli.py`/
  `install.py` already use via `Path(__file__).resolve().parent`, just
  applied to the shell-script launchers too. Verified: copied the whole
  project to a differently-named folder in a different location
  (`totally_renamed_project_XYZ`, outside the original tree entirely),
  ran `main.py` from an unrelated `cwd` (`/`), and separately invoked
  `run.sh` from `cwd=/` -- both served the web UI and API correctly, and a
  full enroll -> verify -> feedback round-trip against the renamed instance
  produced identical results to the original. No file in the codebase
  hardcodes the project folder's name or relies on the process's current
  working directory -- confirmed by grepping the whole tree for
  `os.getcwd`/`Path.cwd()`/hardcoded `signxai/` path fragments (none
  found outside this project's own name as a display string, e.g.
  `APP_TITLE`).

### Bug #25 -- install.py's LLM wheel install used a filename-tag assumption that doesn't match how compiled Python wheels are actually named

- **How this was found:** not user-reported -- found by independently
  re-verifying the v4 "verified end-to-end" LLM install claim against
  fresh, current upstream documentation while responding to a request to
  validate the whole system for real Windows/macOS use. That re-check
  is what surfaced the problem, not a failure report.
- **Root cause:** `install.py` built a direct GitHub Releases download
  URL assuming a wheel filename like
  `llama_cpp_python-<version>-py3-none-<platform>.whl`. `py3-none` is a
  packaging tag that specifically means "pure Python, no compiled
  extensions" -- but `llama-cpp-python` wraps a compiled C++ library
  (llama.cpp) via CPython's C API, so its real wheels are necessarily
  tagged per-Python-ABI-version instead
  (`llama_cpp_python-<version>-cp310-cp310-<platform>.whl`,
  `cp311-cp311-...`, `cp312-cp312-...`, etc. -- confirmed directly by
  searching multiple independent current sources: the project's own
  PyPI page, ReadTheDocs, and third-party wheel mirrors, all showing
  this exact tag pattern). Separately, the upstream release/tag naming
  scheme has also changed to per-backend-suffixed tags (`v0.3.35-metal`,
  `v0.3.35-cu121`, `v0.3.35-vulkan`, ...) rather than one tag with a full
  asset matrix. Both facts mean the v4 hardcoded URLs were very likely
  404s that had never actually been exercised against a real qualifying
  Windows/macOS machine -- see the v4 status entry above for the
  "verified end-to-end" claim this supersedes.
- **Fix (`install.py`):** replaced the whole hand-built-URL approach with
  what the project's own current README/PyPI page/ReadTheDocs all
  independently document and recommend: `pip install llama-cpp-python
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<backend>`,
  a PEP 503 "simple" package index per backend (`cpu`, `cu118` through
  `cu132`, `metal`, `rocm72`, `hip-radeon`, `vulkan`). pip resolves the
  correct wheel for the CURRENT interpreter's actual Python version and
  platform automatically via this index -- the same mechanism PyPI
  itself uses -- so this project's own code no longer guesses a filename
  or a version tag at all. `_llm_backend()` picks `metal` for Apple
  Silicon, `cu121` when `nvidia-smi` detects a GPU (any OS), `cpu`
  otherwise; falls back to an unpinned from-source build
  (`pip install llama-cpp-python`, needs a C++ compiler + cmake) if the
  index install fails for any reason.
- **Also fixed for consistency:** `device.py`'s user-facing hardware note
  still asserted "CUDA wheels are Linux-only, Windows always uses CPU" --
  a claim from the same superseded investigation. Corrected to describe
  the actual current behavior (same backend selection attempted on any
  OS, with a documented graceful fallback) instead of asserting an
  unverified platform restriction. `requirements-llm.txt`'s hard version
  pin and matching claims were also updated to match.
- **Verified, with an honest limit on how far verification could go in
  this sandbox:**
  - Ran the real, complete `install.py --llm` end to end: correctly
    selected the `cpu` backend, correctly attempted the documented
    `--extra-index-url` install, and (since this specific development
    sandbox's network egress is restricted and does not allow
    `abetlen.github.io` -- confirmed directly, a `curl` to it returns
    `403 host_not_allowed` from this environment's proxy) correctly fell
    through to pip's normal behavior of using the sdist from PyPI's
    default index instead, then began a genuine from-source compile
    after installing `cmake` to allow it. Observed real object files
    (`ggml-base.so`, `ggml-cpu.c.o`, ...) successfully compiling via
    `gcc`/`g++` before the build was interrupted by this environment's
    own turn-based process lifecycle (background processes here do not
    persist across conversation turns) -- the compile itself was
    progressing correctly, not failing, when observation stopped.
  - **NOT verified**: an actual complete, successful prebuilt-wheel
    install via the `abetlen.github.io` index (blocked from this
    specific sandbox, not necessarily from a real user's machine, which
    will have normal unrestricted internet access), and a complete
    from-source build finishing successfully (interrupted by the
    environment, not by a code failure -- see above). **This is the
    single most important item for the user to verify**: run
    `python install.py --llm` on a real machine and confirm it either
    completes via the prebuilt-wheel index quickly, or completes a
    from-source build if no matching wheel exists and a compiler is
    present.

### Bug #26 -- crop bounding box was not robust to stray marks, causing position/framing to affect the verdict (user-reported: "if the image is a little above or below center, the system considers that the image is like that or the style")

- **How this was found:** user-reported, then reproduced directly and
  quantified before attempting any fix -- see docs/BUILD_LOG_v6.md for
  the exact scripts and numbers.
- **Root cause, confirmed with a synthetic reproduction:** the crop step
  (`preprocess.py::_crop_to_content`) computed the ink bounding box with a
  naive min/max over every ink pixel. A single small stray mark far from
  the actual signature -- a scan artifact, a staple hole, dirt, a bit of a
  ruled notebook line, or a trailing flourish tail that only got fully
  captured in one particular photo -- silently dragged that bounding box
  toward it. Because the normalization step fits this box onto a fixed
  canvas preserving aspect ratio, one outlier mark could distort the
  ENTIRE signature's measured aspect ratio and canvas position, even
  though the actual handwriting was identical. Quantified directly: a
  single small stray mark added to a real CEDAR signature shifted its
  measured aspect ratio from 2.28 to 1.77 (a ~22% relative shift) and its
  centroid position by a similarly large amount, using the pre-fix code.
  This is exactly what surfaces as "position/framing affects the verdict"
  -- several of the scalar features actually compared against a signer's
  profile (aspect ratio, ink centroid X/Y -- visible directly in a real
  generated report, e.g. "Aspect ratio (w/h)" hitting the z-score cap at
  +6.00) are sensitive to exactly this kind of crop-boundary noise.
- **Research grounding:** offline-signature-verification literature
  treats "normalization and centering" as a foundational, hard-required
  preprocessing step specifically because scale/position must not leak
  into what's supposed to be a shape-only comparison (search terms used:
  "offline signature verification preprocessing bounding box centering
  translation normalization robust", "test-time calibration signature
  verification translation invariance centroid feature" -- current papers
  and patent literature consistently list centroid/position as a feature
  ONLY in the ONLINE (pen-tablet dynamics) setting, where it's measured
  against a live coordinate system with pressure/timing context, not as a
  reliable static-image feature after ordinary cropping).
- **Fix, two complementary parts, both needed (each covers a different
  failure mode):**
  1. `_remove_distant_outliers()` (new): discards ink components that are
     BOTH small (relative to the largest stroke group and to total ink
     mass) AND spatially isolated from every substantial component. This
     catches a genuinely separate artifact (a corner speck, a staple
     hole) without ever touching real ink.
  2. The crop bounding box itself now uses the 0.5th-99.5th percentile of
     ink pixel coordinates per axis, not literal min/max -- catches a
     long thin flourish/tail that's PHYSICALLY CONNECTED to the real
     signature (so part 1's component-based filter can't separate it),
     since percentile bounds are inherently density-weighted: a thin
     sparse tail contributes few pixels relative to the dense signature
     body and naturally falls outside a 99% window, while genuine strokes
     (the overwhelming majority of ink almost everywhere) are untouched.
- **Two real design flaws caught and fixed BEFORE shipping, kept
  documented rather than silently corrected:** the first version of part
  1 anchored on the single LARGEST connected component and expanded a
  margin from its bounding box alone -- wrong, because a real signature is
  very often made of several separate connected components across its
  full width (individual letters, pen lifts, a detached first initial);
  on a real CEDAR test image this discarded a legitimate ~40%-of-total
  letter group. A second attempt tried adaptive-kernel dilation to bridge
  nearby strokes, keeping only the largest resulting blob -- also wrong
  for the same underlying reason (two legitimately separate letter groups
  in the same real signature can have a wider gap than a stray artifact
  does from the page edge). Both were caught by testing against real
  multi-component CEDAR signatures before shipping, not assumed correct
  from the design alone -- see docs/BUILD_LOG_v6.md for the exact
  before/after pixel counts.
- **Verified:** zero ink pixels lost on 5 real, artifact-free CEDAR
  signatures (identical ink-pixel sums to the original un-fixed code);
  the synthetic stray-mark test's aspect-ratio shift dropped from ~22% to
  under 1%; a full 10-writer/20-verification sweep re-run after the fix
  showed genuine-signature accuracy unchanged (5/10 correct, 0/10 falsely
  FORGED, same as before) and forged-signature false-accepts actually
  IMPROVED (3/10 -> 0/10 incorrectly called GENUINE) -- strong evidence
  that crop-boundary noise was also contributing to some of the forgery
  misclassifications, not just the position-sensitivity the user directly
  reported.

### Bug #27 -- PDF/report heatmap read as a flat color wash, and the annotated "rectangle detector" boxes were a rigid, colliding grid (user-reported, with a generated PDF as evidence)

- **User's report:** "there is not heatmap present, and those rectangle
  detector are to generic and choppy... don't get collide with other
  metric data" -- with a real generated PDF attached showing labels like
  "R1: Aspect Ru moment 1" running into "R2 Hu moment 1".
- **Root cause 1 (heatmap):** `explain.py::heatmap_overlay` blended a
  JET-colormapped pixel-difference image over the WHOLE canvas at a flat
  45% opacity, including the large blank white background that dominates
  most signature photos. JET's low end is a saturated dark blue, not
  white or transparent, so near-zero difference (almost the entire
  background) rendered as a solid blue wash across the image -- visually
  reading as "no heatmap" rather than a real one. The user's own pasted
  definition of a heatmap ("shows the magnitude of a phenomenon as
  color... giving obvious visual cues about how it's clustered or
  varies") requires near-zero regions to stay visually neutral so only
  genuine divergence pops out -- the previous implementation did the
  opposite.
- **Root cause 2 (rectangle detector):** `explain.py::annotated_regions`
  carved the canvas into a FIXED 4x8 grid and boxed whichever cells
  scored highest, with no relationship to the actual shape of any real
  divergence -- boxes routinely cut a stroke in half or landed on a
  mostly-blank cell adjacent to the real difference. Every label was
  placed at the same fixed offset from its own box with no collision
  checking against OTHER boxes' labels, so nearby/overlapping top-ranked
  cells (common with a rigid grid, since several adjacent cells often
  score similarly) produced visibly overlapping text -- exactly what the
  attached PDF showed.
- **Fix 1 (heatmap):** contrast-stretch the blurred difference map to its
  own observed range before colormapping (so a typically-small raw pixel
  difference actually uses the colormap's full range instead of only ever
  reaching its low end), and alpha-modulate the blend by divergence
  magnitude (gamma-shaped) instead of a flat blend ratio -- near-zero
  pixels stay the clean original image; only genuinely divergent regions
  are colorized, increasingly opaquely as divergence increases. Switched
  from JET to TURBO (a more perceptually uniform colormap).
- **Fix 2 (rectangle detector):** replaced the rigid grid with contour
  detection on the thresholded, morphologically-closed divergence map --
  boxes now correspond to real divergence blobs, content-adaptive rather
  than grid-locked. Added greedy non-max suppression (IoU >= 0.2 discards
  the lower-energy box) so a single blob that survives as 2-3
  overlapping contours is only reported once. Label drawing is delegated
  entirely to the existing `annotate_regions` collision-avoidance logic
  (Bug #11, this file) instead of a second, weaker implementation
  duplicated in `annotated_regions` -- one label-placement
  implementation, reused. Also clamped label X position to the canvas
  (previously only Y was clamped), and shortened the truncation length so
  a label near the right edge can't run off-canvas.
- **Verified:** regenerated a real case's heatmap and annotated-regions
  images after the fix and visually confirmed (see docs/BUILD_LOG_v6.md)
  -- heatmap shows a clean white background with color only along the
  actual ink strokes, intensity varying with real divergence; the
  annotated image shows 2-3 genuinely distinct, non-overlapping regions
  with fully on-canvas, non-colliding labels, a direct visual improvement
  over the grid version.

### Bug #28 -- LLM install on Windows: pip silently chose a source-only release over the CUDA wheel, then failed with "nmake ... no such file" (user-reported, with a real Windows terminal log)

- **User's report:** a real, complete Windows install log showing (1)
  `pip install llama-cpp-python --extra-index-url .../cu121` resolving to
  `Using cached llama_cpp_python-0.3.35.tar.gz` (a SOURCE distribution,
  not a wheel) despite the cu121 index being specified, then (2) the
  resulting from-source build failing with
  `CMake Error ... 'nmake' ... no such file or directory`.
- **Root cause 1:** pip's multi-index version resolution does not
  strictly prefer an OLDER version that has a published wheel over a
  NEWER version that only has a source distribution. If the abetlen
  wheel-index's build cadence lags the newest PyPI release (very
  plausible -- prebuilt wheels for every backend x every Python version
  take real build-farm time to publish after a source release), pip will
  happily grab the newest sdist instead of an older, wheel-available
  version -- exactly what the log shows. The v5 fix (Bug #25) used a
  plain `pip install ... --extra-index-url ...`, which has no defense
  against this.
- **Root cause 2:** NMake and the MSVC compiler (`cl.exe`) are only on
  PATH inside a "Developer Command Prompt for VS" that has run
  `vcvars64.bat` -- a plain PowerShell/cmd window (what `python
  install.py` actually runs in) never has them, EVEN WITH Build Tools
  correctly installed. The previous guidance ("install Build Tools for
  Visual Studio") was necessary but not sufficient -- it doesn't fix the
  environment problem that actually caused this specific error.
- **Fix 1:** `_wheel_only_pip_install()` -- adds `--only-binary=:all:` to
  the primary install attempt, which makes pip FAIL IMMEDIATELY instead
  of silently substituting a source distribution when no matching wheel
  exists. Verified directly (see docs/BUILD_LOG_v6.md): in this sandbox
  (which also has no reachable wheel for this exact combination), the
  wheel-only attempt now fails in seconds with a clear
  "No matching distribution found" instead of a 10-minute detour into a
  doomed compile. `install_llm_backend()` now also retries with the
  'cpu' backend (by far the widest published wheel coverage across
  Python versions) before ever attempting a compile, so a temporarily
  lagging GPU-backend wheel no longer forces a compiler requirement on
  a user who doesn't need one.
- **Fix 2:** `_find_vswhere()` / `_find_vcvars64()` / `_msvc_build_env()`
  -- locate a real Visual Studio C++ Build Tools installation via
  `vswhere.exe` (Visual Studio's own official, version-independent
  discovery tool, bundled since VS2017) and run `vcvars64.bat` in a child
  `cmd.exe`, capturing the resulting environment (PATH, INCLUDE, LIB,
  ...) so THIS process's own pip install can use it transparently --
  fixing the actual gap, not just describing it. If no such installation
  can be found at all, `_install_llm_from_source()` now fails
  IMMEDIATELY with a clear, actionable message instead of attempting (and
  failing 10+ minutes into) a doomed CMake configure.
- **Verified, with an honest limit given this sandbox has no Windows:**
  the wheel-only fail-fast behavior was verified for real (see above);
  the MSVC auto-detection logic was verified to gracefully return "not
  found" and produce the correct FATAL message on a machine with no VS
  install (this sandbox), by temporarily monkeypatching `platform.system`
  to report "Windows" -- confirms the code path executes correctly
  end-to-end without crashing, but the actual `vswhere`/`vcvars64.bat`
  auto-loading has not been exercised against a real Windows Build Tools
  installation. **Top item for the user to verify.**

### Bug #29 -- web UI: no way to remove an added enrollment sample, feedback had no remark/submit flow, and the visual design needed a full restyle (user-reported, with screenshots and a provided style reference)

- **User's report:** a screenshot showing the enrollment thumbnail row
  with no remove control at all ("there is cross arrow is missing, when
  we needs to remove some images"), a request for an explicit "submit
  response button that confirm the response which have to provide on the
  Remark", a request to visually separate a "Local AI Explanation Model"
  section and a "High Level visualization" section from the engine's own
  analysis, and a provided style-reference document ("Browserbase").
- **Root cause (thumbnails):** `addThumb()` appended a bare `<img>` tag
  with no accompanying control of any kind -- there was genuinely no way
  to remove a sample once added short of clearing the whole form.
- **Fix (thumbnails):** `renderThumbs()` now renders every pending sample
  from the `pendingSamples` array with an always-visible remove (x)
  button (`.thumb-remove`), and removal (`removeThumb(i)`) always
  re-renders from the array so the displayed thumbnails and what's
  actually submitted can never drift out of sync.
- **Fix (feedback flow):** feedback is no longer instant-fire on
  Confirm/Mark-wrong. The examiner now explicitly selects Correct/Wrong
  (`selectFeedback`), can add a free-text remark, and must click a
  separate "Submit feedback" button (`submitFeedback`) before anything is
  sent. The remark is now a real, persisted field
  (`FeedbackRequest.remark` -> `engine.feedback(..., remark=...)` ->
  stored in `profile["feedback"]`) -- verified end-to-end via a live API
  round-trip (create profile, enroll, verify, submit feedback with a
  remark, re-fetch the profile, confirm the exact remark text is stored).
- **Fix (section restructuring):** the Verify results view is now three
  explicitly labeled sections instead of one long undifferentiated
  scroll: "Engine analysis" (per-reference scores, metrics table, SHAP --
  what the classical engine itself found and decided on), "High-level
  visualization" (heatmap, overlay, skeleton, agreement chart grouped
  together), and "Local AI explanation model" (the narrative prose only,
  visually distinguished with a sky-tint card background so it's never
  confused with the engine's own decision evidence above it).
- **Fix (visual restyle):** applied the user-provided Browserbase design
  system wholesale -- near-monochrome canvas (black/white/off-white),
  Space Grotesk display headlines (free substitute for the
  license-restricted GT Planar, matching the guide's own suggested
  fallback logic), Inter body text, JetBrains Mono metadata labels
  (matching the guide's own stated substitutes), 50px pill buttons filled
  black for every primary action, hairline borders instead of shadows for
  card elevation, and lavender-mist/faint-slate/sky-tint pastel surfaces
  for grouping. The signal-orange accent is deliberately NEVER used as a
  button color, matching the guide's explicit rule -- the one exception
  is verdict banners (genuine/forged/inconclusive), which keep their own
  semantic colors since that color is load-bearing information, not
  decoration.
- **Verified:** every CSS class and element ID referenced by `app.js`/
  `index.html` cross-checked programmatically against what's actually
  defined (`grep`-based cross-reference, not just visual inspection);
  `node --check` confirms `app.js` is syntactically valid; the remark
  storage round-trip was verified against a live server. **Not
  independently screenshotted in a real browser** -- this development
  sandbox has no reachable headless-browser download source (confirmed:
  `playwright install` fails with the download host not in this
  environment's network allowlist), so the actual rendered appearance
  could not be visually confirmed here. **Top item for the user to
  verify.**

### Bug #30 -- LLM install could report success on Windows yet still fail to load at runtime (real DLL dependency, confirmed with a real Windows log + screenshot)

- **User's report:** a real Windows terminal log showing `pip install`
  successfully downloading and installing a genuine 448MB `cu121` CUDA
  wheel end to end, followed by a screenshot of the app's own AI Model
  status page showing `Status: error` with:
  ```
  Failed to load shared library 'D:\signxai\.venv\Lib\site-packages\llama_cpp\lib\llama.dll':
  Could not find module '...\llama.dll' (or one of its dependencies).
  ```
  This is genuinely good news buried in a new bug report: it directly
  confirms Bug #28's wheel-only, multi-backend install fix (previous
  entry) works correctly on real Windows hardware -- the wheel-only
  install picked the right backend, downloaded a real matching wheel, and
  installed it successfully. This is a DIFFERENT, later failure: a
  successfully-installed package whose native library still can't load.
- **Root cause:** `pip install` succeeding only means the wheel's FILES
  were unpacked onto disk correctly -- it says nothing about whether the
  native shared library inside a compiled-extension package can actually
  be LOADED by the OS. Windows's own `LoadLibrary` error message ("...or
  one of its dependencies") is the standard wording for exactly this: the
  named DLL exists, but something IT depends on doesn't. For a CUDA-
  enabled build, that's almost always either (a) CUDA runtime components
  (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, ...) not
  present or not a compatible version for this machine's installed NVIDIA
  driver, or (b) the Microsoft Visual C++ Redistributable (x64), which
  many native Python extensions depend on but isn't always present on a
  fresh Windows install. Neither of these is something pip checks or can
  check -- they're OS-level runtime dependencies invisible to the package
  manager.
- **Fix (`install.py`):** added `_verify_llama_cpp_importable()` -- runs
  `python -c "import llama_cpp"` in a fresh subprocess immediately after
  every successful wheel install, actually exercising the native-library
  load rather than trusting pip's exit code. If that import fails,
  `_uninstall_llama_cpp()` removes the broken package (never leave an
  unusable backend silently installed) and `install_llm_backend()`
  continues to the next, more broadly-compatible candidate (the 'cpu'
  backend, which has no CUDA/driver dependency at all) before ever
  requiring a from-source compile. A Windows-specific, CUDA-specific
  actionable message is also printed pointing at the Microsoft Visual
  C++ Redistributable download when this exact failure pattern is
  detected.
- **Fix (`app/llm/manager.py`):** added a matching runtime catch for
  users who reach this state some other way (manually copying a venv, an
  older/partial install, installing by hand) -- `_load()` now catches
  `OSError`/`ImportError` specifically and, when the error text matches
  this known pattern (`"llama.dll"` or `"dependencies"` in the message),
  appends the same actionable guidance (re-run `python install.py --llm`
  to get the new automatic fallback, or install the VC++ Redistributable)
  to `self._error` instead of surfacing a bare, unexplained traceback
  string in the UI.
- **Verified:** `_verify_llama_cpp_importable()` correctly detects a
  missing/broken `llama_cpp` package in this sandbox (real test: reports
  `ok=False` with the actual `ModuleNotFoundError` text); the
  `manager.py` hint-matching logic confirmed to trigger correctly against
  the EXACT error string from the user's own screenshot. **Not verified
  end-to-end against the user's actual failure** -- this sandbox has no
  real GPU/CUDA runtime to reproduce "wheel installs, DLL won't load"
  authentically; the fix's logic is confirmed correct and exercised as
  far as this environment allows, but the user re-running
  `python install.py --llm` on the same real machine is the only way to
  confirm the automatic CPU-backend fallback actually resolves it there.

### Bug #31 -- global-only similarity metrics can dilute a small but structurally meaningful local difference (user-reported: a real forgery's missing zigzag stroke wasn't reflected in the explanation)

- **User's report:** a genuine signature has a stroke with three sharp
  zigzag direction-reversals; a questioned (forged) signature reproduces
  only two. The system's overall verdict was correct, but its reasoning
  didn't specifically identify or explain the missing stroke pattern --
  the user asked for the recognition logic itself to explicitly examine
  the signature region-by-region (explicitly requesting "8 small
  sections," with sections combining their nearest neighbors too), and
  asked that this be designed with real care for different scripts
  (Japanese, Korean, Chinese, Hindi, English), not just Latin cursive.
- **Why this is a real, well-documented failure mode, not just this
  project's bug:** every similarity method already in `similarity.py`
  (SSIM, ink-IoU, HOG-cosine, LBP-cosine, Hu-cosine, DTW, scalar
  z-scores) is a GLOBAL average over the whole signature. A forger who
  reproduces most of a signature well but drops a small structural detail
  in one region can still score well on a global average, because
  everything else that DOES match dilutes the one thing that doesn't.
  Research (search terms used, current: "offline signature verification
  zoning grid local features rectangular zoning ring zoning" and
  "signature verification stroke direction changes corner count local
  descriptor cross-script CJK Devanagari") confirms this is exactly why
  "zoning" -- partitioning a signature into regions and comparing LOCAL
  agreement in addition to global agreement -- is an established,
  published technique, not a novel idea: Cartesian and polar zoning with
  per-zone HOG/LBP appear in multiple independent papers; a US patent on
  "high pressure regions" explicitly requires BOTH the global score AND
  ALL local (vertical- and horizontal-partition) checks to pass before a
  signature is called genuine; Chinese offline-signature research
  explicitly uses per-region "direction distribution" as a feature
  (supporting a direction/corner-based local descriptor, not just
  intensity); and multi-script forensic literature confirms different
  scripts have fundamentally different stroke density/directionality
  structure, which is exactly why this needed to NOT be a naive fixed
  pixel grid.
- **Design (new module, `app/core/zoning.py` -- full reasoning in its own
  module docstring, which should be read in full before modifying it):**
  1. **Content-adaptive zoning, not a fixed pixel grid.** Zone boundaries
     are computed from the reference's own ink-mass cumulative
     distribution (quantile splits along each axis), so every zone
     contains roughly the same AMOUNT OF SIGNATURE CONTENT regardless of
     script or writing direction -- the grid follows the ink, not the
     other way around. Default 4 columns x 2 rows = 8 zones, matching the
     granularity the user asked for directly.
  2. **Merged-neighbor zones**, directly implementing the user's explicit
     request ("its common section combining nearest block to check
     pattern from there also"): every horizontally- and vertically-
     adjacent zone pair is ALSO compared as one merged region, so a
     stroke or zigzag straddling a base zone boundary is still captured
     whole in at least one merged view even when split across two base
     zones.
  3. **A corner/turning-point descriptor specifically targeting the
     user's exact failure mode.** A zigzag IS a sequence of sharp
     direction reversals; the module detects these via a structure-
     tensor-style local-curvature test on the signature's own skeleton
     (for each skeleton point, do nearby skeleton points mostly lie along
     one direction, or spread across two -- i.e. a bend?), then bins
     detected corner points into whichever zone they fall inside. Local
     HOG-cosine similarity and local ink-density agreement (both already-
     used descriptor families elsewhere in this codebase) are computed
     per zone too, and all three are fused into a per-zone score.
  4. **The aggregate zonal score is a genuine new fusion component**
     (`components["zonal"]`, weight 0.12, `DEFAULT_WEIGHTS` rebalanced
     accordingly) -- individual per-zone FLAGS are kept as explanatory
     findings for human review (case data + a new `zone_grid.png`
     visualization + report findings), not as an automatic hard verdict
     override, because testing showed individual zone flags are noisier
     than the well-established global metrics at real handwriting's
     natural fine-scale variation (see the dev notes below) -- but the
     ink-mass-weighted AGGREGATE across all zones is a real, tested,
     stable signal.
- **Three real design flaws caught by testing before shipping, kept
  documented rather than silently corrected (this dev history matters --
  see docs/BUILD_LOG_v6.md for the exact numbers behind each):**
  1. First corner descriptor: polygon-approximation vertex count
     (`cv2.approxPolyDP`) on each zone's FILLED ink contour. Failed on
     synthetic genuine-vs-genuine tests -- a thick stroke's own jagged,
     anti-aliased boundary traces every real direction change TWICE (once
     per side) and adds enough boundary noise that two samples of the
     IDENTICAL signature triggered false structural-mismatch flags.
  2. Second attempt: trace an ORDERED path along the 1px skeleton between
     fixed-arc-length neighbors, measuring the path's own tangent-
     direction change. Failed badly on REAL CEDAR signatures (not just
     clean synthetic ones): real cursive handwriting is full of self-
     intersections and branch points, and a simple greedy walk stops at
     the first branch it reaches -- one real test image's walk covered
     only 41 of 446 pixels (9%) in a single connected skeleton component,
     silently missing most of the actual stroke and producing a false
     "zero corners here" reading that flagged 58 of 60 zones across a
     10-writer genuine-vs-genuine sweep.
  3. Final corner descriptor (shipped): a structure-tensor-style
     eigenvalue test needing NO path ordering at all -- naturally handles
     branches (a junction simply reads as legitimate local curvature).
     Re-verified against the same 10-writer sweep: an initial pure-
     relative-threshold flagging rule still produced 58/60 false
     positives (a 0-vs-1 or 2-vs-3 corner mismatch registers as
     "50-100% disagreement" even though it's normal handwriting
     variation) -- fixed by ALSO requiring a real absolute-count floor
     (`corner_mismatch >= 5`) alongside the relative threshold, which
     brought the same sweep to 0 false positives while still flagging the
     forgery comparisons in the same sweep, which show much larger
     absolute gaps (see docs/BUILD_LOG_v6.md for the full before/after
     table).
- **Verified:**
  - Synthetic reproduction of the user's EXACT scenario (a stroke with
    a controllable zigzag count): genuine-vs-genuine (3 vs 3 zigzags)
    scored 0.987 aggregate with 0 flagged zones; forged (2 vs 3 zigzags)
    scored 0.890 aggregate with 2 correctly-located flagged zones,
    reporting the real corner-count gap directly (e.g. "query=7,
    ref=13").
  - A harder, more realistic synthetic test (the same small zigzag detail
    tucked into just ~8% of a much longer, denser signature, closer to
    how a subtle real forgery detail could be diluted): the GLOBAL HOG-
    cosine metric alone could not reliably separate the two cases (and in
    one configuration even scored the forgery HIGHER than the genuine
    sample) -- the zonal module correctly isolated and flagged the
    specific region once its own within-zone dilution bug (design flaw
    caught by this same test) was fixed.
  - Real CEDAR data, 10-writer sweep: the AGGREGATE zonal score correctly
    ordered genuine-above-forged in 8/10 writers (mean genuine aggregate
    0.828 vs mean forged aggregate 0.774) -- a real, additional
    discriminating signal, honestly reported as imperfect (2/10 writers
    did not separate cleanly) rather than overstated.
  - Full engine integration verified end-to-end (enroll -> verify ->
    case data -> `zone_grid.png` asset -> PDF/HTML report inclusion ->
    CLI zonal-findings table) against real CEDAR images; the 10-writer
    genuine/forged sweep re-run after integrating zonal into the fused
    score showed genuine_correct dropping slightly (5/10 -> 4/10, with
    genuine_wrong still 0/10 and forged_wrong still 0/10) -- a real,
    honestly-reported trade-off from adding a genuinely new, still-
    maturing signal into the fusion, not hidden by re-tuning the weight
    to make the number look better (a 0.06-weight variant was tested and
    produced the identical result, since the whole system self-
    calibrates together via `_relearn`'s threshold fitting).

### Bug #32 -- "LLM: ready" shown in the UI, but generation ran on CPU (spike in CPU/RAM, not GPU) despite real GPU hardware

- **User's report:** "even I got sign of LLM: ready... I get a sudden
  spike in CPU & RAM, when model starts to load. It means model is not
  loading in gpu."
- **Root cause, confirmed against real, independent reports of the exact
  same behavior:** `n_gpu_layers=-1` (this codebase's own setting,
  meaning "offload every layer to GPU") is SILENTLY IGNORED by
  llama-cpp-python if the installed package wasn't actually compiled
  with CUDA/Metal support -- there is no Python-level exception, no
  error, no non-zero return code. It just runs on CPU. This is a
  well-documented, common gotcha independent of this project. Given
  Bug #30's automatic CPU-backend fallback (this file, above) exists
  precisely because a GPU wheel can fail to load, a real, plausible path
  to this exact symptom is: the installer silently fell back to a
  CPU-only backend (correctly, per Bug #30's design) -- but the STATUS
  reporting never distinguished "GPU hardware exists" from "the loaded
  backend can actually use it," so the UI kept showing `gpu_offload_active:
  true` (and therefore "LLM: ready" reading as fully successful) based
  purely on hardware presence, never on what actually happened at load
  time.
- **Fix (`app/llm/manager.py`):** `_capture_native_output()` redirects the
  process's REAL stdout/stderr file descriptors (not just Python's
  `sys.stdout`/`sys.stderr` -- llama.cpp's C++ core writes its own
  load-time diagnostics directly to the native file descriptors, which a
  plain `contextlib.redirect_stdout` does not see at all) around the
  `Llama(...)` construction call (now run with `verbose=True` internally,
  captured and never shown to the user), and `_parse_gpu_offload()`
  parses that captured log for a real, definitive "offloaded N/M layers
  to GPU" confirmation. `status()`'s `gpu_offload_active` field is now
  keyed to this CONFIRMED result, not hardware presence -- with a
  distinct `gpu_offload_unconfirmed` field for the rare case the log
  doesn't match a recognized pattern (honestly reported as unknown,
  never assumed true). The web UI and CLI both now show a clear,
  actionable warning when GPU hardware is detected but 0 layers were
  actually offloaded, pointing at Bug #30/#32 and suggesting a re-install
  to retry getting a real GPU-capable backend.
- **Verified:** the log-parsing regex correctly identifies both a real
  CUDA success pattern (`"offloaded 32/32 layers to GPU"` → confirmed
  True) and a CPU-only fallback pattern (`"offloaded 0/29 layers to
  GPU"` → confirmed False) in isolation; the fd-level capture mechanism
  confirmed to actually intercept native-level output that a
  Python-level redirect would miss (direct test: wrote via `os.write(1,
  ...)`, confirmed captured). **Not verified against the user's own real
  GPU** -- this sandbox has no GPU at all, so the full "load a real model
  on real CUDA hardware and confirm the log line matches" path could not
  be exercised end-to-end. If the user re-installs (per Bug #30's fix)
  and the backend genuinely has GPU support this time, the AI Model
  status page will now honestly show whether offload is actually
  happening -- **this is the way to find out, and the top item to verify
  next.**
- **On the broader request to guarantee a CPU fallback is always
  available:** already true by construction from Bug #28/#30 --
  `install_llm_backend()`'s candidate list always includes `'cpu'`
  before ever requiring a from-source compile, specifically so a device
  that can't use GPU at all still ends up with a fully working (if
  slower) AI layer. This fix doesn't change that guarantee; it only
  makes the app HONEST about which case actually happened, so a user
  with real GPU hardware (like this one) can tell the difference between
  "working as intended on this device" and "silently downgraded, worth
  investigating."

### Bug #33 -- a signature with massively divergent pen-lift/loop-construction counts (z-scores at the hard cap on 4 independent features) still only scored INCONCLUSIVE, not FORGED

- **User's report, with a real generated PDF report as evidence:** "the
  System is not able to detected forg[ed] signature, even it have ton of
  signal that it['s] not [g]en[u]ine." The attached report showed
  `stroke_junctions` at z=+6.00 (the hard cap), `connected_components` at
  z=-6.00, `outer_contours` at z=-6.00, and `enclosed_loops` at z=+6.00 --
  four independent features simultaneously as extreme as this system can
  measure -- yet the verdict was INCONCLUSIVE at 36.5% confidence.
- **Root cause, confirmed mathematically and reproduced directly against
  real user-provided signature photos (not just the PDF's numbers):**
  `scalar_profile_sim` computes `mean(exp(-|z|))` as an UNWEIGHTED AVERAGE
  across all ~22 scalar features. Even with 4 features maxed out at
  `exp(-6) ≈ 0.0025`, averaging with ~18 other features that look
  reasonably normal (continuous geometric/shape measurements a forger
  copying visible shape can get roughly right) pulls the combined number
  back up toward ~0.6-0.8 -- confirmed directly: with the 4 real user-
  provided signature photos (3 genuine samples of one person's actual
  signature, 1 visibly different forgery), the four count-based topology
  features (`endpoints_count`, `junctions_count`, `components_count`,
  `holes_count`, plus `contours_count`) are forensically distinctive --
  literal counts of how many times the pen left the paper, how many
  separate ink shapes exist, how many loops closed -- exactly the kind of
  thing a forger who traces or copies visible SHAPE well still gets wrong,
  because they cannot see or replicate the original writer's actual pen
  movements (this maps directly onto what questioned-document examination
  calls "line quality" and construction sequence). Averaging these
  together with ~17 easier-to-superficially-mimic continuous features
  structurally dilutes them -- the SAME dilution mechanism already fixed
  ONCE for LOCAL/regional comparison in Bug #31 (`zonal`), now confirmed
  to exist at the GLOBAL/whole-signature level too.
- **Fix (`similarity.py`, `engine.py`):** `CONSTRUCTION_KEYS` identifies
  the 5 discrete-count topology/shape features; `construction_consistency_sim()`
  computes the same z-score+cap+exp(-|z|) approach restricted to ONLY
  these 5 (not diluted by the other ~17), and `scalar_profile_sim()`
  gained an `exclude_keys` parameter so the two don't double-count the
  same features. `construction` is now its own top-level fusion
  component (mirroring exactly how `zonal` was split out in Bug #31),
  with a genuinely meaningful weight.
- **A second, related fix found while testing against real photographed
  signatures specifically (`features.py`):** re-verifying a signature
  against its OWN enrolled profile (as close to "certainly genuine" as
  this system can produce) still showed real construction-count
  volatility. Traced to a real, distinct mechanism: real-world PHONE-
  PHOTOGRAPHED signatures (unlike consistent flatbed scans) can have a
  stroke that's fractionally thinner in one shot than another purely
  from lighting/exposure -- enough to drop a couple of pixels below the
  Otsu threshold and open a tiny gap in what is, physically, one
  continuous stroke or one fully-closed loop. Because these are DISCRETE
  COUNTS, a single such gap doesn't nudge a measurement, it FLIPS a whole
  integer count (one component becomes two; one enclosed loop becomes
  zero). Fixed by computing these specific features from a
  morphologically-CLOSED copy of the ink mask (a small 3x3 closing,
  bridging gaps at lighting-noise scale while leaving a real structural
  difference -- an actually missing loop, an actually different pen-lift
  count -- completely unaffected, since those are far larger than a 3x3
  kernel can bridge).
- **Weight rebalancing, evidence-based, not guessed:** re-testing against
  the real user-provided images after the fixes above showed `ssim`/
  `hog`/`lbp` barely distinguishing the genuine samples from the real
  forgery (ssim 0.90 vs 0.81, hog 1.00 vs 0.84, lbp 1.00 vs 1.00 --
  essentially non-discriminating for THIS forger, who evidently copied
  visible shape/texture reasonably well) while `iou`, `dtw`, and the new
  `construction` component showed large, real gaps (iou 0.90 vs 0.16,
  dtw ~0.9-1.0 vs 0.50, construction ~0.34-0.67 vs 0.13). `_HAND_GUESSED_WEIGHTS`
  shifted toward the features that demonstrably carried signal for this
  real case and down from the ones that didn't, still summing to 1.0.
- **Verified, both gains and honest costs, not just the good news:**
  - Real user images: 4.jpeg (the visually-obvious forgery) moved from
    INCONCLUSIVE (score 0.543, before any of this round's fixes) to a
    correctly and clearly FORGED verdict (score 0.458 vs threshold
    0.548) after preprocessing + weighting fixes together. 1.jpeg and
    2.jpeg (genuine) stayed correctly GENUINE throughout.
  - Full 10-writer CEDAR sweep, before vs after this round's changes:
    ```
                        before      after
    genuine_correct:     4/10        4/10
    genuine_wrong:       0/10        0/10
    forged_correct:      0/10        2/10   <- real, first-ever gain
    forged_wrong:        0/10        1/10   <- real, honest new cost
    ```
    Investigated the new false-accept directly (writer 5): `iou=0.14`
    and `hu=0.34` correctly signaled a real mismatch, but `construction=0.80`
    stayed high -- this specific skilled forgery evidently DID replicate
    pen-lift/loop counts closely while failing on shape/pixel-level
    fidelity, the mirror image of the 4.jpeg case (which failed on
    construction while passing on shape/pixel fidelity). All 5 individual
    references unanimously agreed GENUINE for this case, so the edge-case
    engine had no internal disagreement to catch either. This is not a
    bug to chase further within this fix -- it's the honestly-documented
    reality that no single classical feature (or fusion of them) catches
    every skilled forgery; different forgers fail differently, and this
    round's fix demonstrably catches a real failure MODE it didn't catch
    before, at a real, disclosed cost on a different failure mode.

### v3 decision — dataset-based calibration (`training/calibrate_weights.py`) de-prioritized
- **User's stated reason:** cannot fully trust downloading and running
  third-party open-source datasets as a precaution against the device being
  compromised — a legitimate security posture, not a technical objection to
  the approach itself.
- **Decision:** the training pipeline from v2 stays in the codebase (it's
  useful, real, working code, and removing it would be destructive for no
  technical reason) but is now explicitly NOT the recommended or default
  path. `data/population_weights.json` from the v2 validation run has been
  removed from what's shipped by default — `engine.py`'s
  `_load_default_weights()` already gracefully falls back to the
  hand-guessed weights when that file doesn't exist, so this required no
  code change, only a packaging decision.
- **Where effort went instead:** directly improving the zero-external-data,
  few-shot calibration algorithm itself (Bugs #13, #14) — which turned out
  to be the actual root cause of the user's reported problem anyway, not a
  case where more training data would have helped. This validates the
  user's instinct: the bug was in the *statistics*, not the *data*.
- **If the user later decides to use it:** `docs/TRAINING.md` still fully
  documents the pipeline and its "isn't a black-box neural net" reasoning,
  for if/when trust in a specific dataset source is established.

### v3 additions
- **Multi-GPU / dedicated-GPU-over-integrated handling** (`device.py`):
  `nvidia-smi` only ever lists NVIDIA GPUs (an Intel/AMD integrated GPU can
  never be falsely detected — this was already correct), and now, if a
  machine has multiple NVIDIA GPUs, the one with the most VRAM is
  selected, with the full list exposed in device diagnostics for
  transparency. Addresses the user's dual-GPU laptop (Intel iGPU + NVIDIA
  RTX 4050) directly.
- **GPU offload visibility** (`manager.py::status()`): added explicit
  `gpu_detected`, `gpu_name`, `gpu_offload_active` fields so a user can
  verify from the UI/API whether the LLM is actually running on GPU or
  CPU, instead of guessing ("right now it is working on CPU I think").
  `gpu_offload_active` is only true once a model is actually loaded, not
  just detected — avoids a misleading "yes" before anything is running.
- **Report depth** (`llm/prompts.py`, `llm/report.py`): extended the report
  schema with `methodology`, `confidence_explanation`, and
  `recommendation` sections — real substantive content generated both by
  the template path and requested from the LLM path via an expanded
  prompt contract. Verified: resulting PDF is 3 pages, matching the user's
  ask.
- **Windows launch friction fixed**: `run.bat`/`run.sh`/`run_cli.bat`/
  `run_cli.sh` convenience launchers added so users don't need to remember
  which `.venv` path (`Scripts\python.exe` vs `bin/python`) applies to
  their OS — this was the direct cause of a `CommandNotFoundException` the
  user hit copy-pasting the wrong doc line.
- **UI placeholder renamed** from "Vinit Sharma" to "ryotsuuu" per request.

---

## 3b. v7 bugs (chronological, append-only) — real false-FORGED root cause + LLM hang

> **Context for whoever reads this next**: v6 shipped believing Bugs
> #26-#33 had resolved the recognition-robustness reports. They hadn't —
> or rather, they'd fixed real, separate bugs, but the user's own repro
> images (5 real photographed signatures, reused throughout this and
> prior sessions: images 1/2 = enrolled genuine references, image 3 =
> held-out genuine, image 4 = a real forgery, image 7 = photographed on
> ruled/lined paper) still reproduced a confidently-wrong FORGED verdict
> on image 3 in v6. This section is the result of actually running the
> v6 engine against those exact images in a sandbox (not guessing) and
> and tracing every contributing factor to a root cause with printed
> numbers, not assumption. **Read this whole section before touching
> `similarity.py`, `engine.py`'s `_score_against_ref`/`_relearn`, or
> `align.py` again** — several of these bugs look independently
> plausible in isolation but compound, and an earlier draft of one fix
> (see Bug #36) itself introduced a regression that had to be caught by
> re-running the SAME repro images after every change, not just once.

### Bug #34 — `scalar_profile`/`construction` fusion components compared the query to the PROFILE MEAN, not to the specific reference — and with 2 refs, that mean-based z-score is a mathematical artifact, not a real signal

**Reported as:** user's own genuine image 3 scored FORGED (0.5165 vs.
threshold 0.5887) against a profile enrolled with 2 genuine references
(images 1 and 2).

**Root cause, found by direct reproduction, not inspection alone:**
`_score_against_ref` computed `scalar_profile_sim(query, profile_mean,
profile_std)` and `construction_consistency_sim(query, profile_mean,
profile_std)` — note neither takes the REFERENCE being compared against
as an argument. This has two consequences, both confirmed by printing
real numbers from the user's own profile:

1. These two components (32% of the total fusion weight) were IDENTICAL
   across every reference in the per-reference comparison loop — the
   entire point of comparing against every enrolled reference (v5, Bug
   #21) was defeated for a third of the score.
2. With exactly 2 references, `np.std` of any feature across those 2
   points is mathematically GUARANTEED to equal exactly half their
   absolute difference. Printed proof, from the user's own two enrolled
   images: querying with image 1 (one of the two enrolled images itself)
   gave z-scores of *exactly* ±1.000 (to 3 decimals) on ~10 of 22
   scalar features — `stroke_width_cv: z=-1.000`, `components_count:
   z=-0.962`, `hu_1/hu_2/hu_3/hu_6: z=1.000` etc. This is not a
   coincidence or a tuning issue: it is a guaranteed property of
   `std([a, b])` when the query IS one of `{a, b}`. `exp(-1) ≈ 0.368`,
   so even a PERFECT match to one specific enrolled reference was
   capped at ~37% similarity on these two components purely from this
   artifact — enough by itself to sink a genuine score below threshold.

**Fix:** new `similarity.pairwise_scalar_sim(query, reference, std,
...)` — z = (query − *this specific reference*) / natural-variation-std,
not (query − population mean) / population-std. `std` (the profile's own
per-feature natural-variation estimate) is still used, correctly, as the
TOLERANCE scale ("is this difference large relative to how much this
signer's samples normally vary"), not as the thing being measured
against. An exact match to a specific reference now gives z=0 exactly,
by construction, regardless of how many references are enrolled — the
artifact above cannot occur anymore, not just "occurs less often."
`engine.py::_score_against_ref` now calls this per-reference for both
`scalar_profile` and `construction`. The OLD population-mean-based
`scalar_profile_sim` is kept, unchanged, for the human-readable metrics
table (comparing a query to "this profile's typical signature" IS the
right question for a human reviewer — it just isn't the right question
for the automated per-reference fusion score).

**Verified:** re-running the exact 5-image repro after this fix alone
(before Bugs #35-#40 below) raised image 3's fused score from 0.5165 to
0.5835 (FORGED → right at the INCONCLUSIVE/threshold boundary) and
image 1/2 self-checks from 0.81 to 0.92 fused score, with image 4 (real
forgery) unaffected (still correctly FORGED). Full stack of fixes below
gets image 3 further, above the raw threshold (0.5835 -> 0.6119 in
final testing) but still inside the INCONCLUSIVE band rather than a
clear GENUINE — see Section 4's v7 status for
final numbers.

### Bug #35 — "hu" fusion component used raw cosine similarity over 7 raw Hu-moment values with wildly different scales, and at least one of them provably flips SIGN between two of the user's own genuine samples

**Root cause, found by printing real Hu-moment values:** `hu_6` measured
`+3.4973` on the user's enrolled image 1 and `-2.6755` on their enrolled
image 2 — the SAME signature, both examiner-confirmed genuine, opposite
sign on this one moment. `hu_5` and `hu_7` showed the same pattern
against image 3 (a third genuine sample): `hu_7` was `-3.535` vs. `+5.382`,
`hu_5` was `+3.221` vs. `-4.621` — full sign inversions, not drift. This
is not a bug in the Hu-moment computation — it's a well-documented
property of Hu's original 7 invariants that higher-order moments (5, 6,
7 specifically) are increasingly sensitive to noise/discretisation, and
moment 7 is specifically a skew/reflection indicator that's DESIGNED to
flip sign under a mirror flip of the shape. A cosine similarity over all
7 raw values (very different magnitudes: hu_1 ~0.02-0.06 vs. hu_5/6/7
~2-5) is dominated by whichever dimension has the largest magnitude, and
a sign flip on one of those dominant dimensions can crater the whole
"hu" component even between two undisputedly genuine samples.

**Fix:** `similarity.hu_pairwise_sim` now scores Hu similarity the same
way as any other scalar feature — per-dimension |query − reference|,
scaled by this profile's own observed natural variation on that exact
dimension (reusing `pairwise_scalar_sim`) — AND restricts the scored
"hu" fusion component to the 4 lower-order, empirically-stable moments
(`HU_FUSION_KEYS = {hu_1, hu_2, hu_3, hu_4}`), excluding 5/6/7 from the
automated score entirely (they remain visible, unweighted, in the human
metrics table). Verified directly: restricting to the 4 stable moments
gave 0.45 similarity for a real genuine pair (image 3 vs. image 1) vs.
0.28 for a real forged pair (image 4 vs. image 1) — a real, usable
margin; using all 7 gave ~0.26 for BOTH, no usable margin at all.

### Bug #36 — pixel-term (ssim/iou) alignment penalty fired on the BEST case, not the worst case — and the first attempt at fixing it caused a real regression, caught by re-running the same repro images

`align.py`'s `aligned_ok` means "a non-identity transform beat identity"
— which is FALSE when identity was already optimal (best possible case:
query already near-perfectly registered against this reference, nothing
can improve on doing nothing) just as much as when nothing could align
at all (worst case). Confirmed directly: comparing an enrolled image
against itself gave `aligned_ok=False` (identity already had ink IoU
1.0) and was then penalised 10% on ssim/iou as if alignment were
UNRELIABLE — the opposite of correct.

**First fix attempt (caught before shipping):** keyed the penalty off
`best_iou >= 0.30` instead of the `aligned_ok` boolean. This fixed the
self-check case, but re-running the FULL repro set (not just the one
case being fixed) showed it introduced a new problem: it also fed into
`_edge_case_review`'s "alignment reliability" check, which used the same
0.30 bar to decide whether to downgrade a verdict to INCONCLUSIVE — and
image 4 (a REAL forgery) legitimately can't achieve good overlap with
the genuine references even after best-effort alignment (that's
correct — the shapes really do differ), which flipped its verdict from
correctly-FORGED to incorrectly-INCONCLUSIVE. Poor overlap against a
real difference is itself evidence, not "unreliable evidence" — the two
questions ("is overlap good enough to trust ssim/iou at full weight" vs.
"did alignment fail so catastrophically nothing here can be trusted at
all") got conflated into one threshold.

**Actual fix:** two SEPARATE bars for two separate questions.
`_MIN_RELIABLE_IOU = 0.30` (in `_score_against_ref`) still gates the
ssim/iou 10% penalty — "is overlap good enough to trust pixel terms at
full weight". A much lower `_MIN_SANE_IOU = 0.08` (in
`_edge_case_review`) gates the INCONCLUSIVE-downgrade check — "did
alignment fail so badly (near-zero overlap even after best effort) that
NONE of this evidence can be trusted for any verdict" — which only fires
for genuine registration catastrophes (wrong image, blank page, corrupt
crop), never for "these two signatures are just different enough that
even good alignment shows moderate overlap." Verified: after this
correction, image 4 is back to correctly FORGED, and the self-check
penalty fix is preserved.

**Process note, worth keeping:** this is a concrete example of why
Section 6's "test with the full repro set after every change, not just
the case you're fixing" rule matters — the first attempt would have
shipped a real regression if only the originally-reported case had been
re-checked.

### Bug #37 — raw pixel ink-IoU is pathologically sensitive to sub-pixel misalignment and photo-to-photo stroke-width differences, unrelated to whether it's the same signer

Printed proof: comparing image 3 against image 1 (both genuine, per the
user) gave ssim=0.82-0.83 (real structural agreement) but raw ink-IoU of
only 0.26-0.29 — on a ~5-6px-wide pen stroke, a 1-2px residual
registration offset or width difference (purely from lighting/focus/JPEG
compression of a *different photograph*, not from how the pen was held)
collapses IoU disproportionately, because IoU on thin lines has none of
SSIM's local-neighbourhood tolerance. With `iou` weighted at 0.16 in the
fusion sum, this alone was worth ~0.09-0.10 of the fused score.

**Fix:** `similarity.ink_iou` now dilates both masks by a small, fixed
radius (`_IOU_TOLERANCE_PX = 3`, tuned empirically against the repro
set) before computing IoU — a standard "tolerance band" technique for
thin-stroke overlap comparison, applied symmetrically to both images.
Verified this does NOT rescue a real forgery: image 4's iou moved only
0.20→0.24 (still a clear rejection) while image 3's recovered to a much
more representative range.

### Bug #38 — `training/calibrate_weights.py`'s fitted weights file could never actually be loaded, silently, forever

Found while auditing this script for consistency with the Bug #34/#35
fixes above (not user-reported). `training/calibrate_weights.py`'s
`FEATURE_ORDER` had only 7 entries; `engine.py`'s real fusion
(`_HAND_GUESSED_WEIGHTS`) has had 9 since Bugs #31/#33 added `zonal` and
`construction`. `_load_default_weights()` only accepts a population
weights file whose keys EXACTLY match `_HAND_GUESSED_WEIGHTS.keys()` —
so any file this script ever produced was silently rejected at load
time, forever, with no error anywhere in either script. Fixed:
`_score_pair` now computes all 9 components using the same pairwise
(reference-specific) functions `engine.py` uses post-Bug #34/#35, so a
freshly-calibrated file is both loadable AND calibrated against the
actual scoring math that will run real verifications.

### Bug #39 — see Bug #35 (kept as a placeholder for numbering continuity — the Hu-moment fix above was originally drafted as two separate bugs and merged during writeup; no separate content)

### Bug #40 — ruled/notebook-paper lines survive into the binary ink mask untouched, corrupting every downstream measurement

**Reported as:** user's own image 7 (a signature photographed on lined
paper) — described by the user as "should be INCONCLUSIVE... because the
background having some line." Rendered the actual preprocessed binary
mask for this exact image and confirmed visually: 3 near-full-height
lines dominate the frame, running straight THROUGH the signature ink.
Neither existing filter catches these — `_remove_specks` only drops tiny
components (these lines are large), and `_remove_distant_outliers` (Bug
#26) only drops components spatially OUTSIDE the signature's region
(these lines run directly through it). Every downstream measurement that
depends on ink shape, position, or topology (aspect ratio, Hu moments,
stroke width/junction/endpoint counts, and every pixel-level comparison)
is corrupted by treating ruled-paper lines as if they were pen strokes.

**First fix attempt (caught before shipping):** a pure morphological
OPEN with a long axis-aligned (pure vertical/horizontal) structuring
element, on the assumption page rules are perfectly axis-aligned in
pixel space. Tested directly against the real image: it removed
*nothing* — the longest unbroken single-column run was only ~100px
against a ~567px target on an 810px-tall frame, because a real
photographed page is essentially never perfectly axis-aligned (a few
degrees of camera tilt — distinct from the 90-degree-multiple EXIF
rotation already handled — drifts a "vertical" line across several
columns over hundreds of pixels, breaking any single-column run long
before it reaches a length threshold).

**Actual fix:** `_remove_ruled_lines` (new, runs on the FULL-frame binary
mask before cropping/speck-removal) uses a probabilistic Hough transform
(`cv2.HoughLinesP`) to find long, straight, near-vertical or
near-horizontal line SEGMENTS (within `_LINE_ANGLE_TOL_DEG = 8°` of
either axis — covers ordinary camera tilt) with a generous minimum
length (`_LINE_LENGTH_FRAC = 0.70` of the frame's shorter dimension) and
tolerance for small gaps (where the signature's own ink crosses the
rule). Detected segments are drawn, dilated slightly, and subtracted
from the mask. Verified on the real reported image: 2 of 3 lines fully
removed, the third substantially thinned, with the signature ink itself
visibly intact. **Honest limitation:** the third line isn't perfectly
removed (likely where it directly overlaps dense signature ink, breaking
Hough segment continuity below the length threshold in that stretch) —
image 7 still resolves to FORGED after this fix (was already FORGED
before; the user's own description of the "correct" verdict for this
image was ambiguous, so no further tuning was done against this one
example specifically — see Section 7's new open question).

### Bug #41 — repeated re-verification + confirmation of an ALREADY-enrolled image could promote near-duplicate "new" references, collapsing the natural-variation estimate

**Reported as:** user's own description — "some time i recheck images 1
and 2 ... it starts to fail after 2 or 3 time." Root cause, confirmed by
direct simulation: `feedback()`'s test-time-calibration path (v5, Bug
#23) will promote ANY confirmed-GENUINE query into the profile as a new
reference if the examiner opts in, with no check for whether that query
is essentially a duplicate of something already enrolled. Simulated
exactly the reported workflow (verify image 1 → confirm correct + add as
reference, repeated 6 times): before this fix, each promotion added a
near-identical sample, which (combined with Bug #42 below) drove the
calibrated threshold toward its 0.92 hard ceiling within a handful of
repetitions — after which even the TRULY enrolled images started
failing, exactly the reported symptom.

**Fix:** before promoting, check whether this query already scored
`>= 0.97` against ANY existing reference (`_NEAR_DUPLICATE_SCORE`) — if
so, skip promotion (feedback is still recorded, with a clear
`promotion_skipped_reason` surfaced to the API/UI) since a near-duplicate
cannot add real information about this signer's natural variation and
can only distort the statistic that variation is measured against.
Verified: repeating the exact simulation above with this fix, `n_refs`
correctly stays at 2 across all 6 repeated confirmations.

### Bug #42 — feedback's threshold nudge had no bound relative to the profile's own calibrated evidence, in either direction

Found while testing Bug #41's fix (the promotion guard alone didn't fully
resolve the simulation above — the threshold kept drifting even with
`n_refs` correctly staying at 2). Root cause: `feedback()`'s
`FEEDBACK_THRESHOLD_STEP` nudge (±0.015 per event) is applied directly to
the live `stats["threshold"]`, with only the global `[MIN_THRESHOLD,
0.92]` clamp — no bound tied to what the actual enrolled evidence
supports. Confirmed: 6 repeated "correct" confirmations on the same
GENUINE case walked the threshold from 0.5691 down to 0.4941 with no
sign of stopping short of the global floor; the same mechanism in the
opposite direction (repeated "incorrect" feedback, or feedback on a
FORGED case) would walk it toward 0.92 — reproducing the exact
false-FORGED symptom this whole bug-fix pass exists to resolve, just via
a different mechanism than Bug #41.

**Fix:** `_relearn` now records `stats["calibrated_threshold"]` — the
fresh, purely statistical value it just computed. `feedback()`'s nudge is
now bounded to within ±0.10 of that value, in addition to the existing
global clamp. A real, evidence-grounded correction within that band is
still possible (that's the point of online feedback); it can no longer
walk arbitrarily far from what the enrolled evidence actually supports
no matter how many feedback events accumulate between recalibrations
(recalibration — i.e. a fresh `_relearn` call — resets the band's
center). Verified: the same 6-repetition simulation now stops drifting
at exactly `calibrated - 0.10` and holds there.

### Bug #43 — LLM load could hang indefinitely with no timeout, anywhere in the call chain, on either the GPU or the CPU backend

**Reported as:** "LLM: loading signal for more than 2min... not working
in GPU and not even in CPU the fallback method." Root cause: the
blocking `Llama(...)` native constructor call was invoked in-line with
NO timeout anywhere — if that call genuinely hangs (a documented,
real-world llama-cpp-python failure mode, distinct from the "silent CPU
fallback" Bug #32 already found and fixed), there is no Python exception
to catch, because nothing is failing — it's genuinely blocked. `_status`
stays at `"loading"` forever with no way for the user to tell "slow" from
"broken." A contributing factor found alongside this: `n_ctx=8192` (the
context window) directly inflates load time and memory footprint,
because llama.cpp allocates its ENTIRE KV cache upfront at construction
time, sized by `n_ctx` — checked the app's actual prompt size
(`llm/prompts.py`) and confirmed the real evidence-JSON + schema + retry
round trip comfortably fits under 2000 tokens, so 8192 was never
actually needed for this app.

**Fix, two parts:**
1. `GEN_PARAMS["n_ctx"]` reduced 8192 → 4096 (still 2x+ headroom over
   measured real usage) — halves KV-cache allocation time/memory on
   every backend.
2. `_run_with_timeout` (new): runs the blocking construction in a
   background thread and waits at most `LLM_LOAD_TIMEOUT_SECONDS`
   (default 180s, overridable via the `SIGNXAI_LLM_LOAD_TIMEOUT` env
   var) — converting an indefinite, unexplained hang into a concrete,
   actionable "error" status with a clear message, on EITHER the GPU or
   CPU path. Cannot literally kill a blocked native call from Python
   (acknowledged honestly — if the GIL is held in native code, the
   background thread keeps running), but if it later finishes on its
   own, `self._pending_construct` lets the NEXT load attempt pick up
   that late success instead of wasting it or double-loading. A
   lightweight heartbeat thread updates `self._progress` every 15s with
   real elapsed time so the UI shows movement instead of a static,
   indistinguishable-from-frozen "loading" even during a legitimately
   slow (not hung) load.

**Verified (self-run, with a mocked `llama_cpp.Llama` that sleeps
longer than the test timeout):** confirmed the manager reports a
definitive `status: "error"` with an actionable message within the
configured timeout window (not before, not never), and confirmed
separately that a late-finishing background attempt is correctly picked
up as `"ready"` by the next `load_now()` call instead of being wasted.
**Not independently verified against a real GGUF model or real GPU
hardware** (this sandbox has neither) — this is the top item for the
user to verify on their own machine: does the "still loading... Xs
elapsed" progress message actually appear during a real load, and does a
genuinely stuck load now convert to an "error" status within
`LLM_LOAD_TIMEOUT_SECONDS` instead of hanging forever?

### Bug #44 — GPU offload was always attempted at full strength (`n_gpu_layers=-1`) with no check that the detected GPU actually has enough VRAM for this model

Found while investigating Bug #43 for root causes beyond "no timeout
existed" — the user's own real install log (Section 2) shows an RTX
4050 Laptop GPU with 6.0GB VRAM. This model's GGUF+mmproj files alone
total ~4.5-5GB on disk; add this app's own KV cache (non-trivial even at
the now-reduced `n_ctx=4096`) and llama.cpp's runtime overhead, and the
real requirement plausibly exceeds 6GB. An over-committed GPU doesn't
always fail cleanly with an out-of-memory exception — severe
driver-level paging/thrashing can look exactly like an indefinite hang,
independently explaining the user's report even before Bug #43's timeout
existed.

**Fix:** before attempting GPU offload, estimate the real memory
requirement from the model's actual on-disk size (a reasonable proxy —
a loaded GGUF's resident size tracks its file size closely) plus a
documented KV-cache/overhead reserve (`_KV_CACHE_OVERHEAD_GB = 1.5`) and
a 15% safety margin. If the detected VRAM can't plausibly hold it, skip
the GPU attempt entirely and go straight to CPU, with a clearly logged
reason (surfaced via `self._progress` too, so the UI can show WHY GPU
wasn't used instead of the user having to guess). This can only ever
convert a doomed/hanging GPU attempt into a working CPU load — it never
makes GPU offload less likely to succeed when there genuinely is enough
VRAM. **Not independently verified against real GPU hardware** (same
honest caveat as Bug #43) — top item for the user to verify: on their
actual RTX 4050 (or similar tight-VRAM card), does the LLM now load on
CPU immediately with a clear "insufficient VRAM" log line, rather than
attempting GPU and hanging?

---

## 3c. v7.1 bugs (same session, user's first real hands-on test after v7) — LLM finally loaded and immediately hit a second bug it had been masking; a v7 fix regressed on the user's own signature style

> **Context**: the v7 fixes above were verified against this project's
> own 5-image repro set, in a sandbox with no GPU and no real model
> download. This round is the user's FIRST real run of v7 on their own
> machine (Windows, RTX 4050 6GB) with real new photographs and the real
> model — and it surfaced two things v7's own sandbox testing structurally
> could not have caught: a bug that was only reachable once the LLM
> actually finished loading (which v7's fix made possible for the first
> time), and a real, previously-untested interaction between the Bug #40
> ruled-line fix and this specific user's own actual signature style
> (which the original 5-image repro set didn't happen to stress-test
> the same way). Both are now fixed and tested; see each entry for what
> could and couldn't be verified without the user's own hardware/photos.

### Bug #45 — LLM examination crashed on every real attempt: `generate()` was passed PIL Image objects where it has always expected file-path/data-URL strings

**Reported as:** "CPU fallback loads it but it didn't make LLM to
provide explanation" — confirmed from the user's own log line: `LLM
examination unavailable, using template: 'Image' object has no
attribute 'startswith'`.

**Root cause:** `VisionLLMManager.generate(self, images: list[str], ...)`
has always expected plain strings (file paths or data-URLs — its own
docstring says so, and its body calls `img.startswith("data:")` on each
one). `report.py::_llm_examination` opened `PIL.Image.open(...)` objects
and passed those directly instead — a plain type mismatch. Confirmed
this is genuinely pre-existing (not a v7 regression): it predates every
v7 change to this file and could never have been caught before, because
the LLM could never get far enough past "loading" (Bug #43) to actually
reach this call. Fixing Bug #43 is what let a real run reach this code
path for the first time in this project's history and surface this
second, independent, previously-invisible bug.

**Fix:** pass the evidence images' file PATHS directly as strings
(`str(case_dir / "query.png")` etc.) — `generate()` already knows how to
wrap a bare path in a `file://` URL itself; no PIL round-trip was ever
needed. Verified end-to-end with a mocked `generate()` that asserts
every element it receives is a string and returns a valid JSON
examination: the call now completes and produces a validated
`Examination` object with no crash, where before it crashed on every
single real attempt (all 3 retries, per the user's log, before falling
back to the template).

### Bug #46 — the AI Model status page couldn't show WHY GPU wasn't used, or real load progress, because that reasoning wasn't persisted anywhere the API/UI could read it

**Reported as:** the user directly asked for the "Local AI Explanation
Model" section to "show me its reasoning and what it must needs to
show."

**Root cause:** Bug #44's VRAM-gating decision and Bug #43's progress
heartbeat both only ever touched `self._progress`, a single string
that's immediately overwritten to `"ready"` (or an error) once loading
finishes — so by the time anyone actually looked at the status page,
the "why CPU not GPU" reasoning was already gone. Separately, the web
UI's status panel never auto-refreshed while `status == "loading"`, so
even the heartbeat message that DOES exist during loading required a
manual page refresh to ever be seen — indistinguishable, in practice,
from a frozen page.

**Fix:** a new persistent field, `self._gpu_skip_reason`, set once at
load time and kept for the model's whole loaded lifetime (not
overwritten by later progress updates), surfaced through
`/api/llm/status` alongside `n_ctx` and `load_timeout_seconds` (also
requested implicitly — "what it must needs to show"). The web UI's AI
Model panel now displays this reasoning prominently, shows the live
`progress` string while loading, and polls itself every 4 seconds while
`status == "loading"` instead of requiring a manual refresh — so a real
load now visibly shows movement, and a real VRAM-driven CPU fallback
now visibly explains itself, both without the user needing to know to
check server logs.

### Bug #47 — the Bug #40 ruled-line-removal fix stripped a real user's own genuine signature strokes, because their signature's own style (a deliberate double-underline) is structurally indistinguishable from 2 ruled lines by angle and count alone

**Reported as:** "signature images are getting twisted and not in proper
dimension... 3.jpeg is throwing be FORGED... same for 1.jpeg, which i
had already inserted in input sample" — i.e. an ENROLLED reference
image itself was scoring FORGED, and the evidence visualizations looked
visibly corrupted/distorted (confirmed directly from the user's own
screenshots: a "Visual evidence" panel showing near-total red/blue
non-overlap between query and reference, and a zonal grid overlay drawn
on a heavily rotated, ruled-line-crossed image).

**Investigation, in order:**
1. First checked whether the widened Hough-based line detection
   (needed to also catch ruled lines in a ROTATED photo — see the
   user's second screenshot, which showed a real ruled line surviving
   completely untouched because the whole photo was tilted ~15-20
   degrees, well past the original fix's `+/-8 degree` axis assumption)
   was the right direction. It was, but implementing it exposed the
   real, deeper problem.
2. Removed the axis-alignment assumption entirely and replaced it with
   the actual distinguishing signal of ruled paper: MULTIPLE
   mutually-parallel long lines, at any angle. Tested against a
   synthetic single straight stroke at 9 different angles (0 through
   170 degrees) to check this could never fire on a lone genuine
   flourish — it failed every single one, always removing the stroke.
3. Root cause of that failure: `cv2.HoughLinesP` returns SEVERAL
   overlapping/adjacent segment detections for one physical thick line
   (confirmed directly: a single synthetic 3px-thick stroke produced
   multiple near-identical Hough segments), and a naive angle-only
   grouping miscounted these duplicate detections of ONE line as
   multiple "parallel lines". Fixed with a two-stage approach: first
   deduplicate detections that share both angle AND position
   (perpendicular offset) into one physical line, THEN group the
   deduplicated lines by angle only. Re-tested: all 9 synthetic
   single-stroke angles now correctly left untouched, and the real
   ruled-paper test image still correctly had its lines removed.
4. Then tested against this project's own 5-image repro set (all 5
   have NO ruled paper) as a sanity check before considering this
   fixed — and found the "fixed" version still incorrectly stripped
   3,000-5,000+ ink pixels (30-50% of total ink) from EVERY one of the
   4 non-ruled-paper images tested. Root cause, found by printing the
   actual deduplicated lines detected: this user's own signature style
   genuinely includes TWO real, near-parallel underline strokes as a
   deliberate flourish — which is, by angle and count alone, LITERALLY
   INDISTINGUISHABLE from 2 ruled lines. No amount of angle-tolerance
   tuning fixes this, because the ambiguity is real, not a measurement
   error.
5. Fixed with a decisive, different signal instead of tuning the same
   one further: genuine ruled paper lines span almost the ENTIRE
   photographed frame (because the page is bigger than the signature
   drawn on it, and a photograph normally includes margin around the
   subject) — a signature's own flourish, however long relative to its
   OTHER strokes, essentially never spans the full frame edge-to-edge.
   Added `_MIN_FRAME_SPAN_FRAC = 0.85`: a candidate line's actual
   measured extent must reach at least 85% of the longest a straight
   line at that exact angle could possibly be while still fitting
   inside the photographed frame, BEFORE it's even considered for the
   parallel-family grouping. This is a fundamentally different kind of
   check (absolute geometry relative to the frame, not relative
   comparison between lines) and is what actually resolves the
   ambiguity step 4 found — verified directly: with this filter, ALL 4
   of this project's own real signature images (including the ones with
   the genuine double-underline) now correctly show ZERO pixels
   removed, while the real ruled-paper test image still correctly has
   its lines detected and removed.

**Verified:** re-ran the full 9-angle single-stroke safety sweep (all
pass), the real ruled-paper image (still correctly cleaned), AND all 4
non-ruled-paper repro images (all now correctly show zero removal,
where the intermediate broken version showed 30-50% ink loss on every
one) — plus a full end-to-end engine regression (enroll + verify all 5
repro images): every verdict and score is byte-for-byte identical to
the pre-Bug-#47 numbers in Section 4's v7 status table, confirming this
fix changes ONLY the ruled-line behavior and nothing else.

**Not independently verified:** the user's own two new specific test
images (the ones that produced the FORGED verdicts on what should be an
enrolled self-check) were not available as raw files — only two PDF
reports and two UI screenshots were provided, which was enough to
diagnose and fix the mechanism above with high confidence, but not
enough to re-run the EXACT reported cases and confirm the verdict
itself now comes out correctly. **This is the top item for the user to
re-verify**: re-run the same enrollment and the same two verification
images that produced cases `510225173871` and `7fb6bee95229` against
this fixed version, and confirm both now score correctly (the enrolled
self-check in particular should be at or near a perfect match, not
FORGED).

---

## 3d. v7.2 bugs (user's second real hands-on round) — the URI-construction fix for Bug #45 was itself platform-broken, and a repeated cross-signer pattern in two of the nine fusion components

### Bug #48 — Bug #45's own fix broke on the ONE platform it needed to work on: Windows

**Reported as:** the crash changed shape after the v7.1 delivery, from
`'Image' object has no attribute 'startswith'` to
`<urlopen error [WinError 3] The system cannot find the path specified: ''>`
— on the user's real Windows machine, the exact target Bug #45 was
fixed for.

**Root cause:** Bug #45's fix was right about WHAT to pass to
`generate()` (a path string, not a PIL object) but wrong about HOW to
turn that path into a `file://` URL — it used `f"file://{path}"`, plain
string concatenation. This happens to produce a valid URI on POSIX
(`file:///home/user/x.png`) purely by accident, but is not a valid URI
on Windows: a real path like `D:\signxai\data\reports\...\query.png`
has backslashes (not valid URI separators) and a drive letter where a
URI expects a host. Concatenating naively produces
`file://D:\signxai\...`, which urllib's file-URL opener parses as host
`"D:"` and an EMPTY path — precisely the reported `WinError 3` on `''`.
This is exactly the kind of single-platform assumption
`PROJECT_MEMORY_v8.md`'s own process rules exist to catch, and it
slipped through because this sandbox has no Windows machine to run the
fix against end-to-end.

**Fix:** `_path_to_file_uri()`, using `pathlib.Path(path).resolve().as_uri()`
— the standard library's own OS-aware path-to-URI conversion, which
correctly handles drive letters, backslash-to-forward-slash conversion,
and percent-encoding on every platform, instead of hand-rolled string
formatting. **Verified without a real Windows machine** (still not
available in this sandbox) using `pathlib.PureWindowsPath` — the
standard library's platform-independent Windows-path-manipulation
class, built specifically for testing Windows path logic on any OS —
confirming `PureWindowsPath(r"D:\signxai\data\reports\...\query.png").as_uri()`
correctly produces `file:///D:/signxai/data/reports/.../query.png`.
This is real verification of the underlying stdlib logic, not a guess,
but is still not the same as an actual end-to-end run on Windows —
**this is the top item for the user to confirm**: does report
generation now actually complete on their machine, past this specific
error?

### Bug #49 — two of the nine fusion components (`hu`, `scalar_profile`) have now dragged down confirmed-genuine matches across TWO independent real signers' data, not just an isolated case

**Reported as:** "for genuine, it is now providing Inconclusive" — and
this time, the user's own real engine measurements were captured
verbatim (embedded in the LLM prompt payload, logged because the LLM
itself was still failing on Bug #48 at the time this data was
captured) — real, hard numbers, not a report summary: a case with
ssim=0.97, iou=0.91, dtw=0.97, construction=0.91, zonal=0.92 — unanimous
by every pixel/topology signal — landed at only 0.71 fused and got
downgraded to INCONCLUSIVE by the reference-disagreement edge-case
check, because `hu` (0.21) and `scalar_profile` (0.53) dragged the
weighted average down despite carrying 20% of the total weight between
them.

**Why this is a real, repeated pattern and not one unlucky case:**
this project's OWN self-testing when verifying Bug #34/#35's fix
already found `hu` topping out around 0.45 and `scalar_profile` around
0.53 for a confirmed-genuine pair — the SAME two components,
underperforming on confirmed-genuine comparisons, on TWO independently
real signers' data (this project's own test signer, and now this
user's). That is a real signal, not noise.

**Fix, two complementary parts, each verified separately against real
numbers before combining:**

1. **Reweighted `_HAND_GUESSED_WEIGHTS`** away from `hu`/`scalar_profile`
   and toward the components with a consistent track record across
   both real datasets (`construction`, `dtw`, `zonal`) — but MODERATELY,
   not aggressively. A first, larger reweighting draft was tested
   directly against the user's own logged numbers (best-reference score
   0.809 -> 0.873, gap between a real match and a real mismatch from
   this same user's data widening from 0.353 -> 0.389 — genuine
   improvement, verified with their real numbers, not assumed) — but
   re-testing that SAME larger change against this project's own 5-image
   repro set (a DIFFERENT real signer) revealed `construction` is one of
   the WEAKER components for that signer (~0.31-0.34) where
   `scalar_profile` is comparatively stronger — the opposite pattern.
   Picking a global weight vector that favors whichever component suits
   whichever single signer was most recently reported is exactly the
   overfitting this project's process rules exist to prevent (Section
   6). Settled on a smaller, more conservative reweighting instead —
   real, verified improvement, without betting the whole fusion on one
   signer's particular feature-strength profile.
2. **Raised `_FEATURE_PRIOR_REL`** (the natural-variation tolerance
   introduced in Bug #34) from 0.18 toward a more generous value — this
   lever is more generalizable than reweighting (it doesn't pick
   winning/losing components, just loosens how harshly ordinary
   phone-photo variation is punished on these two specific families).
   Tested a larger increase (0.28) first and REJECTED it after
   verifying it measurably eroded discrimination against this project's
   own known real forgery (image 4's score moved from 0.4845 closer to
   its threshold band) — loosening a shared tolerance helps ANY pair
   look more similar, including ones that should score low, which is
   the wrong trade. Settled on a smaller increase (0.22) that still
   helps the genuine-case pattern without measurably narrowing the
   real-forgery margin — verified directly, not assumed.

**Honest, explicit limitation:** this is real, verified, evidence-based
improvement, not a claim to have solved the underlying tension. A
single global weight vector, tuned against at most 2 real signers'
data, cannot fully resolve the fact that different signers' signatures
apparently vary more or less across different feature families. The
architecturally correct fix — per-profile ADAPTIVE weighting, trusting
whichever components are empirically stable for THIS SPECIFIC signer's
own enrolled samples rather than one fixed vector for everyone — is a
real, well-specified idea this round surfaced, added to Section 5's
roadmap, not yet implemented.

---

## 3e. v8 bugs (user provided real, saved case data for the first time — the most decisive round yet)

> **Context**: this round the user attached `data.zip` — a real export of
> their actual `data/` directory: profile.json files, enrolled reference
> images, and full case.json + saved intermediate images (query.png,
> reference.png, query_original.png) for real verification attempts.
> This is qualitatively different from every prior round: instead of a
> PDF report or a log excerpt, this is the ACTUAL bytes the engine
> processed, at every stage. Every bug below was found by directly
> loading and inspecting these real files (`view`-ing the actual saved
> binary masks, running `preprocess.preprocess()` against the actual
> enrolled reference photos), not by inference from summarized numbers.
> This is the strongest evidence this project has had access to yet,
> and it found the single most consequential bug of the whole project.

### Bug #50 — global Otsu binarization classified 46% of an entire real photo as "ink," because of an uneven lighting gradient across the page

**Reported as:** the user's own vague but ultimately correct intuition —
"the image is getting disoriented in 8 scale frame pixel... the
preprocessing layer... is also a culprit that making the end result or
remark wrong."

**Found by:** loading a real enrolled reference image
(`profiles/65bde2bf7c5a/ref_0.png`, an actual phone photo of a signature
reading "Saurabh") directly through `preprocess.preprocess()` and
rendering the resulting binary mask. The result: broad diagonal streaky
regions and mottled speckle noise covering most of the 1200x1600 frame,
with the real signature almost lost among it — **886,129 of 1,920,000
pixels (46%) classified as ink**, when the actual signature occupies a
small fraction of the page. This single failure cascaded through
everything downstream: the crop-to-content bounding box computed as
`(0, 0, 950, 1599)` — essentially the entire page, not the signature —
which then corrupted every feature and comparison that followed for
this reference. This is almost certainly the single largest contributor
to this project's repeated "can't recognize a new signature type"
reports across every prior round: it isn't specific to one signer or
one photo, it affects ANY photo taken with uneven lighting, a shadow, or
ordinary lens vignetting — arguably the norm, not the exception, for a
handheld phone photo that isn't a flat, evenly-lit scan.

**Root cause:** plain global Otsu thresholding (used since v1, never
revisited) picks ONE brightness threshold for the entire image, which
implicitly assumes uniform lighting across the whole frame. A real phone
photo commonly has smoothly-varying brightness (a lamp to one side, a
shadow from the photographer's own hand, natural lens vignetting) —
when the dark end of that gradient drops below Otsu's single global
threshold, large regions of plain background paper get misclassified as
ink, exactly as observed.

**Fix:** `_flatten_illumination()` — standard document-image-analysis
"flat-fielding": estimate the smoothly-varying background brightness
with a large-kernel median blur (large enough to wash out the thin ink
strokes entirely, leaving only the page's own brightness variation),
then divide the original by that estimate before Otsu ever sees it.
Applied as a single new step ahead of the existing binarization, so
every other stage of the pipeline is otherwise unaffected. **Verified
directly, not assumed**: ink-pixel count on the exact reported image
collapsed from 886,129 to 10,298, and the resulting mask visually
contains ONLY the real handwritten ink — the lighting artifact is
completely gone (rendered and inspected directly). Crop bounding boxes
for this project's own 5-image repro set (none of which had this
problem) are byte-identical before and after this change — this fix
only activates when there's actually an illumination problem to correct.

**Downstream impact, verified directly against real data:** re-enrolling
the real "Saurabh" profile with this fix produced a threshold of 0.577
(up from a previously near-vacuous 0.3 — the two enrolled references
are now measured as far more mutually consistent, because the lighting
artifact that was corrupting one of them is gone), and re-running the
exact real cross-signer mismatch case from this same data (a "Vinit
Patil" query tested against the "Saurabh" profile — two genuinely
different people's signatures) now correctly scores 0.375 (clearly
FORGED) instead of the previous individually-"GENUINE" 0.40-0.53
readings.

### Bug #51 — a UNANIMOUS, correctly-scored FORGED verdict was still downgraded to INCONCLUSIVE by the alignment-reliability edge case

**Found by:** the SAME real cross-signer mismatch case above. Even after
Bug #50's fix corrected the score itself (0.375, clearly below
threshold), the case's `edge_case_review` showed `triggered: true` with
`alignment_reliability: 0.0`, downgrading a clean, unanimous verdict
(`individual_verdicts: ["FORGED", "FORGED"]`, `reference_agreement: 1.0`,
`opposing_evidence_fraction: 0.0` — zero disagreement whatsoever) to
INCONCLUSIVE.

**Root cause:** `_edge_case_review`'s alignment-reliability check (v7
Bug #36) never considered whether the references actually AGREED with
each other — it fired purely on raw overlap quality, regardless of
consensus. This directly contradicts the check's own documented purpose
("catch verdicts NOT well supported by evidence") — a unanimous verdict
across every reference, with zero disagreement, IS well-supported
evidence, whether that consensus is GENUINE or FORGED. Poor pixel
overlap between two genuinely different signatures is not a measurement
failure; it's the FORGED signal working exactly as intended.

**Fix:** the alignment-reliability observation is now split into two
tiers. It's always computed and surfaced (a new `informational_flags`
field) for a human reviewer's transparency regardless of outcome, but it
only counts toward the automatic INCONCLUSIVE downgrade
(`triggered`/`flags`) when it's ALSO paired with genuine disagreement
among the individual reference verdicts (`agree_frac < 1.0`) — a real
registration catastrophe (wrong file, blank page, corrupt crop) would
realistically also produce inconsistent per-reference readings, not
clean unanimous agreement, making this a reasonable, evidence-grounded
proxy for "alignment genuinely failed" vs. "alignment correctly measured
a real, large difference." **Verified directly**: the same real case now
correctly reports `triggered: false`, `verdict: FORGED`, with the
alignment observation preserved as an informational note rather than
silently overriding a well-supported verdict.

### Bug #52 — install.py's GPU-backend decision only checked "is an NVIDIA GPU present," never "can this machine actually load a CUDA-linked binary"

**Reported as:** user-requested directly — "I want the --llm
installation script check does cuda based utility are ready to use in
system."

**Root cause:** the existing check ran `nvidia-smi` and, if it succeeded,
unconditionally picked the `cu121` wheel index — confirming only that an
NVIDIA GPU and driver are present, not that the driver's CUDA runtime
version is new enough for that wheel, or that Windows' Microsoft Visual
C++ Redistributable (a separate, commonly-missing prerequisite) is
installed. This is precisely the gap behind this project's own
previously-documented real failure (Bug #25): the cu121 wheel installed
successfully and only failed at actual import time with
`FileNotFoundError: Could not find module`, after a ~450MB download.

**Fix:** `_check_cuda_runtime_ready()` — a cheap pre-flight check run
BEFORE attempting the GPU wheel: (1) parses `nvidia-smi`'s own reported
"CUDA Version: X.Y" (the driver's maximum supported CUDA runtime — no
separate toolkit install needed for this) and requires it to meet the
cu121 wheel's actual floor (CUDA 12.1); (2) on Windows, checks for
`vcruntime140.dll` in `System32` as a heuristic for VC++ Redistributable
presence, the exact prerequisite this project's own prior real failure
named. Either failing skips straight to the CPU wheel with a clear,
actionable message (a driver-update or VC++-Redistributable download
link), saving a doomed large download and a confusing later error.
**Verified** the version-string parser against this project's own real
recorded `nvidia-smi` transcript (Bug #25's install log) — real evidence
the parsing works against actual `nvidia-smi` output, not a guess. This
is a pre-flight check, not a replacement for the existing post-install
GPU-offload confirmation (Bug #32), which remains authoritative once a
wheel is actually installed and loaded.

### Bug #53 — user-requested model swap: the specific model named turned out to be technically incompatible; switched to a smaller, correctly-matched alternative instead

**Reported as:** user asked directly to switch the LLM to
`google/gemma-4-12B-it-assistant`, to get better-quality "elaboration and
explanation."

**Investigated before changing anything** (web search, Hugging Face
model cards) rather than implementing the literal request blind: that
exact model is a "draft"/MTP (multi-token-prediction) assistant model,
explicitly documented by its own publisher as **not intended to be run
as a standalone chat model** — it only works paired with a full-size
`google/gemma-4-12B-it` for speculative-decoding acceleration, and
cannot generate a report on its own. Its real full-size counterpart is
also LARGER than the model already in use (~7.2GB vs ~5.7GB at Q4_0) —
switching to it would have made this project's actual, repeatedly-
reported problem (GPU offload failing on a 6GB card because the model
needs ~8.3GB, per the user's own pasted error this round) measurably
WORSE, not better.

**Fix implemented instead:** switched the single global model (see
Section 1's "one model, no tiers" policy, unchanged in principle) to
`google/gemma-4-E2B-it-qat-q4_0-gguf` — same publisher, same naming
convention, same QAT/Q4_0 format and Gemma-4 multimodal support as the
model already in use, at roughly HALF the size (~2.84GB vs ~5.7GB,
checked directly against the real Hugging Face GGUF file sizes, not
assumed). This comfortably fits a 6GB GPU's VRAM even with realistic
KV-cache/runtime overhead, which the previous model measurably did not.
`MIN_VIABLE_TOTAL_GB` and `MODEL_PRESENT_MIN_GB` in `app/device.py`
updated to match. **Not verified against a real download or real GPU**
(same sandbox limitation as every prior LLM-related fix) — top item for
the user to confirm: does the smaller model now actually offload to
their RTX 4050 successfully?

### Bug #54 — one global fusion-weight vector cannot fit every signer; added real per-profile adaptive weighting

**Motivation:** explicitly requested this round ("the system layers had
to be made in that consideration" that "each person signature is
different and so as their style") and already flagged as a concrete
roadmap item after Bug #49 found direct evidence of the underlying
problem: `construction` was one of the most reliable components for one
real signer's data and one of the weaker ones for a different real
signer — no single global weight vector can be right for both.

**Implementation:** at `_relearn` time, alongside the existing pairwise
intra-reference comparison (already computed for threshold calibration),
now also captures each pair's full per-COMPONENT breakdown (previously
discarded, only the final fused score was kept). For each of the 9
fusion components, computes this SPECIFIC profile's own self-consistency
— how similar the component's readings are between this signer's own
enrolled reference pairs. A component that stays high and consistent
between a signer's own genuine samples is evidently a reliable signal
for THAT signer; one that swings around even between their own
confirmed-genuine samples is evidently noisy for them. This reliability
estimate is blended with the global weight vector (reliable components
get relatively more weight, noisy ones relatively less, renormalized to
sum to 1), using the same pair-count-based fade-in blend already
established elsewhere in this function — needs at least 3 references (2+
pairs) to say anything meaningful; with exactly 2 references (a single
pair, no way to distinguish "reliable" from "lucky"), falls through
unchanged to the global weights, consistent with this project's
established practice of not trusting few-sample estimates. The resulting
per-profile weights and the raw reliability scores are both stored on
the profile (`stats.weights`, new `stats.component_reliability`) for
transparency.

**Verified:** enrolled a synthetic 3-reference profile (mixing this
project's own test images with a real user reference) and confirmed the
adaptive weights are computed, correctly blended (3 pairs / 6 reliable-
pair-count = 50% blend toward the adaptive estimate), and sum to exactly
1.0. Full regression re-run against every existing test case (this
project's own 5-image set, the real Saurabh mismatch, the real vinit
patil genuine case) — all verdicts and scores unchanged or improved from
before this change, confirming it doesn't regress the 2-reference
(unadapted) path other rounds already validated.

**Honest limitation:** this has NOT been validated against a real
multi-signer benchmark (the same caveat as Bug #49) — it's a real,
mechanism-verified implementation of the previously-proposed roadmap
idea, not a claim that the specific reliability formula (simple mean
pairwise similarity, floored at 0.05) is optimally tuned. Real multi-
signer data (Section 5 Phase 2c) remains the way to actually validate
or improve this further.

---

## 4. Current status / latest numbers

- **Version:** v1 (this delivery) — first runnable version. Prior state was an unrunnable
  backend skeleton (see Bug #2).
- **End-to-end test performed, 2026-07-20 (real CEDAR images, live server, not mocked):**
  - Enrolled writer 1's 2 genuine CEDAR samples -> adaptive threshold calibrated to 0.6947 (1-ref case
    separately tested calibrated to 0.55).
  - Verified genuine (same writer, held-out sample) -> **GENUINE**, score 0.8556, confidence 90.5%.
  - Verified skilled forgery (same writer's forger) -> **FORGED**, score 0.4481-0.5171 across two
    test runs, confidence 80.6-92.3%.
  - Both web static frontend (index.html/style.css/app.js all HTTP 200) and REST API
    (profiles, enroll, verify, feedback, asset serving, PDF/HTML report generation) confirmed
    working. Terminal CLI (`cli.py`) confirmed importable and its non-interactive functions
    (`do_device`, `do_case_log`, `block_bar`) confirmed rendering correctly with `rich`.
  - Name-collision dedup workflow confirmed: second `POST /profiles` with an existing name
    correctly returns 409 with guidance; `check-name` correctly reports the existing match.
  - LLM layer confirmed to fail gracefully when not installed (`No module named 'llama_cpp'`
    logged, cleanly falls through to `template_examination()`, no crash) — this IS the correct
    behavior for a machine without the optional LLM deps installed, not a bug.
- **NOT tested:** the LLM layer with actual GGUF weights loaded and generating — this build
  sandbox has no GPU and restricted network (can't pull multi-GB model files here). The prompt
  contract, retry loop, and manager code are implemented and reviewed but only exercised via
  the "not installed" fallback path so far. **This is the top item for the user to verify on
  their own machine** and report back on.
- **What's stubbed/deferred:** none currently — v1's core scope (enroll, verify, explain via
  template, web UI, CLI, installer) is implemented and tested. LLM narrative quality is
  unverified per above.
- **Accuracy:** not independently benchmarked at scale in this delivery — only 2 real signature
  pairs were tested (1 genuine match, 1 forgery, both correctly classified — but N=2 is a smoke
  test, not a benchmark). Section 2's ProtoSig numbers (85-90% range for 2-3 samples, different
  system) are the best current *reference point*, not a measured result of this codebase.
  Phase 2 (Section 5) is a full CEDAR benchmark run of this exact engine.

### v7 status (2026-08-30) — real false-FORGED bug root-caused and fixed, verified against the user's own 5-image repro set; LLM hang fix implemented and unit-tested, not yet confirmed on real hardware

- **The headline bug is fixed, not just mitigated.** The user's own
  reported case (real genuine image scoring FORGED) was reproduced
  directly in a sandbox against the user's own 5 photographed test
  images, root-caused to 2 independent, previously-undiscovered
  statistical/design bugs (#34, #35) plus 3 compounding ones (#36, #37,
  and the ruled-paper-line contamination in #40), and verified fixed
  against the SAME images after each fix, not just asserted:
  | image | v6 (before) | v7 (after) |
  |---|---|---|
  | 1 (enrolled ref, self-check) | GENUINE, fused score 0.81 | GENUINE, fused score 0.92 |
  | 2 (enrolled ref, self-check) | GENUINE, fused score 0.81 | GENUINE, fused score 0.92 |
  | 3 (held-out genuine) | **FORGED, 0.5165** (the reported bug) | **INCONCLUSIVE, 0.6119** (above threshold 0.5841 but inside the INCONCLUSIVE band — see note below) |
  | 4 (real forgery) | FORGED, 0.4577 (correct) | FORGED, 0.487 (still correct) |
  | 7 (genuine, photographed on ruled paper) | FORGED, 0.3581 | FORGED, 0.398 (line-removal implemented and confirmed working visually; verdict itself unchanged — see Section 7) |
- **Two feedback-loop bugs (#41, #42) also fixed and verified**, directly
  addressing the user's separate report that repeatedly re-checking
  already-enrolled images and confirming them eventually made even those
  images fail. Simulated the exact reported workflow (6 repeated
  verify+confirm cycles on an enrolled image): before the fix, this
  reproduced a runaway threshold; after, both the near-duplicate
  promotion guard and the bounded threshold-nudge hold the profile
  stable across all 6 repetitions.
- **LLM hang (Bug #43/#44) has a real, tested fix at the Python/logic
  level** (timeout + late-success reconciliation + progress heartbeat +
  VRAM-aware GPU gating + reduced `n_ctx`), verified with a mocked
  hanging constructor standing in for `llama_cpp.Llama` (this sandbox has
  no GPU and no network access to download the real ~5GB model). **This
  is the single most important thing for the user to confirm on their
  own machine** — does a real load now show progress and time out
  cleanly instead of hanging silently past 2 minutes?
- **One fix attempt was caught and corrected before shipping** (Bug #36's
  first draft caused a real regression on image 4, caught only by
  re-running the FULL repro set, not just the case being fixed — see
  Section 6 for why this is now a hard process rule, not just a good
  idea).
- **Full regression pass**: every module imports cleanly; a full
  enroll → verify → feedback → report-generation cycle runs end to end
  without error; `cli.py`'s device diagnostics run correctly;
  `training/calibrate_weights.py` was fixed for a latent (not
  user-reported) staleness bug (#38) found during this pass.
- **What's still open, honestly:** image 3 moved from confidently-wrong
  FORGED (0.52, below threshold) to INCONCLUSIVE with a positive margin
  above the calibrated threshold (0.61 vs. 0.58) — correctly reflecting
  real uncertainty rather than a wrong confident answer, but
  reaching that took 5 compounding fixes for one borderline case with
  only 2 enrolled references — this system's own long-standing guidance
  (enrolling 3-5 references) still matters more than any further
  threshold tuning would. Image 7's verdict didn't change even though
  the ruled-line removal is confirmed working visually — see Section 7.
  No fix in this pass was tuned by adjusting a parameter until one
  specific test image flipped the "right" way with no other
  justification; every number changed (`_IOU_TOLERANCE_PX=3`,
  `_FEATURE_PRIOR_REL=0.18`, `_MIN_RELIABLE_IOU=0.30`,
  `_MIN_SANE_IOU=0.08`, `_NEAR_DUPLICATE_SCORE=0.97`,
  `HU_FUSION_KEYS`) is defended above by a mechanism-level reason plus a
  check that it didn't erase discrimination against the real forgery
  (image 4) — but they are still empirically-chosen constants on a
  5-image, 1-person repro set, not a calibrated statistical fit. Real
  multi-signer data (Section 3's Bug #17/v3-decision discussion) remains
  the only way to actually validate these numbers beyond "didn't break
  the 5 images we have."

### v7.1 status (2026-08-30, same day) — user's first real run surfaced a masked crash and a real regression in the fresh ruled-line fix; both root-caused and fixed same-session

- **Trigger:** the user ran v7 for real for the first time (Windows,
  RTX 4050 6GB, real model download) and reported 3 things: the LLM
  loaded (fast — 12.2s, confirming Bug #43's fix worked) but crashed on
  every actual report-generation attempt; an ENROLLED reference image
  itself scored FORGED; and the evidence visualizations looked visibly
  distorted. Two real PDF case reports and two UI screenshots were
  provided as evidence (no raw signature photos this round).
- **Bug #45 (LLM crash) — fixed and verified.** A genuine, pre-existing
  type-mismatch bug (`report.py` passed PIL Image objects where
  `manager.generate()` has always expected strings) that could only be
  reached once Bug #43's fix let the LLM actually finish loading for
  the first time. Verified with a mocked `generate()` that asserts
  every image argument is a string — the crash is gone, the call
  completes and produces a validated examination.
- **Bug #46 (AI Model transparency) — implemented.** The user asked
  directly to see the model's reasoning. Added a persistent
  `gpu_skip_reason` field (survives past the loading window, unlike
  the transient progress string), exposed `n_ctx`/`load_timeout_seconds`,
  and made the web UI's status panel auto-poll every 4s while loading
  instead of requiring a manual refresh.
- **Bug #47 (ruled-line removal regression) — the important one.** The
  Bug #40 fix, while it worked on this project's own repro set, had
  never been tested against this SPECIFIC user's actual signature
  style, which turned out to include a genuine double-underline
  flourish that is literally indistinguishable from "2 ruled lines" by
  angle and parallel-count alone. Found this the hard way: fixing the
  angle-tolerance to also catch ROTATED ruled paper (needed per the
  user's own screenshot showing a missed line on a tilted photo)
  required testing against genuine signature strokes as a safety check
  — which is what caught the real double-underline conflict, BEFORE
  shipping, by testing against this project's own 5-image set as
  standard practice (Section 6, rule 10) even though none of those 5
  images are ruled paper. Fixed with a fundamentally different,
  decisive signal (a candidate line must span ~85%+ of the full
  photographed FRAME, not just be long relative to other strokes) —
  verified this resolves the ambiguity cleanly: all 4 non-ruled-paper
  repro images now show zero false removal, the real ruled-paper image
  still gets cleaned correctly, and a 9-angle synthetic single-stroke
  sweep confirms a lone genuine flourish is never touched at any angle.
- **Full regression re-run**: all 5 repro images' verdicts and scores
  are unchanged from the pre-#47 numbers above — this fix only changes
  ruled-line behavior, verified, not anything else.
- **Honestly still open:** the user's own two specific new test images
  that produced the reported FORGED cases were not available as raw
  files this round (only PDF reports + UI screenshots) — enough to
  diagnose and fix the underlying mechanism with high confidence, not
  enough to re-run the EXACT reported cases and confirm the verdict
  itself now comes out right. Top item for the user to check next.

### v7.2 status (2026-09-01) — user's second real round: the Bug #45 fix was itself platform-broken (found and fixed), and real cross-signer evidence drove a moderate, evidence-checked fusion reweight

- **Trigger:** the user ran the v7.1 delivery for real again. The LLM
  crash changed shape (progress, not regression — Bug #45's actual fix
  held; a NEW, different bug in the same area was exposed) to a
  Windows-specific file-URI construction error, and — critically — the
  user's own real engine measurements were captured verbatim in the
  (still-failing) LLM prompt logs, giving this round something the
  previous ones didn't: exact, real component-similarity numbers for
  a confirmed-genuine comparison on a second real signer.
- **Bug #48 (Windows file-URI bug) — fixed, verified as far as
  possible without Windows itself.** Root cause: Bug #45's fix passed a
  path string, correctly, but built the `file://` URL with naive string
  concatenation, which is not a valid URI on Windows (backslashes,
  drive letter). Fixed with `pathlib.Path.as_uri()`. Verified using
  `pathlib.PureWindowsPath` (the stdlib's own OS-independent
  Windows-path test class) to confirm the correct URI is produced —
  real verification of the logic, still not a substitute for an actual
  Windows run, which remains the top item to confirm.
- **Bug #49 (hu/scalar_profile cross-signer weakness) — real, evidence-
  based, deliberately moderate fix.** The user's own real numbers
  (ssim=.97, iou=.91, dtw=.97, construction=.91, zonal=.92, all
  excellent, vs. hu=.21 and scalar_profile=.53 dragging the fused score
  down to an edge-case-downgraded INCONCLUSIVE) matched a pattern this
  project had already seen once in its own self-testing — a real,
  repeated signal, not noise. Fixed with two complementary, separately-
  verified levers: a MODERATE fusion reweight (a larger draft was
  tested and rejected after it was found to hurt a different real
  signer's data — this project's own repro set — where `construction`
  is comparatively weak; picking a global winner component that suits
  only the most-recently-reported signer is exactly the overfitting
  this project's rules exist to prevent) and a moderate increase to the
  natural-variation tolerance prior (a larger increase was tested and
  rejected after it measurably eroded margin against a known real
  forgery). Both final values were chosen by testing against BOTH real
  datasets available (this project's own 5 images AND the user's own
  logged numbers), not either one alone.
- **Full regression re-run**: all 5 repro images still verdict
  correctly (forgery stays FORGED with a preserved margin, self-checks
  stay GENUINE, the repeated-recheck near-duplicate guard and bounded
  threshold from Bugs #41/#42 still hold) — confirmed after these
  changes, not assumed.
- **Explicit, load-bearing honesty about what this round did NOT
  solve:** a single global fusion-weight vector cannot fully reconcile
  two real signers whose signatures apparently vary differently across
  feature families. This round's fix is real, verified improvement
  within that limit — the actual architectural answer (per-profile
  adaptive weighting) is now a concrete, well-specified roadmap item
  (Section 5), not yet built.

### v8 status (2026-09-02) — user provided real saved case data for the first time; found and fixed this project's single most consequential preprocessing bug

- **Trigger:** the user attached `data.zip` — an actual export of their
  real `data/` directory (profile.json, enrolled reference images, full
  case.json + saved intermediate images for real verification attempts).
  This is the first round with direct access to the actual bytes the
  engine processed at every stage, not a summary or a log excerpt.
- **Bug #50 (global-Otsu lighting-gradient failure) — the headline
  finding.** Rendering a real enrolled reference photo's binary mask
  directly showed 46% of the entire 1200x1600 frame misclassified as
  "ink" — a lighting gradient/vignette, not the actual small signature —
  which corrupted the crop bounding box (computed as nearly the whole
  page) and everything measured downstream. Fixed with standard
  illumination flattening (large-kernel background estimation + divide)
  before Otsu. Verified directly: ink-pixel count on the exact reported
  image collapsed from 886,129 to 10,298, visually containing ONLY the
  real signature. This is very plausibly the single largest contributor
  to every "can't recognize a new signature type" report across this
  project's whole history — it affects any unevenly-lit phone photo, not
  one signer or one case.
- **Bug #51 (unanimous-verdict override) — found immediately after,
  using the SAME real case.** Even with Bug #50's score fix, a clean,
  unanimous FORGED verdict (zero disagreement across references) was
  still downgraded to INCONCLUSIVE by the alignment-reliability edge
  case, which never checked whether the references actually agreed.
  Fixed: poor alignment now only downgrades the verdict when it's paired
  with genuine disagreement, not when every reference already agrees.
- **Bug #52 (CUDA pre-flight check) — implemented as directly requested.**
  `install.py` now checks the driver's actual CUDA version and (on
  Windows) VC++ Redistributable presence BEFORE attempting the GPU
  wheel, closing the exact gap behind this project's own previously-
  recorded real failure (Bug #25).
- **Bug #53 (model swap) — investigated before implementing, not just
  executed literally.** The user's specifically-requested model turned
  out to be a non-standalone draft/speculative-decoding model, and its
  real full-size counterpart would have made the user's actual VRAM
  problem worse. Switched to `google/gemma-4-E2B-it-qat-q4_0-gguf`
  instead — same family, half the size, actually fits the user's
  hardware.
- **Bug #54 (per-profile adaptive weighting) — the previously-flagged
  roadmap item, now built.** Each profile now estimates its own
  per-component reliability from its enrolled references' mutual
  consistency, blending toward the global weight vector as more
  references accumulate. Verified mechanically (weights sum to 1,
  correctly blend-in with reference count); not yet validated against a
  real multi-signer benchmark.
- **Full regression re-run against everything available**: this
  project's own 5-image set (all verdicts correct, image 3 now a clean
  GENUINE), the real Saurabh cross-signer mismatch (now correctly
  FORGED, was individually-"GENUINE" before this round), and the real
  vinit-patil genuine case (still correctly GENUINE) — all confirmed
  after every change in this round, not assumed from reasoning alone.
- **What's still unverified, honestly:** the model swap and CUDA
  pre-flight check cannot be tested end-to-end without the user's own
  Windows+GPU hardware (same sandbox limitation as every prior LLM-
  related fix). The adaptive-weighting mechanism is verified to run
  correctly, not verified to actually improve real-world accuracy at
  scale — that needs the same multi-signer benchmark this file has
  flagged as the top remaining need since Bug #17.

### v2 status (2026-07-23) — real user bug reports fixed, dataset calibration added

- **Trigger:** user ran v1 on real Windows hardware with real signature photos and hit real
  problems: LLM install failure (Bug #10), LLM status stuck on "error" (Bug #9), and — the
  serious one — **genuine, enrolled signatures classified as FORGED** (Bugs #6+#7+#8), plus a
  PDF report layout bug (Bug #11) and a missing click-to-expand metric detail feature.
- **Root-caused and fixed all of them** — see Section 3, Bugs #6-#11. The false-FORGED bug was
  three compounding issues (z-score blowup, mismatched calibration formula, double
  preprocessing), not one — each was isolated with a targeted reproduction test before fixing,
  per process rule #7.
- **Verified fix on the user's own real uploaded images** (not synthetic data): before the
  fixes, the user's own enrolled signature scored FORGED; after, it correctly scores GENUINE
  (0.90 vs 0.71 threshold), while a differently-styled signature and an independent CEDAR
  forgery sample both still correctly score FORGED — confirming the fix resolved the
  false-positive bias without breaking real forgery detection.
- **Added `training/calibrate_weights.py`**: auto-extracts a user-supplied
  `kaggle-dataset-sign.zip` into `Sign Dataset/`, builds genuine/forged comparison pairs, and
  fits the engine's 7 fusion weights via logistic regression (see Section 1's refined
  no-pretraining invariant, and `docs/TRAINING.md`). **Validated end-to-end** during
  development on a 15-writer CEDAR-shaped subset (same NNN/NNN_forg folder pattern as the
  user's described dataset): 240 pairs, held-out AUC 0.813, accuracy 0.75. The resulting
  `data/population_weights.json` shipped in this delivery is FROM THAT VALIDATION RUN, not
  the user's actual Kaggle dataset (which wasn't available in the build sandbox) — re-run the
  script with the real dataset zip for a proper calibration; the file will be overwritten and
  auto-picked-up on next run.
- **Answered open questions** (multi-language signatures, LLM scope/boundary) — see Section 7
  and `docs/ARCHITECTURE.md` §7 for the full reasoning.
- **Not yet re-verified:** the Windows `install.py --llm` fix (Bug #10) — needs the user to
  re-run it on their actual Windows machine, since this sandbox has no Windows target to test
  against directly.

### v3 status (2026-07-24) — the core reported bug fixed, verified with a real sweep

- **Trigger:** user reported their own genuine signature still scoring INVALID after v2,
  correctly self-diagnosing "system make it scoring threshold high." Also flagged: distrust of
  the v2 dataset-training approach (security concern, not technical), Windows launch confusion,
  GPU not being used, reports too short, and confirmed (via a parallel debugging session with a
  different AI agent) a model-tier/installed-model mismatch bug.
- **Root cause of the core bug, found and fixed (Bug #13):** with only 2 enrolled references,
  the threshold had a mathematically-guaranteed ZERO slack (std of 1 pairwise comparison is
  always exactly 0). Fixed with a shrinkage estimator toward a hand-set, non-dataset-trained
  prior. Also fixed a related gap (Bug #14): scores just above threshold had no "uncertain"
  buffer, only scores below did.
- **Honest sweep result** (10 CEDAR writers, 2 references enrolled each, tested against a
  held-out genuine sample + a forgery sample per writer — 20 verifications, none of the test
  images were enrolled):
  ```
  genuine_correct:    6/10   (0 wrong — the core bug: ZERO genuine signatures
                               misclassified as forged, across the whole sweep)
  forged_correct:     3/10
  forged_wrong:       2/10   (forgeries that slipped through as GENUINE)
  inconclusive:       9/20   (routed to human review — the intentional safety
                               valve for a threshold now honestly acknowledging
                               2-reference uncertainty)
  ```
  Read honestly: this is a **conservative** system with only 2 references — it correctly never
  falsely rejects a genuine signature in this sweep, at the cost of nearly half of all
  verifications (9/20) being flagged for human review rather than auto-decided, and 2/10
  forgeries slipping through as confidently genuine. This is the real, inherent trade-off of
  fixing zero-slack-threshold with only 2 samples — not fully resolved, honestly bounded. The
  single most effective lever the user has to improve this further is enrolling 3-5 references
  instead of 2 (Bug #13's shrinkage fades toward the observed, tighter std once ~4 references
  are enrolled).
- **Not yet independently benchmarked:** report page count (3, per a real PDF page count check)
  and GPU offload code paths (correct by inspection and consistent with `n_gpu_layers=-1` logic
  already present, but not run against an actual NVIDIA GPU in this sandbox — no GPU here).
  **Top item for the user to verify**, alongside the Windows install fix from v2.

### v4 status (2026-07-25/26/27) — LLM install genuinely fixed (3rd attempt), real SHAP added, CNN-SigNet written (untested)

- **Trigger:** user's real Windows log showed the LLM install STILL failing
  (every wheel index falling through to source build) despite two previous
  "fixes," plus the web UI trying to download the wrong model tier (31B).
  Also: a real INCONCLUSIVE report the user didn't understand, and three
  explicit core requirements never integrated — CNN-SigNet, Grad-CAM, SHAP.
- **LLM install (Bug #18) — root-caused properly this time, not assumed:**
  fetched the actual wheel index pages directly and found the previously
  "fixed" version pin (0.3.16) was never published on any index at all —
  the v2 fix was never empirically verified, just assumed reasonable. Also
  discovered official CUDA wheels are Linux-only for any current version —
  no Windows CUDA wheel exists, so every prior Windows GPU attempt was
  structurally doomed regardless of version. Then found that even a
  verified-correct version+index combination didn't reliably resolve via
  pip's `--extra-index-url` mechanism — switched to direct GitHub Releases
  wheel URLs instead, bypassing index resolution ambiguity entirely.
  **Verified end-to-end in a clean-room test**: real 23MB prebuilt wheel
  downloaded and installed, `import llama_cpp` succeeds, manager
  recognizes it correctly. Windows path uses the same mechanism with a
  different filename — logically identical, not independently testable on
  actual Windows hardware here.
  **[v5 correction, Bug #25]: this "verified end-to-end" clean-room test
  and the direct-GitHub-Releases-URL approach it verified were both wrong
  in a way not caught until the next round — the URLs assumed a universal
  `py3-none-<platform>` wheel filename, but llama-cpp-python's real wheels
  are tagged per-Python-version (`cp310`/`cp311`/`cp312`/...) since it
  ships compiled C extensions. Whatever that clean-room test actually hit,
  it wasn't representative of a real install on a machine with a
  different Python version. Superseded by the officially-documented
  `--extra-index-url` package-index approach — see Bug #25.**
- **Single-model simplification (explicit user instruction):** removed the
  entire multi-tier model system. Exactly one model
  (`google/gemma-4-E4B-it-qat-q4_0-gguf`) is ever downloaded, by anyone, on
  any hardware that qualifies. This eliminates the wrong-tier-downloaded
  bug class by construction, not by patching around it.
- **INCONCLUSIVE investigation:** researched signature-verification
  ensemble literature, found real support for majority-voting over linear
  averaging, implemented a bounded "consensus penalty," tested it honestly
  against the same 10-writer CEDAR sweep used for v3's baseline — **no
  improvement, reverted** (see the "Tested and rejected" entry above
  Bug #19). The user's specific report case was re-examined and found to
  be the system behaving as designed (score 0.017 above threshold,
  comfortably inside the intentional ±0.07 uncertainty band) — not a bug,
  though the underlying tension (few references -> wide uncertainty band)
  is real and already documented, with "enroll more references" remaining
  the most effective lever.
- **Real KernelSHAP added and fully verified**: genuine Shapley-value
  attribution over the ~22 classical scalar features (not a heuristic
  approximation) — verified via the efficiency property (sum of Shapley
  values equals full-minus-baseline score, checked to floating-point
  precision in testing) and wired into the live verify() endpoint, web UI,
  and PDF/HTML reports. Fast enough for a synchronous request
  (~0.02-2s depending on load). This is fully tested, working code.
- **CNN-SigNet + Grad-CAM written, NOT independently tested end-to-end**:
  a real Siamese CNN + genuine gradient-based Grad-CAM was implemented
  (`app/core/signet.py`), trained per-profile from only that profile's own
  augmented enrolled samples (no external data, respecting the project's
  hard invariant). `pip install torch` could not be verified working in
  this development sandbox: the default PyPI Linux build either exhausted
  available disk space (bundles several GB of CUDA libraries as required
  dependencies) or, installed `--no-deps` to avoid that, failed to even
  **import** with `ImportError: libcudnn.so.9: cannot open shared object
  file` — meaning recent PyPI torch builds are hard-linked against CUDA
  at the C-extension level, not just at GPU-use time. This is concrete,
  demonstrated evidence (not speculation) that this dependency carries
  real install risk, consistent with this project's whole history of
  fighting exactly this class of problem with `llama-cpp-python`. The
  code is written carefully using standard patterns and syntax-checked,
  and the **fallback path (torch absent) was fully verified** — the app
  works normally, gracefully, with `signet.available() == False`, adding
  no risk to the tested core pipeline. **The torch-enabled path itself is
  unverified and should be treated with correspondingly less confidence
  than the rest of this codebase** until run on a machine where torch
  actually imports (recommended: `pip install torch --index-url
  https://download.pytorch.org/whl/cpu` for a genuinely CPU-only build,
  per `requirements-cnn.txt`'s notes).
- **Top items for the user to verify:** the Windows LLM install fix
  (now attempted 3 times — this one has the most rigorous verification
  behind it, but only real Windows hardware can confirm it), GPU offload
  on real NVIDIA hardware, and CNN-SigNet's actual training/Grad-CAM
  behavior once torch is installed on a machine where it actually imports.

### v5 status (2026-08-18) — multi-reference recognition engine, alignment rewrite, LLM auto-detection, real TTC

- **Trigger:** four explicit, direct user reports — (1) the recognition
  engine only using one random-feeling reference and flipping
  GENUINE/FORGED/INCONCLUSIVE inconsistently, with visibly rotated,
  non-overlapping heatmap/overlay images; (2) a manually-copied LLM model
  not being detected/loaded; (3) feedback being an alert popup with no
  real training signal; (4) a general request to review against the
  signature-verification research literature and fix what's actually
  wrong, not just what was asked literally.
- **Fixed, with real before/after verification (see Bugs #20-#23 above
  for full detail):**
  - Alignment (`align.py`) rewritten as a scored multi-candidate chooser
    (identity / centroid / PCA-moments / ORB+ECC, rotation-clipped
    ±25°) instead of trusting ORB unconditionally — verified a
    synthetically-rotated (12°) genuine CEDAR image's ink IoU against its
    reference improved from 0.085 (raw) to 0.167 (aligned), with a visual
    overlay check confirming correct overlap.
  - Recognition (`engine.py::verify()`) now scores against EVERY enrolled
    reference (previously computed but then discarded down to just the
    best one), fuses a ProtoSig-style aggregate prototype score in with
    the best individual match, and runs an independent edge-case
    re-evaluation pass over the distribution of individual comparisons
    before trusting the fused verdict. Full per-reference transparency
    (`case["reference_scores"]`) and a new visual asset
    (`agreement_chart`) expose exactly which samples the questioned
    signature was compared against.
  - LLM detection (`device.py`, `manager.py`, `main.py`) — added a real
    on-disk size check (`model_present_on_disk`, ≥4.5GB), a `"downloaded"`
    status, and a FastAPI startup hook (`autostart()`) that checks
    model-path+size and GPU availability and loads automatically in the
    background — the exact "[1] model path & size [2] GPU available"
    check the user specified. Root cause was a genuine status-gate
    deadlock in `report.py` (see Bug #22) — not a detection heuristic
    problem, a logic bug.
  - Real TTC (`engine.py::feedback()`, opt-in `add_as_reference`) —
    confirmed-GENUINE feedback can now promote the query into the profile
    as a new reference and relearn all profile statistics, capped and
    gated behind explicit examiner confirmation. Visible, persistent
    training panel replaces the old alert-only popup.
- **Honest sweep result, 10 CEDAR writers, 5 references enrolled each,
  tested against 1 held-out genuine + 1 skilled forgery per writer (20
  verifications total, none of the test images enrolled), AFTER the
  dev-time edge-case retune documented in Bug #21:**
  ```
  genuine_correct:        5/10   (0 wrong — genuine signatures are NEVER
                                   misclassified as FORGED in this sweep)
  genuine_inconclusive:   5/10   (routed to human review)
  forged_correct:         0/10   (see honest limitation below)
  forged_wrong:           3/10   (skilled forgeries that scored high
                                   enough to be called GENUINE)
  forged_inconclusive:    7/10   (routed to human review — the safe
                                   outcome when confidence is genuinely low)
  ```
  Compare to v3's 2-reference sweep (genuine_correct 6/10, forged_correct
  3/10, forged_wrong 2/10, inconclusive 9/20 total): v5 trades a modest
  increase in forged_wrong (2->3, N=10 each so this is a ±1-sample-sized
  difference, not a statistically solid trend) for a real, mechanism-level
  improvement in genuine reliability (still 0 false-FORGED, but now more
  genuine cases resolve confidently as GENUINE instead of parking at
  INCONCLUSIVE — a direct benefit of the prototype-averaging design) and,
  more importantly, transparent, per-reference-inspectable reasoning
  behind every verdict instead of a single hidden "best match."
- **Honest, NOT-glossed-over limitation:** skilled-forgery detection is
  still the weak point, and this is an inherent limitation of classical/
  handcrafted features (HOG/LBP/Hu/SSIM/DTW), not a bug to hunt down
  further within this architecture — see Bug #21's closing note and
  Section 7 for the real next step (the already-written but untested
  CNN-SigNet/Grad-CAM path, `signet.py`, is the literature-supported
  answer to exactly this gap; it needs `torch` actually importing on a
  real machine to evaluate, which this sandbox cannot do — see v4's own
  note on this).
- **Not yet re-verified on real hardware:** the LLM autostart flow was
  verified with a synthetic fake-sized model file and a monkeypatched
  device profile (no real 5GB+ model or GPU exists in this build
  sandbox) — **this is the top item for the user to verify**: copy the
  real model into `data/models/google--gemma-4-E4B-it-qat-q4_0-gguf/`,
  start the server, and confirm the LLM status panel shows `"downloaded"`
  then transitions to `"ready"` within a few seconds without any manual
  click.

### v6 status (2026-08-24) — recognition robustness, real heatmap, Windows LLM build fix, Browserbase restyle

- **Trigger:** direct user follow-up after using v5 for real: (1) the
  recognition engine still mis-judged signatures based on photo framing/
  position, confirmed by the user as a real, reproducible pattern; (2)
  the LLM install worked (detection fixed) but the actual model load
  failed on a real Windows machine, with a full terminal log attached;
  (3) a generated PDF report showed no visible heatmap and colliding
  rectangle labels; (4) UI feedback with screenshots (missing thumbnail
  remove control) plus a request to restyle the web app against a
  provided design reference (Browserbase) and add a real remark/submit
  flow to feedback.
- **Fixed, with real before/after verification (see Bugs #26-#29 above
  for full detail):**
  - Preprocessing (`preprocess.py`) now uses a density-based connected-
    component outlier filter plus a percentile-trimmed crop bounding box
    instead of naive min/max -- confirmed to eliminate >95% of a
    synthetic stray-mark's distortion of aspect ratio while losing zero
    real ink on clean signatures, and improved the 10-writer sweep's
    false-forgery-accept rate from 3/10 to 0/10 as a side effect.
  - The divergence heatmap (`explain.py::heatmap_overlay`) is now
    intensity-modulated (alpha scales with actual divergence magnitude)
    instead of a flat-opacity blend, fixing the "solid blue wash, no
    heatmap visible" report. The annotated-regions "rectangle detector"
    now finds real divergence blobs via contour detection instead of a
    rigid grid, with non-max suppression and full label collision
    avoidance (reusing the existing robust labeler instead of a second,
    weaker one) -- fixing the colliding, generic-looking boxes shown in
    the user's own generated PDF.
  - LLM install (`install.py`) now fails fast on a wheel-only attempt
    instead of silently substituting a source-only newer release (the
    actual reason the user's install fell through to a doomed compile),
    retries the widely-available 'cpu' backend before ever requiring a
    compiler, and automatically locates + loads a Visual Studio C++
    build environment via `vswhere`/`vcvars64.bat` when a from-source
    build genuinely is needed -- directly fixing the "nmake ... no such
    file" error from the user's real log, which was an environment-
    loading problem, not a missing-Build-Tools problem.
  - The web UI (`web/`) was restyled wholesale against the user-provided
    Browserbase design reference, restructured into three clearly
    labeled result sections (engine analysis / high-level visualization
    / local AI explanation), gained a real remove control on enrollment
    thumbnails, and gained an explicit remark-plus-submit feedback flow
    with the remark actually persisted server-side.
- **Honest, not-glossed-over limitations carried into this version:**
  the MSVC auto-detection and the visual restyle could only be verified
  as far as this sandbox allows (no real Windows Build Tools install, no
  reachable headless browser) -- see Section 7's "New in v6" entries for
  exactly what still needs the user's own hands-on confirmation.

---

## 5. Roadmap toward the actual goal

- **Phase 1 (DONE, v1):** Make it runnable, cross-platform, one-command-installable. Fix the
  LLM backend mismatch. Wire a minimal real web UI + terminal UI. This memory file.
- **Phase 2a (DONE, v2):** Fixed the false-FORGED bug's first three root causes (Bugs #6-#8),
  dataset calibration pipeline built and validated (later de-prioritized per user request, v3).
- **Phase 2b (DONE, v3):** Fixed the REMAINING root cause of the false-FORGED-on-genuine bug —
  small-sample threshold statistics (Bug #13) — verified with an honest 10-writer sweep (see
  Section 4). Multi-GPU handling, report depth, Windows launch friction, EXIF orientation.
- **Phase 2c (NEXT):** A full CEDAR benchmark (not just a 10-writer/20-verification sweep) to
  get statistically solid accuracy numbers, and — given the v3 sweep showed 2-reference profiles
  routing ~45% of verifications to INCONCLUSIVE — specifically benchmark accuracy AS A FUNCTION
  OF reference count (2 vs 3 vs 5) to give the user (and the UI) a concrete, evidence-based
  recommendation like "3 references cuts inconclusive rate roughly in half" rather than a vague
  "more is better." Add automated regression tests so future changes can't silently regress
  accuracy (CI/CD requirement from the user's original brief, not yet implemented).
- **Phase 2d (NEW, v7.2, high priority):** per-profile ADAPTIVE fusion weighting. Bug #49
  found real evidence that different real signers' signatures vary more or less across
  different feature families (`construction` reliable for one real signer, weak for another;
  `scalar_profile` the reverse) — a single global weight vector, however carefully tuned,
  structurally cannot fit every signer well. Concrete proposal: at `_relearn` time, alongside
  the existing per-feature natural-variation std (Bug #34), also track each of the 9 fusion
  COMPONENTS' own agreement across the enrolled references' pairwise comparisons (i.e. how much
  does `construction` vary between THIS signer's own reference pairs, vs. how much does `hu`
  vary) — a component that stays consistently high across a signer's own genuine pairs is
  evidently a reliable signal FOR THAT SIGNER and should be trusted more; one that swings a lot
  even between that signer's own confirmed-genuine samples is evidently noisy for them and
  should be trusted less. This replaces guessing a single global vector with letting each
  profile's own enrolled evidence answer the question directly. Needs at least 3 enrolled
  references to estimate meaningfully (the same practical minimum this project already
  recommends for other reasons) and needs real multi-signer data to validate the approach
  itself before trusting it in production — natural fit for the same Phase 2c benchmark.
- **Phase 3:** Expand web UI to apply the 5 provided style references at "certain levels" of
  the app (user's phrase) — e.g., forensic-lab dashboard could use the Claude parchment/serif
  system for the report/document view (editorial, printed-report feel fits a forensic report),
  while the landing/marketing page could use a different reference. **This needs a decision
  from the user on which reference goes where** — see Open Questions, Section 7. Ships with
  ONE coherent style (Claude parchment/serif) applied throughout, not a mix, so the app looks
  finished rather than stitched-together; expanding to multiple is Phase 3, not a defect.
- **Phase 4:** Package as installable desktop apps (not just "python main.py") if the user
  wants a double-clickable app — currently out of scope, flag as open question.
- **Phase 5 (explicitly deferred by user to end of v3 work):** "System Working Logic" document
  — comprehensive roadmap/algorithm/per-file documentation with a visual diagram (PNG or HTML),
  for presentation purposes. Requested but explicitly ordered last, after all other v3 items.

---

## 6. Process rules for this project

1. **Decision layer never crosses into the explanation layer and vice versa.** (Section 1.)
2. **No bash/.bat-only scripts.** Every setup/launch action is a cross-platform Python entry
   point (`install.py`, `main.py`, `cli.py`).
3. **No claim of accuracy/capability in docs, UI copy, or code comments without a citation**
   to either Section 2 (verified facts) or an actual test run recorded in Section 4. This
   mirrors the standard already set earlier in this conversation (ProtoSig honest numbers,
   VLM rationalization-trap evidence) — don't regress to marketing language.
4. **Every bug gets appended to Section 3, never silently fixed and forgotten.** If you fix
   something and don't write it here, the next person (human or LLM) re-discovers it the hard
   way and burns tokens on it.
5. **Heavy/optional deps (torch, llama-cpp-python) stay import-lazy inside `app/llm/`.** Core
   app must always run with only `requirements.txt`.
6. **Update this file at the end of every phase**, before presenting the zip. Append to
   Section 3/4/5, don't rewrite history.
7. **Before shipping any version, run at least one real enroll→verify round-trip with real
   image data**, not just import/syntax checks. File-by-file code review does not catch
   cross-file contract mismatches (see Bug #5) — only an actual execution does.
8. **Before any edit, ask: what does this touch, and what could break as a result? Then verify
   that specific thing afterward** — a tool call reporting success and the change being correct
   are not the same fact. Concrete example from this session: an edit meant to append Bugs
   #6-#12 to Section 3 instead silently replaced Bug #5's text — the edit tool reported success,
   the string replacement was mechanically valid, but the *content* was wrong (deleted history
   this file's own rule #6 says must never be deleted). It was only caught by re-reading the
   file's section list immediately afterward, not by the edit succeeding. Applies to code too:
   Bug #7 (mismatched calibration formula) was exactly this class of risk — a change in one
   function silently invalidating an assumption in another, undetectable without re-running the
   actual enroll→verify flow. Rule of thumb: the riskier or more shared the thing being edited
   (this file, a calibration formula, anything more than one file depends on), the more
   necessary a re-read or re-test immediately after, not just before packaging a delivery.
9. **Never pin a dependency version without empirically checking it exists where it needs to
   exist** (a wheel index, a specific platform tag). Bug #10 pinned `llama-cpp-python==0.3.16`
   reasoning it was "known to work" — never actually checked. Bug #18 found it wasn't published
   on any wheel index at all, on any platform. Same mistake, twice, in the same dependency. If a
   pin matters for install reliability, verify it against the real index/release page before
   writing the number down — "should be right" is not verification.
10. **After ANY change to shared scoring/threshold logic (similarity.py, engine.py's
    `_score_against_ref`/`_relearn`/`_edge_case_review`, align.py), re-run the FULL repro image
    set — every enrolled reference, every held-out genuine, AND the known forged sample — not
    just the one case the change was meant to fix.** v7's Bug #36 is the concrete example this
    rule exists to prevent: a first fix attempt correctly resolved the reported self-check
    penalty issue, but silently flipped a REAL FORGERY (image 4) from correctly-FORGED to
    incorrectly-INCONCLUSIVE — a regression that would have shipped if only the originally-
    reported case had been re-checked. A fix that improves one case while quietly breaking
    another is not a fix; it's a different bug wearing the first one's clothes.

---

## 7. Open questions (unresolved — revisit as evidence comes in)

### New in v7.1 (unresolved)

- **The two specific test images that produced FORGED verdicts on what
  should include an enrolled self-check were not available as raw
  files** — only two PDF reports (case IDs `510225173871` and
  `7fb6bee95229`) and two UI screenshots. Bug #47's fix was diagnosed
  and verified against this project's own repro set plus targeted
  synthetic tests, which gives high confidence in the MECHANISM, but
  the exact reported cases haven't been re-run against the fix. **Top
  item for the user**: re-enroll the same 2 references and re-verify
  the same 2 query images that produced those case IDs, and confirm
  the self-check in particular now scores correctly.
- **Bug #45/#46 (LLM crash fix and status transparency) were verified
  with mocked components** (a fake `generate()` for #45, direct field
  inspection for #46), not a real end-to-end run with the actual GGUF
  model producing a real forensic narrative. The user's next real run
  (now that the crash itself is fixed) is the first real test of
  whether the LLM's actual JSON output validates cleanly against the
  `Examination` schema on the first or second attempt, not just that
  the plumbing to reach that point no longer crashes.
- **The "twisted"-looking evidence visualizations** (heavily rotated
  images shown without cropping the corners, per the user's
  screenshots) were not changed in this round — Bug #47's fix addresses
  the ACCURACY regression (ruled-line removal wrongly stripping genuine
  ink), not the VISUAL presentation of a legitimately-rotated alignment
  result. If large rotations are still visually confusing after Bug #47
  is confirmed fixed, that's a separate, real UX polish item (e.g.
  padding/cropping the warped image's corners before display, or
  capping how much rotation the evidence view itself shows even when
  the underlying alignment used more) — not yet investigated.

### New in v7 (unresolved)

- **Bug #43/#44's LLM-hang fixes (timeout, VRAM gating, progress
  heartbeat) could not be tested against a real GPU, a real ~5GB GGUF
  download, or genuinely tight RAM** — this sandbox has none of the
  three, and no network access to Hugging Face. The timeout/reconciliation
  mechanism itself was verified with a mocked `llama_cpp.Llama` that
  sleeps past the configured timeout (see Bug #43's writeup) — that
  proves the Python-level control flow is correct, but NOT that it
  fixes the user's actual reported hang, which could in principle have
  a different root cause than the two identified (VRAM headroom,
  n_ctx-driven KV-cache size) if those turn out not to be it. **Top
  item for the user to verify**: does a real load on their machine now
  show the "still loading... Xs elapsed" heartbeat message, and does a
  load that would have hung now report a clear timeout error within
  `LLM_LOAD_TIMEOUT_SECONDS` (default 180s) instead of sitting at
  "loading" indefinitely? If it STILL hangs past the timeout with no
  error surfaced, that would mean the hang is inside something the
  background-thread wrapper can't observe (e.g. a hang during Python's
  own `import llama_cpp` before `_load()`'s try block even starts) and
  needs a different fix.
- **Bug #40's ruled-line removal is a real, verified improvement
  (visually confirmed to remove page-rule lines from the binary mask on
  the actual reported image) but did not change that image's final
  verdict** (still FORGED, was FORGED before). The user's own
  description of what the "correct" verdict should be for this
  specific image was ambiguous in the request that prompted this fix
  (it reads as either "should be INCONCLUSIVE" or "correctly resolves
  despite the lines" depending on interpretation) — no further tuning
  was done chasing this one ambiguous case. If the true ground truth
  for this specific image is GENUINE or INCONCLUSIVE rather than
  FORGED, that's a concrete, well-defined next data point to bring back
  with a clear statement of which it should be.
- **Every new numeric constant introduced in this pass**
  (`_IOU_TOLERANCE_PX=3`, `_FEATURE_PRIOR_REL=0.18`,
  `_MIN_RELIABLE_IOU=0.30`, `_MIN_SANE_IOU=0.08`,
  `_NEAR_DUPLICATE_SCORE=0.97`, `_LINE_ANGLE_TOL_DEG=8.0`,
  `_LINE_LENGTH_FRAC=0.70`, the `±0.10` feedback-threshold band) is
  defended by a mechanism-level reason plus a check against the one
  real forged sample available (image 4) — but all of it comes from a
  5-image, 1-signer repro set. This is real progress over "untested
  guess," but it is not the same thing as calibration against a proper
  multi-signer dataset (CEDAR or otherwise) — see Section 5's Phase 2,
  still not done, still the single highest-leverage next step for
  trusting these numbers beyond this one person's 5 photos.
- **The Hu-moment instability finding (Bug #35) was itself somewhat
  surprising** and is worth the user's awareness even though it's now
  handled: moments 5/6/7 flipping sign between the user's own genuine
  samples suggests real, non-trivial photo-to-photo variation in how
  this signature is being captured (angle, lighting, crop) — enrolling
  references under more consistent photography conditions (same
  distance, angle, lighting, ideally a flatbed scan rather than a phone
  photo if at all possible) would likely improve every downstream
  metric in this system, not just this one, and is a cheap thing for
  the user to try independent of any further code changes.

### New in v6 (unresolved)

- **Bug #32's GPU-offload confirmation logic could not be tested against
  a real GPU** -- this sandbox has none. The log-parsing regex and the
  fd-capture mechanism were verified in isolation with synthetic inputs;
  a real end-to-end "load on real CUDA hardware, confirm the status page
  agrees" run has not happened. Top item for the user to verify with
  their actual GPU.
- **Bug #33's rebalanced weights were tuned against ONE real forged/
  genuine image set (4 photos) plus the same 10-writer CEDAR sweep used
  throughout this file** -- real, honest evidence, but still a small
  sample. The new false-accept found on writer 5 (a forgery that
  replicated construction counts well) is left undisturbed deliberately
  rather than chasing one case with another weight tweak that would
  likely just move the failure elsewhere -- but if the user has more
  real forged/genuine pairs to test against, that would be far more
  informative than further tuning against this same small set.
- **The morphological-closing fix for topology features (Bug #33) uses a
  fixed 3x3 kernel.** This was sized to bridge lighting-noise-scale gaps
  without touching real structural differences, and tested against the
  4 real user photos plus CEDAR, but has not been tested against photos
  with notably different resolution/DPI where a fixed pixel-size kernel
  might be too small or too large relative to stroke width.

- **Bug #30's fix (LLM DLL-load verification + CPU fallback) could not be
  tested against the user's own real failure** -- this sandbox has no
  GPU/CUDA runtime to reproduce "wheel installs successfully, DLL won't
  load" authentically. The individual pieces were verified as far as
  possible; the user re-running `python install.py --llm` on their real
  machine is the only way to confirm the automatic fallback actually
  resolves it there.
- **Bug #31's zonal analysis is a genuinely new, still-maturing signal,
  not a mature one on the level of the existing global metrics.**
  Individual per-zone flags are still noisier than desired at real
  handwriting's natural fine-scale variation even after tuning (an
  absolute + relative combined threshold reduced a 10-writer sweep's
  false positives from 58/60 to 0/60, but this was tuned against ONE
  sweep, not a large validation set) -- this is exactly why flagged zones
  are explanatory findings for human review, not an automatic verdict
  override. The aggregate zonal score's fusion weight (0.12) is a
  reasoned starting point, not empirically optimized against a large
  benchmark. If the user has real multi-script (CJK/Devanagari/etc.)
  signature samples, testing zoning.py against those directly would be
  far more informative than further synthetic tests -- this was designed
  to generalize by construction (content-adaptive, not fixed-grid or
  script-specific), but genuine cross-script validation hasn't happened.
- **The v6 preprocessing/zonal changes together produced a small
  honestly-reported regression** (genuine_correct 5/10 -> 4/10 in the
  standard 10-writer sweep, with both false-accept rates still at 0/10)
  -- worth re-examining if a larger validation set becomes available, to
  see whether this is real signal or sweep-to-sweep noise at this small
  sample size.

- **The MSVC auto-detection path for Windows LLM source builds (Bug #28)
  has not been run against a real Visual Studio Build Tools installation**
  -- only the "not found, fail gracefully" branch was exercised in this
  sandbox. Needs the user to confirm on their real Windows machine that
  it actually locates and loads a real install correctly.
- **The Browserbase web UI restyle (Bug #29) has not been visually
  screenshotted in a real browser** -- verified structurally (every CSS
  class/ID cross-referenced, JS syntax-checked) but not visually, since
  this sandbox has no reachable headless-browser download source. Needs
  the user to actually look at it.
- **Bug #26's fix (crop-boundary robustness) was tuned against synthetic
  stray-mark tests and the real CEDAR dataset, not against the user's own
  specific photos that originally triggered the complaint.** If a
  different kind of framing variance (not modeled by the two synthetic
  tests) still causes position-sensitivity, that's a real, distinct case
  to bring back with an example image, not evidence the fix didn't work
  at all.
- **The Bug #26 sweep improvement (forged_wrong 3/10 -> 0/10) is from the
  SAME 10-writer/1-forgery-sample sweep used throughout this file** -- a
  real, measured result, but still a small sample; treat it as directional
  evidence the preprocessing fix helped, not a validated production
  forgery-detection rate.

### New in v5 (unresolved)

- **Skilled-forgery detection with purely classical features is the
  system's real remaining weak point** (Bug #21's honest sweep: 0/10
  confidently correct, 3/10 misclassified as GENUINE). This is not a bug
  to patch with another threshold tweak — it's an architectural ceiling
  of handcrafted features (HOG/LBP/Hu/SSIM/DTW) against forgeries that are
  specifically skilled at mimicking overall shape. The literature-
  supported next step is finishing verification of the already-written
  `signet.py` CNN-SigNet + Grad-CAM path on a machine where `torch`
  actually imports (see v4's note on the sandbox's torch/CUDA install
  problems) — that needs the user's own machine, not further classical
  tuning here.
- **LLM autostart verified only with a synthetic fake-sized model file
  and a monkeypatched device profile** (no real 5GB+ GGUF or GPU exists
  in this build sandbox) — needs the user to confirm on their real
  machine that a manually-copied model is picked up automatically within
  a few seconds of server start, with no click required.
- **The 50/50 best-match/prototype fusion blend and the edge-case
  engine's opposing-evidence/dispersion thresholds are documented,
  reasoned constants (Bug #21), not fit on any dataset.** If the user
  runs a larger benchmark and finds a different blend works better, that
  should be revisited empirically rather than assumed correct because it
  "sounds reasonable."

- **`install.py`'s pip-repair path (Bug #24) was verified against a
  reproduced failure class on Linux, not the user's actual Windows
  machine.** The user should re-run `python install.py` on the machine
  that hit the original error and confirm it now either installs cleanly
  or prints one of the new actionable repair messages instead of the
  bare "No module named pip" traceback.
- **Bug #25's fix (the `--extra-index-url` package-index approach) was
  not verified to a fully successful install in this sandbox** -- the
  index domain itself is blocked by this specific development
  environment's network egress rules, and the from-source fallback
  compile was progressing correctly but was cut short by this
  environment's own process lifecycle, not by a failure. Top item for
  the user to verify for real: run `python install.py --llm` on an
  actual machine.
- **`run.bat`/`run_cli.bat` (Bug #24) were fixed by code review + the
  documented `%~dp0` behaviour, not run on a real Windows machine** (this
  sandbox is Linux-only) — `run.sh`/`run_cli.sh` (the bash equivalents)
  were verified directly. Low risk (`%~dp0` is a very old, stable batch
  feature) but flagged for honesty.

- Which of the 5 provided style references (monopo saigon, Caldera, Claude, ChatGPT, 21 TSI)
  should apply to which part of the app? Ships with Claude's parchment/serif system everywhere
  for coherence. Needs user decision for Phase 3.
- Desktop packaging (PyInstaller/Electron-style wrapper) vs. "run `python main.py`" — not
  specified by the user, currently assuming the latter is acceptable.
- What forgery dataset (if any) the user wants used for the Phase 2c full benchmark — CEDAR
  was used for the v3 10-writer sweep; reuse at larger scale or a different dataset?

### New in v3 (unresolved)

- **GPU offload not independently verified on real GPU hardware.** The user's machine has an
  NVIDIA RTX 4050 (6GB VRAM) — `device.py`/`manager.py`'s logic (`n_gpu_layers=-1` when CUDA
  detected) should offload to it correctly by inspection, and the multi-GPU selection logic
  (Bug #12) was tested with synthetic hardware profiles, not this exact real dual-GPU laptop.
  Needs the user to confirm `gpu_offload_active: true` actually appears in `/api/llm/status`
  once a model is loaded on their machine.
- **Whether 3 pages is "enough" detail** — a real PDF page count of 3 was verified in this
  session, matching the literal ask, but "in detail" is subjective; if the user wants more or
  less after seeing it, that's a quick prompt/template tuning pass, not a re-architecture.
- **Whether the user wants to revisit dataset-based calibration later** — currently
  de-prioritized for a stated security reason (see the v3 decision above), not removed. If
  trust in a specific dataset source changes, `training/calibrate_weights.py` still works.
- **EXIF orientation fix unverified against a real failing case** — no test image with EXIF
  rotation metadata was available in this session; the fix is defensively correct but its
  actual impact on the user's specific photos is unconfirmed.

### Answered in v3

- **"There needs to be a way to make each input signature a unique type of the same person...
  and based on all input signature data we should make a metric data for all input and compare
  & compute for input image that needs to validate is valid or not."** — This is exactly what
  `_relearn`'s mean/std profile already does (unchanged architecture from v1/v2): every
  enrolled sample contributes to a per-feature mean and std that captures this specific
  person's natural variation, and a new signature is compared against THAT distribution, not
  against a single fixed reference. What was broken (Bug #13) wasn't the concept — it was that
  with only 2 samples, the std estimate had a hard mathematical floor of zero, giving no room
  for the natural variation the user correctly identified as real and expected ("there may be a
  little difference a person can make in signature, and it is common"). The v3 fix directly
  implements that expectation: natural variation is now assumed to exist even before there's
  enough data to measure it precisely, via the shrinkage prior.

### Answered in v2

- **"How will the model learn/identify multi-language handwritten signatures (Hindi, Chinese,
  Japanese, Tamil, etc.) given it's only exposed to English-based training?"** — Answer: the
  classical engine (`features.py`) never does OCR or character/script recognition. It treats a
  signature purely as an ink shape: stroke curvature, topology (loops, junctions, endpoints),
  Hu moments (rotation/scale-invariant shape descriptors), ink density, aspect ratio. None of
  this depends on what script or language the signature represents — a Tamil signature and an
  English one are both just "an ink shape with N strokes, M loops, this aspect ratio" to the
  engine. This is architecturally different from (and more script-robust than) a deep embedding
  network trained mostly on Latin-script handwriting, which COULD encode subtle script bias.
  **Caveat, stated honestly:** this reasoning hasn't been empirically validated against real
  non-Latin signatures in this project yet — the v2 dataset calibration (`training/
  calibrate_weights.py`) was validated on a Latin-script (CEDAR) subset, and if the user's real
  Kaggle dataset is also Latin-only, the calibrated *fusion weights* (not the features
  themselves) are only proven correct for Latin-script signatures. The features are
  script-agnostic by design; the weights are only as validated as the data they were fit on.
  Recommend testing with a small multi-script sample set as a follow-up, not assuming.
- **"Define the scope of the system and how it works with the LLM — did the LLM cause the
  false-FORGED issue?"** — Answer, with direct evidence: **no, the LLM did not cause it.** The
  false-FORGED bug (Bugs #6-#8) occurred and was reproduced on machines where the LLM was NOT
  installed (`status: not_installed`, confirmed in the test logs) — the deterministic template
  path was the only one running. This is exactly the proof the two-brain separation
  (`docs/ARCHITECTURE.md` §2) is designed to provide: even with the LLM completely absent, a
  bug in the classical engine is still fully attributable to the classical engine, not
  "the AI got it wrong." The LLM's scope, unchanged from v1 and now empirically demonstrated to
  hold under a real bug: it only narrates metrics `engine.py` already computed, via a
  schema-constrained JSON contract; it never independently re-decides validity. See Bug #9 for
  the one LLM-side bug that DID exist (a status-reporting bug, not a decision-quality bug).

---

## 8. Related docs (read these too, for the subsystem they cover)

- `docs/SYSTEM_WALKTHROUGH.md` — **v8, new.** The document to read for "what does the code
  actually do, file by file, method by method" — a complete trace from photo upload to verdict
  to report, naming exact functions and exactly which signals drive each number, cross-
  referenced to PROJECT_MEMORY bug numbers throughout. Where ALGORITHM_DESIGN.md explains WHY
  each choice was made, this explains WHAT HAPPENS AND WHERE — read this one first if you're
  trying to understand or debug real behavior; read ALGORITHM_DESIGN.md if you're trying to
  understand or change the design itself.
- `docs/ALGORITHM_DESIGN.md` — **v7, new.** The document to read for "why does the engine work
  this way" — every algorithm/statistical choice in the classical pipeline (preprocessing,
  alignment, per-feature similarity, fusion weights, thresholding, TTC) with the literature or
  first-principles reasoning behind it, and explicitly cross-referenced to the PROJECT_MEMORY
  bug numbers that found each one wrong and fixed it. This file (PROJECT_MEMORY) is the
  chronological "what happened and when" record; ALGORITHM_DESIGN.md is the synchronous
  "here's the whole design, organized by decision, as it stands today" reference.
- `docs/ARCHITECTURE.md` — system diagram, module responsibilities, data flow, LLM boundary
  rationale with the supporting evidence.
- `docs/USAGE.md` — how to install, run, and use every part (web, CLI, API) end to end.
- `docs/TRAINING.md` — the v2 dataset calibration pipeline: what it does, how to run it with
  your own dataset, and the actual validation-run results.
- `docs/BUILD_LOG_v6.md` — real commands/scripts and their actual output for every v5 fix
  (Bugs #20-#23) — the receipts behind this file's summaries.
- `docs/System_Working_Logic.html` (+ `.png` render) — v3: presentation-level, single-file
  complete explanation of the whole system with workflow diagrams (enrollment flow,
  verification flow, threshold/decision logic, architecture). Built for onboarding/
  presentation purposes — this file (PROJECT_MEMORY_v8.md) stays the primary source of truth for
  history and decisions; that one is the polished explainer built from it.
- `README.md` — quickstart + project overview (points here for anything deeper).
