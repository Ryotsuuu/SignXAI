# SignXAI — Handwritten Signature Verification Using Explainable AI

Enroll a handful of genuine signature samples for a person. Show the
system a new ("questioned") signature. Get back a verdict — **GENUINE
/ FORGED / INCONCLUSIVE** — a full breakdown of *why*, visual evidence
you can actually inspect, and a plain-language forensic explanation.
Everything runs locally: no external training dataset per person, no
cloud dependency, and (optionally) a local AI model for the narrative
layer instead of a cloud API.

**Start here if you're modifying the code:**
[`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md) is the single source of
truth for this project's history — every bug found, how it was fixed,
what's verified vs. still open, and why design decisions were made.
Read it before making changes; don't re-introduce something it already
documents as tried-and-reverted. For "what does the code actually do"
(a complete file-by-file, method-by-method trace from photo upload to
verdict to report, naming exactly which signals drive each number), see
[`docs/SYSTEM_WALKTHROUGH.md`](docs/SYSTEM_WALKTHROUGH.md). For "why
does the algorithm work this way" (the reasoning behind each choice,
rather than what happens or when), see
[`docs/ALGORITHM_DESIGN.md`](docs/ALGORITHM_DESIGN.md).

---

## Table of contents

1. [What this actually does](#1-what-this-actually-does)
2. [How a verdict gets decided](#2-how-a-verdict-gets-decided-the-important-part)
3. [Quick start](#3-quick-start)
4. [Every way to run it](#4-every-way-to-run-it)
5. [The CLI, command by command](#5-the-cli-command-by-command)
6. [The web app, screen by screen](#6-the-web-app-screen-by-screen)
7. [The REST API](#7-the-rest-api)
8. [The optional local AI layer](#8-the-optional-local-ai-layer)
9. [Project layout](#9-project-layout)
10. [Data & privacy](#10-data--privacy)
11. [Current status & known limitations](#11-current-status--known-limitations)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What this actually does

Give it a person's genuine signature samples once ("enrollment"). From
then on, hand it any new signature and ask: *is this really theirs?*

For each verification you get:

- **A verdict**: `GENUINE`, `FORGED`, or `INCONCLUSIVE` (routed to human
  review when the evidence genuinely doesn't support a confident call —
  this is a deliberate third option, not a failure state).
- **A confidence score** and the raw fused similarity score against the
  decision threshold.
- **Full transparency into how that verdict was reached**: the
  questioned signature's score against *every single enrolled reference
  sample* individually, plus an aggregate "prototype" score combining
  all of them — not just whichever one reference happened to match best.
- **Visual evidence**: a divergence heatmap, an ink-agreement overlay
  (green = matching ink, red = only in the questioned signature, blue =
  only in the reference), a stroke-skeleton comparison, and a bar chart
  of the score against every reference plus the combined profile score.
- **A metrics table**: ~20+ measurable features (stroke width, curvature,
  pressure proxy, ink density, aspect ratio, Hu moments, and more), each
  compared against this specific person's own enrolled average, with a
  z-score and a plain-language interpretation — not compared against
  some generic population.
- **Real SHAP attribution**: genuine Shapley-value analysis (not a
  heuristic) showing exactly which measurements pushed the verdict one
  way or the other, and by how much.
- **An edge-case review**: an independent second pass that checks
  whether the individual reference comparisons actually agree with the
  headline verdict, and downgrades to `INCONCLUSIVE` when they don't —
  this is what stops one unusually good or bad reference match from
  silently deciding the whole case.
- **A plain-language forensic report**, generated either by a local AI
  model (if you've installed one) or a deterministic template — either
  way, grounded strictly in the metrics table above it. The AI never
  decides the verdict; see [Section 2](#2-how-a-verdict-gets-decided-the-important-part)
  for why that split is a hard rule.
- **A downloadable PDF** of the full report, plus JSON/HTML if you need
  to pipe it into something else.
- **A feedback loop**: confirm or correct each verdict, and the
  system adapts — nudging its decision threshold, and (if you opt in for
  a confirmed-genuine case) promoting the questioned signature into the
  profile as a new reference sample, growing what it knows about that
  person's natural handwriting variation over time. This is real
  test-time calibration, not a fire-and-forget popup.

You can use all of this through a **web app**, a **terminal UI**, or
directly through a **REST API** — see [Section 4](#4-every-way-to-run-it).

## 2. How a verdict gets decided (the important part)

The verdict is decided **entirely by a classical, per-person-calibrated
engine — never by the AI model, even when one is installed.**

Why this matters: a real May 2026 study found vision-LLMs are excellent
at telling different people's signatures apart, but 7–8x worse than
specialized classical/embedding methods at catching *skilled* forgeries
specifically — and get *more* confidently wrong, not less, when allowed
to freely reason about a case (the "Rationalization Trap" — full
citation and discussion in
[`docs/ARCHITECTURE.md` §2](docs/ARCHITECTURE.md)). So here, the AI's
only job — when one is installed at all — is to narrate metrics the
classical engine already computed and already used to decide the
verdict. It is architecturally incapable of overriding that verdict; if
no AI model is installed, a deterministic template does the same job
from the same metrics, just without the free-form prose.

The engine itself, in short (full detail in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)):

1. **Preprocess** both the questioned signature and every enrolled
   reference (EXIF-orientation fix, grayscale, denoise, Otsu
   binarization, crop, canvas-normalize).
2. **Align** the questioned signature onto each reference. Several
   candidate transforms are generated (identity / centroid-translation /
   image-moments rotation / ORB+ECC, each rotation-clipped to a plausible
   range) and the one that actually improves ink overlap wins — alignment
   can only ever match-or-beat doing nothing.
3. **Extract features**: ~20+ classical scalar measurements plus
   HOG/LBP/Hu-moment vectors, per signature.
4. **Score against EVERY enrolled reference individually** (SSIM,
   ink-IoU, HOG-cosine, LBP-cosine, Hu-cosine, DTW, and a per-writer
   scalar-profile z-score, fused with weights), *and* build an aggregate
   "prototype" score by averaging those components across every
   reference first. The final decision score blends the best individual
   match with that aggregate — never just one cherry-picked reference.
5. **Decide**: fused score vs. this person's own adaptive threshold
   (calibrated from their own enrolled samples' natural variation, with
   a shrinkage estimator so 1-2 samples don't produce a falsely-tight
   threshold) — `GENUINE` / `INCONCLUSIVE` / `FORGED`.
6. **Re-check**: an independent edge-case pass looks at whether the
   individual per-reference comparisons actually agree with that fused
   call, and downgrades to `INCONCLUSIVE` if they don't — it can only
   ever add caution, never invert a verdict.
7. **Explain**: heatmap, overlay, skeleton comparison, per-reference
   agreement chart, SHAP attribution, and (optional) a real Grad-CAM from
   a small per-profile-trained CNN — generated from what was already
   decided, never feeding back into it.

Every one of these calibrations is per-person and computed only from
that person's own enrolled samples. Nothing about how one person's
signatures are judged depends on any other person's data, and the app
ships with zero pre-trained weights of its own — the only thing it ever
optionally downloads is a general-purpose local AI model for writing the
narrative, which never sees the decision logic.

## 3. Quick start

```bash
python install.py          # one command, any OS — creates .venv/, installs everything needed
```
Then launch it:
```bash
# macOS / Linux
.venv/bin/python main.py           # web app at http://127.0.0.1:8420
.venv/bin/python cli.py            # terminal UI, same engine

# Windows (PowerShell or cmd)
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe cli.py
```
Or use the convenience launcher for your OS — same thing, and it works
no matter what directory you run it from:
```bash
./run.sh          # macOS/Linux, web app
./run_cli.sh       # macOS/Linux, terminal UI
run.bat            # Windows, web app
run_cli.bat        # Windows, terminal UI
```

Want the optional local AI explanation layer too?
```bash
python install.py --llm
```
This downloads a ~2.8GB local vision-language model the first time it
runs. The app works fully without it (a deterministic template writes
the explanation instead) — see [Section 8](#8-the-optional-local-ai-layer).

**Already ran `install.py` once and hit an install error?** Just run
`python install.py` again — it's idempotent (safe to re-run), and if a
previous run left a broken virtual environment behind (see
[Troubleshooting](#12-troubleshooting)), re-running now self-repairs it.

## 4. Every way to run it

| Interface | Command | Best for |
|---|---|---|
| **Web app** | `.venv/bin/python main.py` (or `run.sh`/`run.bat`) | Full visual experience — evidence images, charts, drag-and-drop uploads. Opens automatically at `http://127.0.0.1:8420`. |
| **Terminal UI** | `.venv/bin/python cli.py` (or `run_cli.sh`/`run_cli.bat`) | Scriptable environments, headless servers, or if you just prefer a terminal. Same underlying engine, same verdicts — see [Section 5](#5-the-cli-command-by-command). |
| **REST API** | `.venv/bin/python main.py`, then call `http://127.0.0.1:8420/api/...` | Integrating into your own tooling. Interactive Swagger docs at `http://127.0.0.1:8420/docs`. See [Section 7](#7-the-rest-api). |

All three talk to the exact same engine and the exact same on-disk data
under `data/` — enroll a profile in the web app, verify it from the CLI,
and it's the same profile either way.

Every entry point (`install.py`, `main.py`, `cli.py`,
`training/calibrate_weights.py`, and the `run*` launcher scripts) resolves
its own location from where the file actually lives, not from your
current directory — so it doesn't matter what folder you run these from,
or whether you've renamed the project folder itself. This was directly
tested: renaming the entire project directory and launching from a
completely unrelated working directory produced identical behavior.

## 5. The CLI, command by command

Launch with `.venv/bin/python cli.py` (or `run_cli.sh`/`run_cli.bat`).
You'll see a menu:

```
1  Enroll a signer
2  Verify a signature
3  Case log
4  AI model status / download
5  Device diagnostics
0  Exit
```

**1 — Enroll a signer**
Type a name. If a profile with that name already exists, you're asked
whether to add more samples to it or start fresh. Then paste the path to
each genuine signature image, one at a time, blank line to finish (2-5
samples is a good range; more improves calibration, up to a cap).

**2 — Verify a signature**
Pick a profile from the list, then give the path to the questioned
signature image. You'll see:
- The verdict, fused score vs. threshold, and a confidence bar.
- A table scoring the questioned signature against **every** enrolled
  reference individually, plus the combined "prototype" score — direct
  visibility into what the verdict was actually based on.
- An edge-case warning panel, if the independent re-check flagged
  disagreement between those individual comparisons.
- The deviant features (|z| > 2) table.
- A generated forensic explanation (AI-narrated if available, template
  otherwise) and a path to the full PDF report.
- A prompt: was this verdict correct? If yes and it was `GENUINE`,
  you're also asked whether to add this signature to the profile as a
  new reference sample (real test-time adaptation). Either way, you'll
  see a training/adaptation panel showing exactly what changed —
  updated threshold, profile maturity, whether a new reference was added.

**3 — Case log**
Lists your most recent verification cases across all profiles.

**4 — AI model status / download**
Shows detected hardware, whether the model is already present on disk
(and its size) independently of whether it's loaded into memory, and
lets you trigger a download if it's missing. As of this version, if the
model is already on disk and your hardware qualifies, it's loaded
automatically in the background a few seconds after `main.py`/`cli.py`
starts — you don't need to do anything here unless you're installing it
for the first time.

**5 — Device diagnostics**
Raw hardware detection output: OS, architecture, RAM, GPU(s) detected,
VRAM, and whether this machine meets the floor for the local AI model.

## 6. The web app, screen by screen

Launch with `.venv/bin/python main.py` (or `run.bat`/`run.sh`) — it opens
`http://127.0.0.1:8420` in your default browser automatically.

- **Enroll tab** — create or pick a profile, upload/drag in genuine
  signature samples.
- **Verify tab** — pick a profile, upload the questioned signature.
  Results render inline: verdict banner, per-reference score table,
  edge-case banner (if triggered), the four evidence images (heatmap,
  ink-agreement overlay, stroke-skeleton comparison, reference-agreement
  chart), the full metrics table (click a row to expand its
  plain-language interpretation), SHAP attribution, the narrative
  explanation, a PDF download link, and feedback buttons (with the
  "add to profile if confirmed genuine" checkbox for test-time
  adaptation) that update a persistent training panel — not a popup that
  disappears.
- **Case log tab** — browse past verifications across all profiles.
- **AI Model tab** — same information as CLI option 4: hardware
  detection, on-disk model status, download/load controls.

## 7. The REST API

Base URL once `main.py` is running: `http://127.0.0.1:8420/api`.
Full interactive docs (try-it-out included) at
`http://127.0.0.1:8420/docs`.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/profiles/check-name` | Check whether a profile name already exists. |
| `POST` | `/profiles` | Create a new profile (`{"name": "..."}`). |
| `GET` | `/profiles` | List all profiles. |
| `GET` | `/profiles/{profile_id}` | Get one profile's full detail (references, stats, feedback history). |
| `DELETE` | `/profiles/{profile_id}` | Delete a profile and its enrolled samples. |
| `POST` | `/profiles/{profile_id}/enroll` | Add reference samples (`{"images": ["data:image/...;base64,...", ...]}`). |
| `POST` | `/verify` | Verify a signature against a profile (`{"profile_id": "...", "image": "data:image/...;base64,..."}`). Returns the full case object described in [Section 1](#1-what-this-actually-does), plus a generated report. |
| `POST` | `/cases/{case_id}/feedback` | Confirm or correct a verdict (`{"correct": true/false, "add_as_reference": true/false}`). |
| `GET` | `/cases` | List recent cases (`?limit=20`). |
| `GET` | `/cases/{case_id}` | Get one case's full detail. |
| `GET` | `/cases/{case_id}/asset/{filename}` | Fetch an evidence image (heatmap, overlay, skeleton, agreement chart, etc). |
| `GET` | `/reports/{case_id}` | Full report as JSON. |
| `GET` | `/reports/{case_id}/html` | Full report as a rendered HTML page. |
| `GET` | `/reports/{case_id}/pdf` | Full report as a downloadable PDF. |
| `GET` | `/llm/status` | Local AI model status: hardware detected, on-disk presence + size, load state. |
| `POST` | `/llm/download` | Trigger a background download of the local AI model. |
| `POST` | `/llm/load` | Trigger a background load of an already-downloaded model into memory. |

Images are passed as base64 data URLs (`data:image/png;base64,...`) in
JSON bodies rather than multipart form uploads, so every endpoint is a
plain JSON request/response.

## 8. The optional local AI layer

Purely additive — see [Section 2](#2-how-a-verdict-gets-decided-the-important-part)
for why it can never affect the verdict. When installed, it writes the
plain-language forensic narrative instead of the deterministic template.

**Install it:**
```bash
python install.py --llm
```
This installs `llama-cpp-python` (picking the right prebuilt wheel for
your OS/GPU automatically — CPU, NVIDIA CUDA, or Apple Metal — falling
back to a from-source build, which needs a C++ compiler, only if no
matching prebuilt wheel is available) and downloads a single ~2.8GB model
(`google/gemma-4-E2B-it-qat-q4_0-gguf`) if your hardware has at least
~3GB of combined RAM+VRAM. If it doesn't, the installer tells you so and
skips the download — the app still works fully without it.

**Already have the model downloaded from elsewhere?** Just copy it into
`data/models/google--gemma-4-E2B-it-qat-q4_0-gguf/` (matching that exact
folder-name pattern — slashes in the model ID become double-dashes). The
app checks that folder's actual size on disk every time it starts (and
checks your GPU) and loads it automatically in the background within a
few seconds — no download, no manual "load" click needed. The AI Model
tab / CLI option 4 will show it as found once it's there.

**Hardware requirements:** roughly 3GB of combined RAM + VRAM. NVIDIA
GPUs get automatic CUDA offload; Apple Silicon Macs get automatic Metal
offload; everything else runs on CPU (slower, but fully functional).

## 9. Project layout

```
signxai/
├── main.py                 web app entry point
├── cli.py                  terminal UI entry point
├── install.py               cross-platform installer (creates .venv/, installs deps)
├── run.sh / run.bat          convenience launchers (web app)
├── run_cli.sh / run_cli.bat  convenience launchers (terminal UI)
├── requirements*.txt        core / LLM / CNN / training dependency lists
├── app/
│   ├── config.py             paths, tunable constants
│   ├── device.py             hardware detection (RAM/GPU/model-on-disk)
│   ├── core/
│   │   ├── engine.py           THE DECISION LAYER — enroll/verify/feedback
│   │   ├── preprocess.py       image normalization
│   │   ├── features.py         ~20+ scalar features + HOG/LBP/Hu vectors
│   │   ├── align.py            multi-candidate spatial alignment
│   │   ├── similarity.py       per-comparison-method scoring
│   │   ├── explain.py          heatmap/overlay/skeleton/agreement-chart generation
│   │   ├── shap_explain.py     real KernelSHAP attribution
│   │   ├── signet.py           optional per-profile CNN + Grad-CAM (needs torch)
│   │   ├── zoning.py            content-adaptive local/zonal stroke-pattern analysis
│   │   └── storage.py          on-disk profile/case persistence
│   ├── llm/
│   │   ├── manager.py          model download/load/status, hardware-gated
│   │   ├── report.py           report generation (AI or template)
│   │   └── prompts.py          the vision-AI system prompt
│   └── api/
│       ├── routes.py            all REST endpoints
│       └── schemas.py           request/response models
├── web/                     the web app's static frontend (HTML/CSS/JS)
├── training/
│   └── calibrate_weights.py   optional: fit fusion weights on a labelled dataset
├── data/                    created at runtime — profiles, cases, reports, models (gitignored)
├── docs/
│   ├── ARCHITECTURE.md        system design + the reasoning behind it
│   ├── USAGE.md               step-by-step usage guide
│   ├── TRAINING.md            the optional weight-calibration pipeline
│   ├── BUILD_LOG_v6.md        real test transcripts backing this version's fixes
│   └── System_Working_Logic.html  presentation-style visual walkthrough
└── PROJECT_MEMORY_v8.md    full project history — read this first if you're changing code
```

## 10. Data & privacy

Everything lives under `data/` next to the code, created on first run:
- `data/profiles/<id>/` — enrolled reference images + computed features
  for each person.
- `data/reports/<case_id>/` — every verification's evidence images,
  JSON/HTML/PDF report.
- `data/models/` — the local AI model, if you've installed one.

Nothing is sent anywhere. There's no telemetry, no external API calls
for the core verification pipeline, and the only network access the app
itself ever makes is the one-time model download in
[Section 8](#8-the-optional-local-ai-layer) (and only if you run
`install.py --llm` or click "download" — verification and enrollment
never touch the network at all).

## 11. Current status & known limitations

**v8 (latest round)** — the user provided a real export of their actual
`data/` directory (enrolled reference images, full case records, saved
intermediate processing images) for the first time, enabling direct
inspection of the real bytes the engine processes rather than working
from summaries. This found the single most consequential bug in the
project's history: plain global-image thresholding was misreading an
uneven lighting gradient across a real photographed page as 46% of the
entire frame being "ink" — corrupting the crop, and everything measured
after it, for that image. Fixed with standard illumination flattening
before thresholding; verified directly (ink-pixel count on the real
reported image collapsed from 886,129 to 10,298, correctly isolating
just the real signature) and confirmed to be un-regressive on already-
working images. A second bug found using the same real data: a
unanimous, correctly-scored FORGED verdict (two completely different
people's signatures, zero disagreement between references) was still
being downgraded to INCONCLUSIVE by an alignment-quality check that
never considered whether the references actually agreed — fixed so
poor overlap only softens a verdict when it's paired with real
disagreement, not when every reference already agrees. Also this round:
a real pre-flight CUDA-readiness check in the installer (checks the
actual driver CUDA version and, on Windows, the VC++ Redistributable,
before attempting a GPU wheel download that could be doomed to fail);
a corrected model choice (the user's specifically-requested model
turned out to be a non-standalone draft model — switched to a properly-
matched, half-the-size alternative in the same family instead, after
checking Hugging Face directly rather than assuming); and real
per-profile adaptive fusion weighting, so a signer whose signature
happens to vary more in one feature family than another is no longer
scored against a single global assumption tuned on different people's
data. Full detail, with the real before/after numbers for each fix, in
[`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md) Bugs #50–#54. New:
[`docs/SYSTEM_WALKTHROUGH.md`](docs/SYSTEM_WALKTHROUGH.md), a complete
file-by-file, method-by-method trace of what the code does and which
signals drive each verdict.

**v7 (previous round)** — the two issues most recently reported directly:
(1) a real genuine signature was scoring confidently FORGED — root-caused
(not just patched) to two previously-undiscovered statistical bugs in the
fusion math (a 2-reference standard-deviation artifact, and raw-cosine
Hu-moment comparison being wrecked by a moment that provably flips sign
between a signer's own genuine photos) plus three compounding issues (a
backwards alignment penalty, pixel-IoU being hypersensitive to
photo-to-photo stroke-width noise, and ruled/notebook-paper lines
surviving into the ink mask uncaught) — verified against the reporter's
own 5 real signature photos: the wrongly-FORGED case moved to a
correctly-uncertain INCONCLUSIVE with a positive margin above threshold
(stated plainly: not yet a confident GENUINE, with only 2 enrolled
references — this system's own long-standing advice to enroll 3-5
references remains the most reliable way to resolve a genuinely
borderline case further), while the known real forgery among the same 5
photos stayed correctly rejected throughout every fix. Two further bugs
in the online-feedback loop were found and fixed alongside this (repeated
re-verification of an already-enrolled sample could silently corrupt the
profile's calibration). (2) the local AI model could get stuck on
"loading" indefinitely with no explanation on either GPU or CPU — fixed
with a real timeout (converts a silent hang into an actionable error
message instead of nothing), VRAM-aware GPU-offload gating (skips a
doomed GPU attempt on a card too small for the model instead of hanging/
thrashing), and a smaller default context window (halves load-time
memory allocation). The recognition fixes are verified against real
signature images; **the LLM-hang fix is verified at the logic level with
a simulated hang and still needs confirmation on real GPU hardware** —
see [`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md) Bugs #34–#44 for the
full investigation, evidence, and what's still open, and
[`docs/ALGORITHM_DESIGN.md`](docs/ALGORITHM_DESIGN.md) for why the fixed
algorithm now works the way it does.

**v6 (previous round)** — two more real, deeply-evidenced issues fixed:
(1) the AI Model status page could show "ready" with GPU hardware
detected while generation actually ran on CPU — `n_gpu_layers=-1` is
silently ignored by a non-GPU-capable backend build, with no error at
all; the status page now captures llama.cpp's own native load log to
report a confirmed answer instead of assuming success from hardware
presence alone. (2) A signature with catastrophically divergent pen-lift/
loop/junction counts — z-scores at the hard cap on four independent
features simultaneously, about as extreme a signal as this system can
produce — still only scored INCONCLUSIVE, because those counts were
averaged together with ~17 other, easier-to-superficially-mimic features
and diluted into a moderate-looking score (the exact same dilution
mechanism already fixed once locally for the zonal/local analysis, now
found and fixed globally too). Tested directly against real user-
provided signature photos: a visibly different, forged signature moved
from INCONCLUSIVE all the way to a correctly, clearly FORGED verdict.
Full detail — including the honest cost this introduced on a different,
real skilled-forgery case — in [`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md)
(Bugs #32–#33) and [`docs/BUILD_LOG_v6.md`](docs/BUILD_LOG_v6.md).

**v6 (previous additions)** — three more issues addressed after the initial
v6 release, from continued real use: the LLM installer could report a
successful install on Windows yet the model still failed to load at
runtime (a missing CUDA runtime component or Visual C++ Redistributable
— now detected automatically with a fallback to the CPU backend instead
of leaving a broken install in place); a new **zonal (local) signature
analysis** module was added specifically to catch structurally
meaningful but spatially small differences (e.g. a stroke's direction-
reversal/zigzag count) that whole-signature global metrics can dilute
into insignificance — the signature is divided into content-adaptive
regions (not a fixed grid, so it generalizes across scripts), each
compared individually and with its neighbors, with results shown as a
new visual (`zone_grid.png`) in both the web UI and PDF report; and two
UI text lines were removed. Full detail, including three real design
iterations caught by testing (not shipped on the first attempt) for the
zonal module, in [`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md) (Bugs
#30–#31) and [`docs/BUILD_LOG_v6.md`](docs/BUILD_LOG_v6.md).

**v6 (initial)** — fixed four issues reported directly from real use:
signature recognition was sensitive to photo framing/position (root
cause: the crop step wasn't robust to stray marks or scan artifacts
outside the actual handwriting); the generated PDF report's heatmap
rendered as a flat color wash instead of a real heatmap, and its
divergence-region boxes were a rigid, colliding grid; the LLM installer
worked on Linux/macOS but failed on a real Windows machine (pip silently
picked a source-only release over the CUDA wheel, then failed compiling
because the MSVC compiler environment wasn't loaded); and the web UI
had a missing remove-control on enrollment thumbnails, no way to attach
a remark to feedback, and needed a full visual restyle. All four are
root-caused, fixed, and tested as thoroughly as this environment allows
— see [`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md) (Bugs #26–#29) and
[`docs/BUILD_LOG_v6.md`](docs/BUILD_LOG_v6.md) for full detail and real
before/after test evidence, including two of my own design mistakes
that were caught by testing before shipping (documented, not hidden).

**v5** — fixed four issues reported directly from real use (recognition
collapsing to a single reference, a rotation/alignment bug in the
evidence images, the local AI model not being detected when already
present on disk, and feedback having no real training signal), plus a
Windows install failure (`venv` creating an environment with no working
`pip`), a "run from anywhere" gap in the convenience launcher scripts,
and a broken LLM wheel-install URL scheme found while re-verifying
against fresh upstream docs. Full details in `PROJECT_MEMORY_v8.md`
(Bugs #20–#25).

**Honestly, not glossed over:** with purely classical features, detecting
*skilled* forgeries specifically (as opposed to a random impostor's
signature) is still hard — the latest round of fixes (Bug #33) produced
the first real gains on this in the standard 10-writer benchmark sweep
(0/10 → 2/10 confidently caught), tested directly against real user-
provided forged/genuine signature photos, not just synthetic data — but
at a real, disclosed cost (0/10 → 1/10 new false-accept on a different,
more skilled forgery that happened to replicate construction patterns
well while failing on shape/pixel fidelity — the mirror image of what
the new signal catches). Genuine signatures are still never misclassified
as forged across the same sweep. This remains an architectural reality,
not a solved problem: no single classical feature (or fusion of them)
catches every skilled forgery, because different forgers fail
differently, and this is not something another threshold adjustment
fully fixes. The literature-supported next step is the already-written
(but not yet independently verified end-to-end) CNN-SigNet + Grad-CAM
module in `app/core/signet.py`, which needs a machine where `torch`
actually installs and imports to evaluate.

**Not yet independently verified on real hardware:** GPU offload on a
real NVIDIA/Apple Silicon machine, the local-AI-model autostart flow
against a real downloaded model, the Windows `.bat` launcher and MSVC
auto-detection logic (verified on Linux via equivalent mechanisms and
graceful-failure paths; never run against real Windows/Visual Studio),
the LLM DLL-load fallback against a real GPU/CUDA runtime, the zonal
analysis module against real multi-script (CJK/Devanagari/etc.)
signatures (designed to generalize by construction, tested against
synthetic zigzags and real Latin-script CEDAR data, but not yet against
genuine non-Latin samples), and the v6 web UI restyle's actual rendered
appearance (verified structurally — every CSS class and element ID
cross-checked, JS syntax-validated — but this sandbox has no reachable
headless-browser
download source to take a real screenshot). If you hit anything here,
`PROJECT_MEMORY_v8.md`'s open-questions section names exactly what to
check.

## 12. Troubleshooting

**`python install.py` fails with "No module named pip"**
This means a previous run left a broken virtual environment behind (a
known, now-fixed issue — see Bug #24 in `PROJECT_MEMORY_v8.md`). Just
run `python install.py` again: it now detects and automatically repairs
a `.venv` with a broken pip before doing anything else. If it still
fails after that, the message it prints will tell you exactly what to
try next (usually a temporary antivirus exclusion, or a manual
`ensurepip` command it gives you verbatim).

**The web app doesn't open automatically / "connection refused"**
Give it a few seconds after `main.py` starts, then go to
`http://127.0.0.1:8420` manually. Check the terminal for errors — if
port 8420 is already in use by something else, that's the most common
cause.

**"AI Model" status is stuck on "not_installed" even though I copied the
model in myself**
Double-check the exact folder name:
`data/models/google--gemma-4-E2B-it-qat-q4_0-gguf/` (note the double
dash, and that it must be a folder containing the model files, not the
files directly under `data/models/`). The app checks the *actual total
size* of that folder on every check — it needs to add up to roughly 2.8GB
or more to be recognized as present.

**Installing the AI layer (`--llm`) wants to compile from source and
fails**
This means no prebuilt wheel matched your exact Python version/platform
combination, so it fell back to building from source — which needs a
C++ compiler (Windows: "Build Tools for Visual Studio", workload
"Desktop development with C++"; macOS: `xcode-select --install`; Linux:
`build-essential` + `cmake`). If you don't want to install one, that's
fine — skip `--llm` entirely; the app works completely normally without
it, just with template-generated explanations instead of AI-narrated
ones.

**Something else**
Check [`docs/USAGE.md`](docs/USAGE.md) for the detailed step-by-step
guide, or [`PROJECT_MEMORY_v8.md`](PROJECT_MEMORY_v8.md) — if it's a bug
that's already been hit before, it's very likely documented there with
the exact fix and how it was verified.
