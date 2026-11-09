# Usage Guide

## 1. Install (one command)

```bash
python install.py
```

This works identically on Windows, macOS, and Linux. It:
1. Auto-creates a local `.venv/` (isolates dependencies; works around modern
   Linux's PEP 668 restriction — see `PROJECT_MEMORY_v8.md` Bug #4).
2. Installs the core engine dependencies (numpy, opencv, scikit-image,
   fastapi, rich, etc.) — everything needed for enrollment, verification,
   the web UI, and the terminal UI.

At the end it prints the exact command to run next, e.g.:
```
Web app:      .venv/bin/python main.py        (macOS/Linux)
Terminal UI:  .venv/bin/python cli.py
```
or on Windows: `.venv\Scripts\python.exe main.py`

### Optional: add the local AI explanation layer

```bash
python install.py --llm
```

This additionally installs `llama-cpp-python` (auto-selecting a prebuilt
CUDA wheel if you have an NVIDIA GPU, or the standard wheel which includes
Metal support on macOS) and downloads the recommended vision-language model
for your hardware automatically.

**This step is entirely optional.** Without it, the app still fully works —
verification, metrics, heatmaps, and reports all work — the only difference
is the report's explanation text comes from a deterministic template
("Feature X deviated by Z, exceeding threshold Y") instead of an AI-written
narrative. See `docs/ARCHITECTURE.md` §2 for why the AI is never the one
deciding valid/invalid regardless.

To check whether your machine can run the local AI model without
installing anything:
```bash
python install.py --check
```
There is exactly one model this project downloads —
`google/gemma-4-E2B-it-qat-q4_0-gguf` — needing roughly 3GB+ combined
RAM/VRAM. No tiers, no larger/smaller options (v4 simplification: this was
previously a multi-tier system, removed after a bug where hardware
happened to qualify for a much larger model than intended).

## 2. Run

**Web app** (recommended — full visual experience: signature pad, heatmaps,
metrics table, case log):
```bash
.venv/bin/python main.py            # macOS/Linux
.venv\Scripts\python.exe main.py    # Windows
```
Opens automatically at `http://127.0.0.1:8420`. If your browser doesn't
launch automatically (e.g. over SSH), open that URL manually.

**Terminal UI** (same functionality, no browser needed):
```bash
.venv/bin/python cli.py
```

Both talk to the exact same engine and the exact same `data/` folder — you
can enroll a profile in the web UI and verify against it from the terminal,
or vice versa.

## 3. How to use it

### Enroll a signer
1. Web: go to the **Enroll** tab. Terminal: choose **1**.
2. Enter the signer's name. If a profile with that name already exists, the
   app tells you and asks whether to add to it (same person) or use a more
   specific name (different person, name collision) — see
   `docs/ARCHITECTURE.md` for the exact dedup logic.
3. Provide 1-5 genuine signature samples — draw on the pad, or upload image
   files (web), or give file paths (terminal).
4. Submit. The engine computes this person's baseline feature statistics and
   an adaptive threshold from just these samples.

**More samples = more reliable threshold.** With only 1 sample there's no
variance estimate at all (the engine uses a conservative default). 2-3
samples give a rough estimate; 5 is noticeably better. This isn't a UI
limitation, it's a statistical one — see `PROJECT_MEMORY_v8.md` Section 2 for
the actual measured numbers (from a comparable published system) that
motivate "more samples is better," and don't expect near-perfect accuracy
from 1-2 samples on skilled forgeries — that's an honest limit, not a bug.

### What GENUINE / FORGED / INCONCLUSIVE actually mean

v9 addition (PROJECT_MEMORY_v9.md Bug #59) — this section exists because the
three verdicts were previously implicit in code rather than defined anywhere
a user could check. Precisely:

- **The fused score** is `0.5 × best-matching-reference score + 0.5 ×
  prototype (combined-profile) score` — never a single reference's score
  alone, and never the AI narration layer's opinion (the AI never decides a
  verdict, see `ARCHITECTURE.md`).
- **GENUINE**: fused score ≥ your profile's calibrated threshold **plus** a
  ±0.07 safety band, *and* an independent sanity check found the individual
  per-reference comparisons actually support that call.
- **FORGED**: fused score ≤ threshold **minus** the same 0.07 band, with the
  same sanity check passing. (This means "statistically inconsistent with
  the enrolled reference set" — the system cannot and does not claim to
  detect fraudulent *intent*; see the recommendation text in every FORGED
  report, which says to route to human review, not to treat this as a legal
  finding.)
- **INCONCLUSIVE** happens via exactly two distinct, separately-reported
  paths — every case's JSON response now carries a `verdict_reason` object
  with a `reason_code` telling you which one fired, and the web UI shows
  the matching explanation box automatically:
  1. `borderline_band` — the fused score simply falls within ±0.07 of your
     threshold. This is the **common** case, and by itself does not mean
     anything is wrong with the photo or the pipeline — it means this
     specific questioned signature sits between how confidently-genuine and
     confidently-different signatures have scored for *this* profile
     historically. Check the reference-agreement/alignment-reliability
     numbers shown alongside it: if those are also poor, image quality is
     a likely contributor; if they're solid, it's simply a hard, close call.
  2. `edge_case_downgrade` — the fused score was NOT borderline (it clearly
     pointed to GENUINE or FORGED), but an independent per-reference sanity
     check found the underlying evidence contradicts itself (see the three
     checks below) — so the confident-looking score is not trustworthy and
     was overridden rather than reported at face value.

  The three sanity checks behind path 2, all of which can ONLY push a
  verdict toward INCONCLUSIVE, never the reverse:
  - **Opposing evidence**: do individual reference comparisons land on the
    genuinely opposite side of the threshold from the fused call (e.g. one
    reference says clearly-genuine while another says clearly-forged)?
  - **Reference dispersion**: is the spread between individual reference
    scores large relative to how far the real fused score sits from
    threshold?
  - **Alignment reliability**: did spatial alignment find *any* meaningful
    ink overlap against most references? This exists only to catch a
    genuine registration catastrophe (wrong file, blank page) — it is
    explicitly *not* triggered when every reference unanimously agrees
    despite low overlap, since that pattern means "these really are
    different shapes" (real FORGED evidence), not "alignment failed".


1. Web: **Verify** tab, pick the profile from the dropdown. Terminal: **2**.
2. Provide the questioned signature (pad, upload, or file path).
3. Run. You get:
   - **Verdict**: GENUINE / FORGED / INCONCLUSIVE — see "What GENUINE /
     FORGED / INCONCLUSIVE actually mean" above for exactly what triggers
     each one, with a score and confidence.
   - **Metrics table**: every feature compared to that profile's baseline,
     flagged if it deviates more than 2 standard deviations. **Click any
     row to expand it** — shows what that feature measures and a
     plain-language interpretation of this specific signature's value
     (v2 addition; this was missing in v1).
   - **Evidence images**: divergence heatmap, ink-agreement overlay
     (green = matching ink, red = only in the questioned signature, blue =
     only in the reference), a stroke-skeleton comparison, and (v5) a
     **reference agreement chart** showing the questioned signature's
     score against every enrolled sample plus the combined profile
     ("prototype") score — direct visibility into which samples it was
     actually compared against.
   - **Per-reference table** (v5): the fused score and alignment success
     against each individual enrolled reference.
   - **Verdict reason** (v9): every case now explains itself — a
     `borderline_band` box (score too close to your threshold to call) or
     an `edge_case_downgrade` box (score looked clear-cut but the
     per-reference evidence disagreed with itself) for INCONCLUSIVE, or a
     brief confirmation for a clean GENUINE/FORGED. Never silent.
   - **Explanation**: AI-written (if you installed the LLM layer and it's
     ready) or template-generated (always available) — either way, grounded
     in the metrics table above it, never inventing new claims.
   - **Downloadable PDF** of the full forensic report.
4. **Give feedback** (Confirm correct / Mark wrong). This nudges that
   profile's threshold based on real outcomes over time. If you confirm a
   GENUINE verdict was correct, you can also tick **"Add to profile if
   confirmed genuine"** (v5) — this promotes the questioned signature into
   the profile as a new reference sample and relearns all of that
   profile's statistics from it (real test-time calibration, not just a
   threshold nudge). A persistent training panel shows exactly what
   changed — updated threshold, weights, maturity, and whether the sample
   was promoted — instead of a popup that disappears.

### Check the AI model status
Web: **AI Model** tab. Terminal: option **4**. Shows detected hardware,
recommended model tier, whether the model was found already on disk (and
its size) independently of whether it's loaded into memory yet, and a
download/load button as appropriate. As of v5, if the model is already
present on disk and your hardware qualifies, it loads automatically a few
seconds after the server starts — no click needed. The **"Load model
now"** button is only for retrying manually (e.g. right after copying
model files in while the server is already running).

### Case log
Every verification is saved. Web: **Case Log** tab. Terminal: option **3**.

## 4. What each file does (quick reference)

| File | Role |
|---|---|
| `install.py` | One-command cross-platform dependency installer |
| `main.py` | Web app entry point (FastAPI + static frontend) |
| `cli.py` | Terminal UI entry point (rich) |
| `app/config.py` | Paths, constants, app title |
| `app/device.py` | Hardware detection for the single supported LLM model |
| `app/core/preprocess.py` | Image loading, binarization, deskew, crop |
| `app/core/align.py` | Rigid alignment of query vs. reference before comparison |
| `app/core/features.py` | Classical stroke/geometry feature extraction |
| `app/core/similarity.py` | Per-writer z-scored distance → similarity score |
| `app/core/explain.py` | Heatmap / overlay / skeleton evidence image generation |
| `app/core/shap_explain.py` | v4: real KernelSHAP attribution over classical features |
| `app/core/signet.py` | v4: optional CNN-SigNet + Grad-CAM (needs torch, untested end-to-end) |
| `app/core/engine.py` | **The decision layer** — enroll/verify/feedback orchestration |
| `app/core/storage.py` | JSON persistence for profiles and cases |
| `app/api/routes.py` | REST API wiring the engine to HTTP |
| `app/api/schemas.py` | Request/response models |
| `app/llm/manager.py` | GGUF model download + llama-cpp-python inference |
| `app/llm/prompts.py` | Forensic-examiner system prompt + strict JSON contract |
| `app/llm/report.py` | **The explanation layer** — LLM narration with template fallback |
| `web/` | Frontend (plain HTML/CSS/JS, no build step) |
| `PROJECT_MEMORY_v8.md` | Single source of truth — history, bugs, roadmap. Read first. |
| `docs/ARCHITECTURE.md` | System diagram + design rationale |
| `docs/USAGE.md` | This file |
| `docs/TRAINING.md` | v2: how to calibrate fusion weights against a real dataset |
| `training/calibrate_weights.py` | v2: the calibration script itself |

## 5. Troubleshooting

- **"externally-managed-environment" error**: this is exactly what
  `install.py`'s venv auto-bootstrap fixes — make sure you're running
  `python install.py` (which creates and uses `.venv/`), not calling
  `pip install` directly yourself.
- **LLM status stuck on "downloading" for a long time**: model files are
  several GB; this is expected on a slow connection. Check terminal/log
  output for progress.
- **"No module named llama_cpp"**: you haven't run `install.py --llm` yet,
  or it failed (check for a C++ compiler requirement message in its
  output). The app still works fully without it — reports just use the
  template engine instead of AI narration.
- **Web UI shows "API-only mode, no UI" in the log**: the `web/` folder is
  missing relative to `main.py` — make sure you're running it from the
  project root, not a subfolder.
