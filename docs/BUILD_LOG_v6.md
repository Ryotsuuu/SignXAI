# Build log — v5 (chronological, honest; companion to PROJECT_MEMORY_v8.md)

This documents the actual commands/scripts run during v5 development and
their real output — referenced from PROJECT_MEMORY_v8.md's Bug #20-#23
entries. Nothing here is simulated; every result below came from running
the real code against real CEDAR signature images in the build sandbox.

## Setup

```bash
pip install -r requirements.txt   # numpy, opencv-python-headless, scikit-image,
                                   # scipy, fastapi, uvicorn, pydantic, reportlab, psutil
```
Test images: the same public CEDAR mirror already documented in
`REFERENCES.md` (`nikostsagk/signature-verification`, `cedar_dataset.zip`,
254MB, real 55-writer dataset) — downloaded fresh for this round of testing.

## Bug #20 — alignment rotation fix, verified

Script: preprocess a genuine CEDAR sample (`original_1_10.png`) and its
reference (`original_1_1.png`); separately, rotate the same query image
12° with PIL (simulating a skewed photo) before preprocessing; compare
raw (unaligned) ink IoU against `align.align()`'s output for both.

```
Unrotated query vs ref IoU (no alignment, raw binary): 0.1208
Unrotated query: aligned_ok= True   post-align IoU= 0.3024
Rotated(12deg) query vs ref IoU (no alignment, raw binary): 0.0848
Rotated(12deg) query: aligned_ok= True   post-align IoU= 0.1671
```
A visual ink-agreement overlay (green=match, red=query-only, blue=ref-only)
was also generated for the rotated case, post-alignment — strokes overlap
sensibly (green dominant along the actual signature shapes) rather than
being offset by a spurious rotation, which is what the pre-fix `align.py`
produced on real uploads per the user's report.

## Bug #21 — multi-reference fusion + edge-case engine, verified and self-corrected

**End-to-end functional test** (profile with 5 enrolled genuine references,
`original_1_1..5.png`):
```
GENUINE (held-out, writer 1) — original_1_10.png
  verdict: GENUINE   original_verdict: GENUINE
  score: 0.7573   threshold: 0.5867
  aggregate: {prototype_score: 0.7326, mean_reference_score: 0.7326,
              std_reference_score: 0.0503, best_reference: ref_0.png,
              n_references_compared: 5}
  edge_case_review: {triggered: false}

IMPOSTOR (writer 2) — original_2_1.png
  verdict: INCONCLUSIVE   score: 0.5943   threshold: 0.5867

SKILLED FORGERY (writer 1) — forgeries_1_1.png
  verdict: INCONCLUSIVE   score: 0.6348   threshold: 0.5867
```
All five individual reference scores and their per-component breakdown
were confirmed present in `case["reference_scores"]` for every case above
(not just the best match) — this is the direct fix for "goes through each
sample image."

**Dev-time self-correction, caught by a 10-writer sweep (not shipped
without re-testing):**

First version of `_edge_case_review()` — counted any individual verdict
not exactly matching the fused label as "disagreement":
```
writer 1: thr=0.587  genuine->INCONCLUSIVE(0.679)  forged->INCONCLUSIVE(0.635)  g_edge=True
```
Score 0.679 was clearly above `threshold + INCONCLUSIVE_BAND` (0.657) —
i.e. the fused call was a confident GENUINE — yet got downgraded because
3 of 5 individual reference comparisons landed in the merely-INCONCLUSIVE
band rather than the GENUINE band (individual scores: 0.566, 0.582, 0.649,
0.710, 0.713 — normal per-reference variability, not real disagreement).

Retuned to only count comparisons on the literal OPPOSITE side of the
threshold as disagreement. Same case, after the fix:
```
writer 1: thr=0.587  genuine->GENUINE(0.679)  forged->INCONCLUSIVE(0.635)  g_edge=False
```

**Full before/after 10-writer sweep** (5 references enrolled per writer,
1 held-out genuine + 1 skilled forgery verified per writer, 20
verifications, none of the test images enrolled):

| | before retune | after retune |
|---|---|---|
| genuine_correct | 3/10 | 5/10 |
| genuine_wrong | 0/10 | 0/10 |
| genuine_inconclusive | 7/10 | 5/10 |
| forged_correct | 0/10 | 0/10 |
| forged_wrong | 2/10 | 3/10 |
| forged_inconclusive | 8/10 | 7/10 |

Read honestly: the retune meaningfully improved genuine-signature
confidence (more genuine cases now resolve to GENUINE instead of parking
at INCONCLUSIVE) while keeping the zero-false-FORGED guarantee intact.
Skilled-forgery detection did not improve either version — 0/10
confidently caught in both — and is logged as an open architectural
limitation in PROJECT_MEMORY_v8.md Section 7, not something a threshold
retune fixed or is expected to fix.

## Bug #22 — LLM auto-detection, verified with a synthetic model file

This build sandbox has no real 5GB+ GGUF weights and no GPU
(`detect_device()` genuinely reports ~3.9GB combined RAM, below the 5GB
floor), so the disk-detection LOGIC was verified with a synthetic
sparse file of the right size and a monkeypatched device profile:

```
fake model present_on_disk: True 4.84 GB
status after init (should be 'downloaded'): downloaded True 4.84
status after autostart() attempt: not_installed | error: No module named 'llama_cpp' — run: python install.py --llm
load_now() result: {'ok': False, 'status': 'not_installed', 'reason': "No module named 'llama_cpp'"}
```
This confirms: (1) a model of the right size on disk is correctly
detected as `"downloaded"` immediately on manager init — no click
required; (2) `autostart()` correctly attempts a load automatically;
(3) when the optional `llama-cpp-python` package genuinely isn't
installed (this sandbox's actual state), the failure is safe, logged,
and actionable — never a crash. A near-empty directory (simulating the
v4 `data/models/google--gemma-4-E2B-it-qat-q4_0-gguf` folder that
shipped with no real weight files) was also confirmed to correctly
report `present=False`.

**Not verified here (needs the user's machine):** the actual GGUF load
succeeding with real weights and a real GPU.

## Bug #23 — real TTC, verified via the live REST API

Full server started (`python main.py`), then a real HTTP session:
```
create_profile: 200 {'profile_id': 'dc5748f95e64', 'name': 'APITestWriter1'}
enroll: 200 {n_references: 5, threshold: 0.5867, maturity: 5, ...}
verify: verdict=GENUINE score=0.7573 threshold=0.5867
  assets include: agreement_chart  (new v5 asset)
feedback (correct=true, add_as_reference=true):
  {'threshold': 0.5983, 'weights': {...}, 'maturity': 7,
   'promoted_to_reference': True, 'n_references': 6, 'feedback_count': 1}
profile n_refs after TTC promotion: 6
```
Confirms the query was correctly promoted into the profile (5→6
references) and the whole profile was relearned (threshold changed from
0.5867→0.5983, weights rebalanced) purely from the confirmed feedback —
the real "system trains itself from feedback" behavior the user asked
for, not just a threshold nudge.

The generated PDF report for this session was confirmed to be a valid
3-page PDF (`file` command: `PDF document, version 1.4, 3 page(s)`), and
the new `agreement.png` asset (bar chart of the query's score against
every reference + the prototype) was confirmed to render correctly.

## Known-not-done in this round

- No real-hardware GPU/LLM test (sandbox limitation, documented above).
- No large-scale benchmark beyond the 10-writer/20-verification sweep —
  same caveat v3/v4 already carried forward.
- CNN-SigNet's torch-enabled path remains unverified (pre-existing v4
  limitation, unrelated to this round's changes).

## Bug #24 — install.py pip-bootstrap repair + "run from anywhere," verified

**Reproducing the user's exact failure** (this sandbox is Linux, so the
identical failure CLASS — a venv that exists but has no working pip — was
reproduced directly rather than the Windows-specific trigger for it):
```
$ python3 -m venv --without-pip brokenvenv
$ brokenvenv/bin/python -m pip --version
brokenvenv/bin/python: No module named pip
```
This is the same end state the user's `.venv\Scripts\python.exe` was in.

**Confirming the fix repairs it**, calling `install.py`'s new functions
directly against that broken venv:
```
has pip before: False
    [WARN] This virtual environment's pip did not install correctly ...
    [OK] Repaired via ensurepip.
bootstrap result: True
has pip after: True
```

**Full clean-checkout install, end to end** (`python3 install.py`, no
`--check`, from a copy of the repo with no `.venv` and no `data/`):
- Created `.venv`, verified pip automatically, installed all 37 packages
  from `requirements.txt` (numpy, opencv-python-headless, scikit-image,
  scipy, fastapi, uvicorn, pydantic, reportlab, psutil, rich, and their
  transitive dependencies) with `Successfully installed ...` and no
  errors.
- Re-ran `python3 install.py --check` immediately after: reused the
  existing `.venv` (no re-download), and correctly reported real
  hardware (`ram_gb: 3.9`) now that `psutil` was actually installed —
  confirming the installer is idempotent.

**"Runs from any file/name it's in," verified three independent ways:**
1. Started `main.py` via its full absolute path while the process's
   working directory was `/` (completely unrelated to the project) —
   web UI (`/`), static assets (`/app.js`), and the API
   (`/api/llm/status`) all served correctly (HTTP 200).
2. Renamed the entire project folder to `totally_renamed_project_XYZ` in
   a different location than the original — server started correctly,
   and a full `create_profile` → `enroll` (5 CEDAR images) → `verify` →
   `feedback` (with `add_as_reference`) round-trip against the renamed
   instance produced byte-identical results to the same test against the
   original folder name (`verdict: GENUINE score: 0.7573`, promoted to 6
   references after feedback) — confirming nothing anywhere depends on
   the literal string "signxai" or a specific install path.
3. Invoked `run.sh` (not `main.py` directly) with the process's working
   directory set to `/` — before the fix this would have failed to find
   `.venv/bin/python` (a bare relative path); after the fix (`cd` into
   the script's own directory via `dirname "${BASH_SOURCE[0]}"` first)
   it correctly found and launched the app (HTTP 200 on
   `/api/llm/status`).

**Also confirmed, by grepping the entire tree:** no file uses
`os.getcwd()`, `Path.cwd()`, or a hardcoded `signxai/`-prefixed path
anywhere — every path in the codebase is built from
`Path(__file__).resolve().parent` (or derived from `app/config.py`'s
`ROOT_DIR`, which is itself `__file__`-based), which is what makes all
three tests above pass.

**Not verified in this round (sandbox has no Windows):** `run.bat`/
`run_cli.bat`'s `%~dp0` fix, and the real Windows trigger for the
original pip bug — see Section 7 of `PROJECT_MEMORY_v8.md` for this
logged as an open item, not silently assumed fixed.

## Bug #25 — LLM wheel-install rewrite (`--extra-index-url` package index), verified as far as this sandbox allows

**How the problem was found:** re-checked the v4 "verified end-to-end"
LLM install claim against fresh, current sources — the project's own
PyPI page, ReadTheDocs, and GitHub README, plus several third-party
wheel mirrors — while responding to a request to validate Windows/macOS
behavior. All independently showed real wheel filenames tagged
`cp310`/`cp311`/`cp312`/etc (compiled-extension ABI tags), never the
`py3-none` tag `install.py` had assumed. Also found the upstream
release-tag scheme had moved to per-backend suffixes (`v0.3.35-metal`,
`v0.3.35-cu121`, ...) rather than one tag holding a full asset matrix.

**Confirming this sandbox's own network restriction directly:**
```
$ curl -sI https://abetlen.github.io/llama-cpp-python/whl/cpu/
HTTP/2 403
x-deny-reason: host_not_allowed
```
This confirms the index domain is blocked from THIS development sandbox
specifically — expected and documented in this environment's own network
configuration, not a finding about real user machines.

**Real end-to-end run of the fixed `install.py --llm`:**
```
    [OK] Installing huggingface_hub
    Linux: using the official 'cpu' prebuilt-wheel index (no compiler needed).

>>> Installing llama-cpp-python (prebuilt 'cpu' wheel index)
    $ .../python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
Looking in indexes: https://pypi.org/simple, https://abetlen.github.io/llama-cpp-python/whl/cpu
Collecting llama-cpp-python
  Using cached llama_cpp_python-0.3.35.tar.gz (74.9 MB)
  ...
Building wheels for collected packages: llama-cpp-python
  Building wheel for llama-cpp-python (pyproject.toml): started
  Building wheel for llama-cpp-python (pyproject.toml): still running...
```
Correct backend selected (`cpu`), correct command constructed, and pip
transparently fell back to PyPI's source distribution once the blocked
index didn't respond — this is pip's normal, documented behavior for an
unreachable `--extra-index-url`, not a crash or hang in `install.py`
itself.

**Installed `cmake` (previously missing from this sandbox) to let the
from-source fallback actually attempt a real compile**, and confirmed
via `ps aux` real compilation commands succeeding:
```
gcc ... -c .../ggml-cpu.c -o .../ggml-cpu.c.o          [completed]
g++ ... -shared -o bin/libggml-base.so.0.20.0 ...      [completed]
```
The build was still actively compiling (further `cc1` processes running)
when this environment's own turn-based process lifecycle ended it —
background processes here do not persist across conversation turns, so a
multi-minute C++ compile cannot be watched to completion within one
turn. This is an environment limitation, not evidence of a code failure;
every step observed before that point was progressing correctly.

**Honest summary:** the fix is correct by construction (uses the
project's own officially-documented install command verbatim, no
custom filename-guessing left in this codebase at all), and every piece
of it that COULD be exercised in this sandbox was exercised and behaved
correctly. The one thing that could not be verified here is a fully
completed install (either via the index, or via a finished source
build) — both due to this specific sandbox's restrictions, not a known
or suspected defect. Flagged plainly in `PROJECT_MEMORY_v8.md` Section 7
as the top item for the user to confirm on a real machine.

## Bug #26 — crop-boundary robustness (recognition/position sensitivity), verified with real reproductions

**Reproducing the reported bug directly**, before attempting any fix —
took a real CEDAR signature and added a tiny 2px synthetic stray line far
from the real ink (simulating a scan artifact or trailing flourish only
captured in one photo):
```
Clean image bbox (in cropped coords): (44, 65, 575, 307)
With-stray-line bbox: (44, 65, 581, 346)

Clean aspect ratio: 2.262672811059908
With-tail aspect ratio: 1.9478260869565218      <- ~14% shift from one tiny mark
Clean centroid: 0.386, 0.367
With-tail centroid: 0.392, 0.318                 <- meaningfully shifted
```
This confirms the user's report precisely: something totally irrelevant
to the actual handwriting was measurably changing how the signature was
judged.

**After the fix** (percentile-trimmed crop + connected-component outlier
filter), same exact test:
```
Clean image bbox: (48, 69, 571, 298)
With-stray-line bbox: (47, 69, 575, 302)

Clean aspect ratio: 2.28110599078341
With-tail aspect ratio: 2.2660550458715596      <- under 1% shift now
Clean centroid: 0.384, 0.375
With-tail centroid: 0.383, 0.368                 <- negligible shift
```

**A second, harder stress test** (a small isolated dot in a far corner,
not a physically-connected tail) initially defeated the percentile-only
fix (aspect ratio still shifted 2.28 -> 1.77) — this is what led to
adding the connected-component outlier filter as a SECOND, complementary
mechanism. After both fixes together:
```
aspect ratio clean vs with-dot: 0.6666... vs 0.6666...   (identical)
centroid clean vs with-dot: (0.485, 0.418) vs (0.485, 0.418)  (identical)
```

**Two design flaws in the connected-component filter caught by testing
before shipping** (not assumed correct from the design):
- First version anchored on the single largest component + a fixed
  margin. On a real CEDAR image with 6 connected components (separate
  letters/strokes spread across the signature's full width), this kept
  only 2425 of 5803 real ink pixels (56% loss) — it had thrown away a
  legitimate ~40%-of-total letter group just because it wasn't the
  single largest piece.
- Second version tried adaptive-kernel dilation to bridge nearby strokes
  before taking the largest blob. Still wrong on the same test image —
  two legitimately separate letter groups had a bigger gap between them
  (a real feature of that particular handwriting) than a synthetic stray
  mark did from the page edge in the other test, so dilation-based
  clustering discarded real ink again (3254 of 5803 pixels kept).
- Final version (never discard a "substantial" component regardless of
  distance; only ever discard small AND far components) fixed both:
  zero ink pixels lost on the real 6-component test image, AND the
  synthetic-artifact tests above both pass cleanly.

**Sanity check across clean signatures** (5 different real writers, no
synthetic artifacts) — ink-pixel sums after the fix were byte-identical
to the original pre-fix code's output, confirming zero regression on
ordinary signatures:
```
[1323727, 1445348, 1691259, 2450605, 1416202]   <- identical before/after
```

**Full 10-writer sweep re-run after the fix** (same methodology as
Bug #21's sweep in PROJECT_MEMORY_v8.md):
```
                        before Bug #26   after Bug #26
genuine_correct:              5/10            5/10
genuine_wrong:                 0/10            0/10
forged_correct:                0/10            0/10
forged_wrong:                   3/10            0/10   <- real improvement
forged_inconclusive:            7/10           10/10
```

## Bug #27 — heatmap and rectangle-detector visual fixes, verified against real generated cases

**Before/after heatmap**, generated from the exact same real case
(skilled forgery vs. genuine references, CEDAR writer 1) — the old
version's output (not shown here, described directly in the user's
uploaded evidence) rendered as a near-solid blue-tinted rectangle over
the whole image, because JET's low end (near-zero difference, i.e. most
of any signature photo's blank background) is a saturated dark blue, not
white. The new version, same case, confirmed visually: clean white
background, color appears only along the actual ink strokes where the
two signatures genuinely diverge, with hue/intensity varying by how much
— a real heatmap by the definition the user provided.

**Before/after annotated regions**, same case — the old rigid-grid
version is what the user's own uploaded PDF showed: overlapping labels
("R1: Aspect Ru moment 1" running into "R2 Hu moment 1"). The new
contour-based version was iterated three times against the same real
case until clean:
1. First attempt: labels correctly avoided the WORST collisions but
   still ran off the canvas's right edge ("2. Region 2 divergence: Hu
   mome" truncated by the image boundary) — X position wasn't being
   clamped, only Y was.
2. Second attempt (after clamping X): boxes still overlapped each other
   substantially (one real divergence blob was being reported as 2-3
   separate "top regions" since morphological closing didn't fully merge
   it) — labels stacked but no longer ran off-canvas.
3. Third attempt (added greedy non-max suppression at IoU >= 0.2):
   confirmed visually — 3 genuinely distinct, non-overlapping boxes, no
   label collisions, each label fully on-canvas.

## Bug #28 — LLM Windows install fix, verified as far as this sandbox allows

**Reproducing the exact reported failure mode** (pip silently choosing a
source distribution over a wheel) — confirmed the OLD code's plain
`pip install ... --extra-index-url ...` command is exactly what the
user's log shows doing this; the mechanism (pip's version resolution
doesn't strictly prefer an older wheel-available version over a newer
sdist-only one) is a documented, general pip behavior, not specific to
this project's index.

**Confirming the new wheel-only fail-fast behavior works correctly**, a
real end-to-end run of the fixed `install_llm_backend()` in this sandbox
(which also has no reachable wheel for this exact platform/Python
combination — a good stand-in for "no matching wheel published yet"):
```
$ .../python -m pip install --only-binary=:all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
ERROR: Could not find a version that satisfies the requirement llama-cpp-python (from versions: none)
ERROR: No matching distribution found for llama-cpp-python
    [no matching prebuilt wheel] Installing llama-cpp-python (prebuilt 'cpu' wheel index)

    Falling back to a from-source build (no matching prebuilt wheel was found for this exact Python version/platform).
```
This is the key confirmation: failure now takes SECONDS with a clear
message, not the 10+ minutes the user's log shows before hitting the
confusing CMake error.

**Confirming MSVC auto-detection fails gracefully when no VS install
exists** (this sandbox, real test — not simulated):
```
vswhere found: None
vcvars64 found: None
msvc env found: False

--- testing _install_llm_from_source() with no compiler ---
    No C++ compiler (cl.exe) found on PATH -- looking for a Visual Studio
    Build Tools install to load automatically...
    [FATAL] No Visual Studio C++ Build Tools installation could be found
    on this machine, so a from-source build is not possible right now.
    Install 'Build Tools for Visual Studio' ...
result: False
```
Confirms the code path executes correctly end-to-end (locates nothing,
reports clearly, returns False without crashing or attempting a doomed
compile) — but the actual `vswhere`/`vcvars64.bat` auto-loading against a
REAL Visual Studio installation has not been exercised, since this
sandbox has none. **Top item for the user to verify for real.**

## Bug #29 — web UI restyle and feedback/remark flow, verified structurally

**Remark storage, verified end-to-end against a live server** (real HTTP
round-trip, not just code review):
```
feedback response: {'threshold': 0.697, ..., 'promoted_to_reference': False, ...}

stored feedback entries:
[{
  "case_id": "56888e2c8418", "correct": true, "promoted_to_reference": false,
  "remark": "the tail flourish was cut off in this photo",
  "at": "2026-08-23T08:13:30+0000"
}]

REMARK STORAGE VERIFIED OK
```

**Structural verification of the restyled frontend** (this sandbox has
no reachable headless-browser download source — `playwright install`
fails with the browser-download host not in the network allowlist — so
actual rendering could not be screenshotted):
- Every CSS class referenced anywhere in `index.html`/`app.js`
  cross-checked programmatically against what's actually defined in
  `style.css` — zero real mismatches (a few grep false-positives from
  JS template-literal fragments, manually confirmed not to be class
  names).
- Every static `getElementById()` call in `app.js` cross-checked against
  literal `id="..."` attributes in `index.html` — zero mismatches
  (dynamic, per-case IDs like `fbCorrect-${caseId}` are created and
  looked up within the same JS scope, which is correct and doesn't need
  a static HTML counterpart).
- `node --check web/app.js` confirms the rewritten frontend script is
  syntactically valid JavaScript.
- Full functional round-trip (enroll -> verify -> render result HTML ->
  submit feedback with remark) run against a live server with real
  signature images, confirming the new section structure's underlying
  data flow works correctly end-to-end.

**Not done:** an actual visual screenshot confirming the Browserbase
restyle renders as intended in a real browser. Flagged plainly as the
top open item for Bug #29 in `PROJECT_MEMORY_v8.md` Section 7.

## Bug #30 — LLM DLL-load verification, verified as far as this sandbox allows

**The good news buried in the bug report:** the user's own log confirms
Bug #28's fix works correctly on real Windows hardware —
```
Windows: NVIDIA GPU detected -- using the official 'cu121' CUDA prebuilt-wheel index.
Downloading https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl (448.3 MB)
...
Successfully installed ... llama-cpp-python-0.3.4
```
A real, correctly-selected, correctly-sized CUDA wheel downloaded and
installed. The NEW failure is one step later: the installed package's
native library wouldn't load.

**Reproducing the detection logic** (this sandbox has no `llama_cpp` at
all, a reasonable stand-in for "installed but broken"):
```
ok, err = install._verify_llama_cpp_importable()
ok: False
err: "ModuleNotFoundError: No module named 'llama_cpp'"
```
Confirms the subprocess-import check correctly identifies a non-working
install.

**Confirming the manager.py hint-matching fires on the user's exact
error text:**
```python
msg = "Failed to load shared library 'D:\\signxai\\...\\llama.dll': Could not find module (or one of its dependencies)."
hint triggered: True
```

**Not verified:** an actual "wheel installs, DLL won't load, falls back
to CPU automatically" run on real Windows/CUDA hardware — this sandbox
has no GPU. Flagged as the top open item for Bug #30.

## Bug #31 — zonal (local) signature analysis, extensively tested and iterated

**Reproducing the user's exact scenario** (a controllable zigzag count):
```
GENUINE query (3 zigzags) vs reference (3 zigzags):
  aggregate_score: 0.9874   flagged zones: 0

FORGED query (2 zigzags) vs reference (3 zigzags):
  aggregate_score: 0.8899   flagged zones: 2
    FLAGGED: row 0, col 2 -- q_corners=7  ref_corners=13
    FLAGGED: row 0, col 3 -- q_corners=6  ref_corners=13
```
Directly confirms the module catches a missing zigzag stroke and reports
exactly where and by how much.

**A harder, more realistic test** — the same zigzag detail squeezed into
only ~8% of a much larger, denser signature (closer to how a subtle real
forgery detail gets diluted):
```
GLOBAL HOG-cosine genuine vs ref:              0.9871
GLOBAL HOG-cosine forged(2zz, small) vs ref:   0.9949   <- WRONG DIRECTION
```
The global metric alone not only failed to distinguish the cases, it
scored the forgery HIGHER than the genuine sample — direct, concrete
evidence of the dilution problem this module exists to address.

**Design flaw #1, caught and fixed during this same test:** the first
corner descriptor (contour polygon approximation on filled ink) blended
a real 51-vs-72 corner mismatch into an innocuous-looking 0.908 zone
score because it shared a zone with a large chunk of unrelated dense
writing — fixed by keying the FLAG condition directly to `corner_agree`
rather than the blended `zone_score`.

**Design flaw #2, caught on real CEDAR data (not just synthetic tests):**
```
query skeleton: 5 connected components, total skeleton pixels = 1009
largest component size: 446
ordered path length: 41        <- only 9% of the component traced!
TOTAL corner points: query = 0   ref = 21
```
The ordered-path-walk corner detector silently failed on real, branchy
cursive handwriting, producing a systematic "0 corners" false reading.
Confirmed at scale — this bug alone produced 58/60 false-positive zone
flags across a 10-writer genuine-vs-genuine CEDAR sweep before being
replaced with the shipped eigenvalue/structure-tensor method.

**Final threshold tuning, real 10-writer sweep, shown in full because the
iteration itself is informative:**
| flagging rule | genuine-vs-genuine false positives (60 zones total) |
|---|---|
| `corner_agree < 0.85` (relative only) | 2 (writer 1 case shown in detail) |
| `corner_agree < 0.7` (relative only) | 0 for writer 1, but 34 total across 10 writers |
| `corner_agree < 0.6 AND corner_mismatch >= 5` (combined) | **0 total across 10 writers** |

Final rule re-checked against the same writer 1 forgery case: still
correctly flags 6 zones with clear, large absolute corner gaps (0 vs 9,
2 vs 13, 2 vs 13, ...).

**Aggregate-score discrimination, full 10-writer sweep:**
```
writer  gen_agg   gen_flags  forg_agg  forg_flags  separated?
1       0.8839    0          0.6770    6           YES
2       0.8986    0          0.8159    6           YES
3       0.8161    6          0.7707    6           YES
4       0.7972    2          0.7892    1           YES
5       0.7771    1          0.8434    3           no
6       0.7649    4          0.8080    5           no
7       0.8983    3          0.7769    6           YES
8       0.7966    6          0.7293    4           YES
9       0.8780    6          0.7985    6           YES
10      0.7724    6          0.7310    6           YES

mean genuine aggregate: 0.8283
mean forged aggregate:  0.7740
genuine > forged in 8/10 writers
```

**Full engine integration, real end-to-end test** (5 enrolled references,
skilled forgery query): `components["zonal"] = 0.7466`, correctly
contributing to the fused score; `case["zonal_analysis"]` populated with
6 flagged zones; `zone_grid.png` generated and visually confirmed to show
clean adaptive zone boundaries, color-coded (green/amber/red) by local
agreement, with the two most divergent zones showing "0/5" and similar
corner-count labels directly on the image; PDF report grew from 3 to 4
pages to include it; CLI zonal-findings table renders correctly.

**Honest regression check:** re-ran the standard 10-writer sweep after
integrating `zonal` into the fused score:
```
                    before Bug #31   after Bug #31
genuine_correct:         5/10             4/10
genuine_wrong:           0/10             0/10
forged_wrong:            0/10             0/10
```
A small, real trade-off from adding a new signal — reported plainly, not
hidden by re-tuning the weight to produce a better-looking number (a
0.06-weight variant was tested and produced the IDENTICAL result, since
`_relearn()`'s threshold fitting recalibrates the whole system together
whenever any component changes).

## Bug #32 — GPU offload honesty, verified as far as this sandbox allows

**Regex parser against real-world log patterns:**
```
CUDA success pattern ("offloaded 32/32 layers to GPU"):  True
CPU-only fallback pattern ("offloaded 0/29 layers to GPU"): False
unrecognized text: None
```

**FD-level capture confirmed to intercept native-level output** (a
Python-level `contextlib.redirect_stdout` would NOT see this):
```python
os.write(1, b'hello from native stdout\n')  # bypasses sys.stdout entirely
# _capture_native_output() correctly captured: 'hello from native stdout\n'
```

**Not verified:** an actual "load on real CUDA hardware, confirm the
status page agrees" run — no GPU in this sandbox. Top open item.

## Bug #33 — construction-consistency dilution fix, extensively tested against real user photos

**Reproducing the user's exact reported scenario** with their own 4
uploaded signature photos (1.jpeg/2.jpeg/3.jpeg = same person's genuine
signature with natural variation; 4.jpeg = visibly different signature
structure):

Before any Bug #33 fix:
```
query=4.jpeg (visually different): score=0.6131  verdict=INCONCLUSIVE
  iou=0.1475  hu=0.5734  dtw=0.4861  scalar_profile=0.3039
  (but) ssim=0.8087  hog=0.8628  lbp=1.0000   <- diluting the signal
```

After splitting out `construction` as its own component:
```
query=4.jpeg: score=0.5430  verdict=INCONCLUSIVE (closer, one individual
              reference now voted FORGED)  construction=0.1412
```

After also fixing topology-count stability (morphological closing for
photographed-signature lighting noise) AND rebalancing weights toward
what real evidence showed was discriminating:
```
query=1.jpeg: verdict=GENUINE  score=0.7959
query=2.jpeg: verdict=GENUINE  score=0.7671
query=4.jpeg: verdict=FORGED   score=0.4584  (threshold 0.5481)
  components: ssim=0.8054 iou=0.1636 hog=... construction=0.13-ish
```
**4.jpeg moved from INCONCLUSIVE all the way to a correctly, clearly
FORGED verdict.**

**Debugging the self-verification puzzle** (re-verifying an enrolled
reference against its own profile showed only moderate construction
scores, not near-perfect) — traced to a real, non-bug mechanism, not
assumed:
```
3.jpeg's own construction scalars EXACTLY matched ref_2.png's stored
scalars (components=4, contours=4, endpoints=22, holes=6, junctions=76)
-- confirming feature extraction is fully deterministic, no instability
bug. The three enrolled references' own construction counts:
  ref_0: components=6   ref_1: components=7   ref_2: components=4
```
A real, substantial natural spread (4-7) across just 3 samples of the
same person's own genuine signature — small-sample statistics
(mean=5.67, std=1.25 from only 3 points) correctly reflect this real
variability; even the "self" sample shows a non-trivial z-score purely
because the profile mean is dragged by all 3 samples including
naturally-varying ones. This is accurate behavior given only 3 samples,
not a bug — confirmed by hand-computing the exact `exp(-|z|)` formula
against the observed z-scores and matching the reported 0.3358 score
precisely.

**Full 10-writer CEDAR sweep, before/after this round's combined
changes:**
```
                    before      after
genuine_correct:     4/10        4/10
genuine_wrong:       0/10        0/10
forged_correct:      0/10        2/10
forged_wrong:        0/10        1/10
```
Investigated the one new false-accept (writer 5) directly rather than
just noting the number:
```
verdict: GENUINE  score: 0.6312  threshold: 0.5235
iou=0.1411  hu=0.3378   <- correctly signaled real mismatch
construction=0.8018     <- but this forgery replicated pen-lift/loop
                           counts closely, staying high
reference_agreement: 1.0 (all 5 individual refs unanimously voted GENUINE)
```
Honest conclusion: this specific skilled forgery failed on shape/pixel
fidelity but succeeded on construction fidelity — the mirror image of
the 4.jpeg case. No further weight-tuning was applied to chase this one
case, since doing so on a single example would likely just move the
failure elsewhere rather than fix it — documented as a real, disclosed
trade-off rather than hidden or over-corrected.
