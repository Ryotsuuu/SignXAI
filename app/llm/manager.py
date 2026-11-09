"""Vision-LLM manager: device-aware GGUF model selection, Hugging Face
download, lazy loading and generation via llama-cpp-python.

BUG #1 FIX (PROJECT_MEMORY_v8.md Section 3): replaced the previous
transformers+bitsandbytes(4-bit, CUDA-only) backend with llama-cpp-python,
which runs the SAME GGUF file on CPU, NVIDIA CUDA, Apple Metal, or Vulkan
through one code path — this is what makes the app truly OS-agnostic. The
manager NEVER crashes the host application — any failure moves it to
``error`` status and the report generator falls back to the deterministic
template engine. llama_cpp is imported inside functions only, so the core
app runs without it installed (see PROJECT_MEMORY_v8.md process rule #5).
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import MODELS_DIR
from ..device import (DeviceProfile, MODEL_ID, detect_device,
                      model_present_on_disk)

log = logging.getLogger("signxai.llm")


def _capture_native_output(fn):
    """Run fn() while capturing anything written to the process's REAL
    stdout/stderr file descriptors (fd 1/2), not just Python's
    sys.stdout/sys.stderr.

    BUG #32 FIX (PROJECT_MEMORY_v8.md, user-reported: "LLM: ready" shown
    in the UI, but a CPU/RAM spike instead of GPU usage when generating a
    report): llama.cpp's C++ core prints its own load-time diagnostics --
    including whether GPU layers were ACTUALLY offloaded -- directly
    through native C-level stdout/stderr, which bypasses Python's
    sys.stdout/sys.stderr entirely. A plain contextlib.redirect_stdout
    does not see this output at all; only a file-descriptor-level
    redirect does. This is exactly why "set n_gpu_layers=-1" is not
    sufficient to know GPU offload happened: if the installed
    llama-cpp-python build doesn't actually have CUDA support compiled in
    (e.g. a CPU-only backend installed as an automatic fallback -- see
    Bug #30), that parameter is SILENTLY IGNORED by the native library,
    with no Python-level exception or return value indicating this -- a
    well-documented, common llama-cpp-python gotcha (confirmed against
    multiple independent real-world reports of the exact same
    silent-CPU-fallback behavior).
    """
    stdout_fd, stderr_fd = 1, 2
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), stdout_fd)
        os.dup2(tmp.fileno(), stderr_fd)
        try:
            result = fn()
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", errors="replace")
    return result, captured


def _path_to_file_uri(path: str) -> str:
    """Convert a plain filesystem path (any OS) into a proper ``file://``
    URI, using ``pathlib.Path.as_uri()`` instead of naive string
    concatenation.

    v7.2 FIX (Bug #48, PROJECT_MEMORY_v8.md, user-reported: the report
    generation crash changed from the PIL-Image type mismatch (Bug #45,
    now fixed) to a NEW failure — `<urlopen error [WinError 3] The
    system cannot find the path specified: ''>` — the moment the user
    actually re-ran it on their real Windows machine): the previous fix
    for Bug #45 was correct about WHAT to pass (a path string, not a PIL
    object) but wrong about HOW to turn that path into a URL. It used
    `f"file://{path}"`, simple string concatenation, which happens to
    work by accident on POSIX (`file:///home/user/x.png`) but is not a
    valid URI on Windows: a real Windows path like
    `D:\\signxai\\data\\reports\\...\\query.png` has backslashes (not
    valid URI path separators) and a drive letter where a URI expects a
    host — naive concatenation produces `file://D:\\signxai\\...`, which
    urllib's file-URL opener parses as host `"D:"` (further mangled by
    the backslashes) and an EMPTY path, exactly matching the reported
    `WinError 3` on `''`. This is precisely the kind of platform
    assumption this project's own process rules (`PROJECT_MEMORY_v8.md`
    Section 6) exist to catch — verified this time by testing the
    actual byte output of the conversion for a realistic Windows-style
    path, not just eyeballing the string, since this sandbox cannot run
    a real Windows process to confirm end-to-end.

    `pathlib.Path(path).resolve().as_uri()` is the standard library's own
    OS-aware conversion (it is exactly what `as_uri()` exists for) --
    it correctly produces `file:///D:/signxai/data/.../query.png` on
    Windows and `file:///home/user/.../query.png` on POSIX, handling
    drive letters, backslash-to-forward-slash conversion, and percent-
    encoding of special characters, none of which naive string
    formatting does.
    """
    return Path(path).resolve().as_uri()


class _LoadTimeout(Exception):
    """Raised when the blocking native model-construction call does not
    finish within LLM_LOAD_TIMEOUT_SECONDS. See `_run_with_timeout` and
    `VisionLLMManager._load` for the full reasoning (Bug #43)."""


def _run_with_timeout(fn, timeout: float):
    """Run ``fn`` (a blocking, possibly-native call) in a background
    thread and wait at most ``timeout`` seconds for it.

    v7 FIX (Bug #43, PROJECT_MEMORY_v8.md, user-reported: "LLM: loading
    signal for more than 2min ... not even in CPU the fallback method"):
    the previous code called the blocking `Llama(...)` constructor
    directly, in-line, with NO timeout anywhere in the call chain. If
    that native call hangs -- which is a documented, real-world
    llama-cpp-python failure mode on some driver/VRAM/mmproj-version
    combinations, distinct from the "silent CPU fallback" Bug #32
    already found -- there is no Python-level exception to catch: it
    is not failing, it is genuinely blocked, so `self._status` stays
    stuck at "loading" forever with nothing to tell the user WHY. This
    cannot be fixed by adding a try/except around the call (nothing is
    raised); it needs an actual wall-clock ceiling.

    A plain call cannot be forcibly killed from Python if it's blocked
    inside native (C/C++) code with the GIL held or a driver call in
    progress -- so this does not (and cannot) guarantee the underlying
    native call actually stops. What it DOES guarantee: the CALLER gets
    an answer within ``timeout`` seconds either way, converting an
    indefinite, unexplained hang into a concrete, actionable timeout
    error. If the background attempt does eventually finish on its own
    after the timeout fires, the (thread, result-box) pair is returned
    so a caller can opportunistically pick up a late success instead of
    wasting it -- see `_load()`'s use of ``self._pending_construct``.
    """
    box: dict = {}
    done = threading.Event()

    def _runner():
        try:
            box["value"] = fn()
        except BaseException as exc:                    # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_runner, daemon=True, name="signxai-llm-construct")
    t.start()
    finished = done.wait(timeout)
    return finished, box, t


def _parse_gpu_offload(captured: str):
    """Parse llama.cpp's own verbose load log for a real GPU-offload
    confirmation. Returns True/False on a definitive signal, or None if
    the log doesn't contain a recognizable pattern (e.g. a
    llama-cpp-python version that changed its log wording -- callers
    should treat None as "unknown", never silently assume True)."""
    low = captured.lower()
    m = re.search(r"offloaded (\d+)/(\d+) layers to gpu", low)
    if m:
        offloaded = int(m.group(1))
        return offloaded > 0
    if re.search(r"ggml_cuda_init|cuda0 buffer|using device.*cuda|metal.*ggml", low):
        return True
    if "n_gpu_layers = 0" in low or "offloading 0 repeating layers" in low:
        return False
    return None


# Generation parameters — pre-defined and centralised, per the project brief
# ("make a good token space and temp, with all important parameters
# pre-defined"). Low temperature: this is a forensic-report task where we
# want consistent, grounded narration of pre-computed metrics, not creative
# variance. See prompts.py for why the contract is strict JSON.
GEN_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 1500,   # v3: raised from 900 -- schema grew by 3 fields
    # v7 FIX (Bug #43, PROJECT_MEMORY_v8.md, user-reported: "LLM: loading
    # signal for more than 2min ... not even in CPU"): llama.cpp allocates
    # its ENTIRE KV cache upfront at model-construction time, sized by
    # n_ctx -- this happens DURING the `Llama(...)` call itself, before
    # any generation, so a larger n_ctx directly inflates load latency
    # (this is the well-documented reason llama.cpp construction time
    # scales with n_ctx, not with how much of it a given prompt actually
    # uses). Checked the actual prompt this app sends
    # (`prompts.py::build_user_prompt` + the repair-retry round trip):
    # the evidence JSON, system prompt, and schema together are well
    # under 2000 tokens even with the full metrics table and a retry
    # round -- 8192 was never needed for this app's own prompt, it was
    # inherited as a generic "big is safe" default. Halved to 4096
    # (still >2x headroom over the measured real usage) to cut
    # construction-time KV-cache allocation, and therefore load latency,
    # roughly in half on every backend (CPU and GPU alike) -- this is a
    # genuine load-time fix, independent of and in addition to the
    # timeout/VRAM-gating fixes below.
    "n_ctx": 4096,
}

# v7 (Bug #43, PROJECT_MEMORY_v8.md): hard wall-clock ceiling on the
# blocking `Llama(...)` construction call. Overridable via env var for a
# slower machine/disk. See `_load()`'s docstring for the full reasoning
# on why an unbounded blocking call was the actual "stuck on loading
# forever" bug, independent of which backend (CPU/GPU) was in use.
LLM_LOAD_TIMEOUT_SECONDS = float(os.environ.get("SIGNXAI_LLM_LOAD_TIMEOUT", "180"))


class VisionLLMManager:
    """Singleton-style manager; use :func:`get_manager`."""

    def __init__(self) -> None:
        self._device: Optional[DeviceProfile] = None
        # v5 (Bug #22, PROJECT_MEMORY_v8.md): status now includes "downloaded"
        # -- weights are present on disk but not yet loaded into memory.
        # disabled|not_installed|downloaded|downloading|loading|ready|error
        self._status = "unknown"
        self._error: Optional[str] = None
        self._progress = ""
        self._llm = None              # llama_cpp.Llama instance (multimodal)
        self._model_size_gb = 0.0
        # v6 (Bug #32, PROJECT_MEMORY_v8.md): True/False once a load has
        # actually captured a definitive GPU-offload signal from
        # llama.cpp's own native load log; None means "not yet loaded" OR
        # "loaded but the log didn't contain a recognizable pattern" --
        # both are honestly reported as unknown, never assumed True.
        self._gpu_offload_confirmed: Optional[bool] = None
        # v7.1 (Bug #46, PROJECT_MEMORY_v8.md, user-reported: "I want
        # explanation in -> AI Model section... show me its reasoning"):
        # `self._progress` gets overwritten to "ready" once loading
        # finishes, so the VRAM-gating decision (Bug #44) it briefly held
        # during loading was lost by the time anyone actually looked at
        # the status. This field persists that reasoning for the whole
        # lifetime of the loaded model, not just the loading window --
        # see `status()`'s `gpu_skip_reason` field.
        self._gpu_skip_reason: Optional[str] = None
        self._lock = threading.Lock()
        # v7 (Bug #43, PROJECT_MEMORY_v8.md): if a load attempt times out
        # (see `_load()`) but the underlying native call keeps running in
        # the background, this holds (thread, result-box) so a LATER
        # `_load()` call can pick up a late success instead of starting
        # a second, redundant construction on top of the still-running
        # first one.
        self._pending_construct: Optional[tuple] = None
        self.refresh_device()

    # ------------------------------------------------------------ status
    def refresh_device(self) -> DeviceProfile:
        """v5 (Bug #22, PROJECT_MEMORY_v8.md): this used to decide status from
        hardware alone, never checking whether the model was ALREADY on
        disk -- so a manually-copied model was invisible until a UI
        "download" click ran the (redundant, wasteful) full download
        again. Now checks the filesystem directly every call."""
        self._device = detect_device()
        present, size_gb = model_present_on_disk(self._device.model_id or MODEL_ID)
        self._model_size_gb = size_gb
        if not self._device.viable:
            self._status = "disabled"
        elif present and self._status in ("unknown", "not_installed", "disabled"):
            self._status = "downloaded"   # on disk, not yet loaded into memory
        elif not present and self._status in ("unknown", "disabled"):
            self._status = "not_installed"
        return self._device

    @property
    def device(self) -> DeviceProfile:
        return self._device or self.refresh_device()

    def status(self) -> dict:
        d = self.device
        # v6 (Bug #32, PROJECT_MEMORY_v8.md, user-reported): this used to
        # be `bool(self._llm is not None and (d.has_cuda or d.has_mps))`
        # -- HARDWARE presence, not whether the loaded backend actually
        # used it. A CPU-only build (e.g. a Bug #30 automatic fallback)
        # would report "gpu_offload_active: true" on GPU hardware while
        # silently running entirely on CPU -- exactly what the user saw
        # ("LLM: ready" + a CPU/RAM spike, no GPU usage). Now keyed to
        # the actual confirmation captured from llama.cpp's own load log.
        gpu_active = bool(self._llm is not None and self._gpu_offload_confirmed is True)
        gpu_unconfirmed = bool(self._llm is not None and self._gpu_offload_confirmed is None
                               and (d.has_cuda or d.has_mps))
        present, size_gb = model_present_on_disk(d.model_id or MODEL_ID)
        return {
            "status": self._status,
            "error": self._error,
            "progress": self._progress,
            "model_id": d.model_id,
            "notes": d.notes,
            "backend": "llama-cpp-python (GGUF)",
            "gpu_detected": d.has_cuda or d.has_mps,
            "gpu_name": d.gpu_name if d.has_cuda else ("Apple Metal" if d.has_mps else None),
            "gpu_vram_gb": d.vram_gb,
            "all_gpus_detected": d.all_gpus,   # every NVIDIA GPU found, for multi-GPU visibility
            "gpu_offload_active": gpu_active,  # True only with a CONFIRMED offload from llama.cpp's own load log
            # v6 (Bug #32): distinct from `gpu_offload_active` -- True
            # when GPU hardware exists and the model is loaded, but the
            # load log didn't give a definitive answer either way (rare;
            # usually means llama.cpp changed its log wording). Lets the
            # UI say "unconfirmed" instead of silently claiming success.
            "gpu_offload_unconfirmed": gpu_unconfirmed,
            # v5 (Bug #22, PROJECT_MEMORY_v8.md): surfaced so the UI can show
            # "model found on disk (X.XGB)" independently of whether it has
            # been loaded into memory yet -- makes a manually-copied model
            # visible even before autostart/load finishes.
            "model_present_on_disk": present,
            "model_size_on_disk_gb": size_gb,
            # v7.1 (Bug #46, PROJECT_MEMORY_v8.md, user-reported): persists
            # across the whole loaded lifetime, not just during loading --
            # None when GPU was never a candidate to skip (no GPU detected,
            # or GPU was used successfully).
            "gpu_skip_reason": self._gpu_skip_reason,
            "n_ctx": GEN_PARAMS["n_ctx"],
            "load_timeout_seconds": LLM_LOAD_TIMEOUT_SECONDS,
        }

    def _model_dir(self):
        assert self.device.model_id is not None
        return MODELS_DIR / self.device.model_id.replace("/", "--")

    # ------------------------------------------------------------ download
    def download_blocking(self) -> dict:
        """BUG #17 FIX (v3, PROJECT_MEMORY_v8.md): synchronous download, no
        background thread. install.py --llm was calling the async
        download() (a daemon thread, designed for the web server which
        stays alive for the thread's whole lifetime) and then polling for
        completion. Race condition: if the poll loop's first check landed
        before the daemon thread had actually started and set status to
        "downloading", the loop saw the pre-download status, exited
        immediately, and install.py's process then terminated — killing
        the daemon thread mid-download, silently. The web UI never hit
        this because the server process (and thus the thread) stays alive
        indefinitely. This method has no such race: it blocks the calling
        process until the download genuinely finishes or fails, which is
        exactly right for a one-shot CLI script.
        """
        if not self.device.viable:
            return {"ok": False, "reason": "hardware below minimum requirement — LLM disabled"}
        self._status = "downloading"
        self._progress = "starting"
        try:
            from huggingface_hub import snapshot_download
            target = self._model_dir()
            target.mkdir(parents=True, exist_ok=True)
            log.info("Downloading %s -> %s", self.device.model_id, target)
            snapshot_download(repo_id=self.device.model_id, local_dir=str(target))
            self._progress = "done"
            # v5 (Bug #22, PROJECT_MEMORY_v8.md): this used to be set back to
            # "not_installed" after a SUCCESSFUL download, which was simply
            # wrong (the weights ARE installed now) and meant the model
            # would never be loaded until some other trigger changed the
            # status. "downloaded" is both accurate and is what
            # autostart()/report.py's gate now key off to load it.
            self._status = "downloaded"
            self._error = None
            return {"ok": True, "reason": "download complete"}
        except Exception as exc:
            log.exception("Model download failed")
            self._status = "error"
            self._error = str(exc)
            return {"ok": False, "reason": str(exc)}

    # --------------------------------------------------------------- load
    def _find_files(self, src) -> tuple[str, str]:
        """Locate the main GGUF and its mmproj file inside the downloaded repo."""
        gguf = mmproj = None
        for f in sorted(src.glob("*.gguf")):
            name = f.name.lower()
            if "mmproj" in name:
                mmproj = str(f)
            elif gguf is None:
                gguf = str(f)
        if gguf is None:
            raise FileNotFoundError(f"No .gguf weights found in {src}")
        if mmproj is None:
            raise FileNotFoundError(
                f"No mmproj (vision projector) .gguf found in {src} — "
                "the model repo may not have finished downloading, or vision "
                "support was not published for this checkpoint yet.")
        return gguf, mmproj

    def _load(self) -> None:
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            # v7 (Bug #43, PROJECT_MEMORY_v8.md): if an earlier timed-out
            # attempt has since finished successfully in the background,
            # use it instead of starting a brand new (redundant, and on a
            # tight-VRAM machine possibly conflicting) construction.
            if self._pending_construct is not None:
                pend_thread, pend_box = self._pending_construct
                if not pend_thread.is_alive() and "value" in pend_box:
                    self._llm = pend_box["value"]
                    self._status = "ready"
                    self._pending_construct = None
                    log.info("Vision LLM: a previously-timed-out load finished "
                            "in the background and is now ready.")
                    return
            self._status = "loading"
            self._progress = "starting"
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                src = self._model_dir()
                if not src.exists() or not any(src.glob("*.gguf")):
                    raise FileNotFoundError(
                        f"Model not downloaded yet: {src}. Call download() first "
                        "or run: python install.py --llm")
                gguf_path, mmproj_path = self._find_files(src)
                self._progress = "locating GGUF files -- done"
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)

                # v7 FIX (Bug #44, PROJECT_MEMORY_v8.md, user-reported:
                # "It is not working in GPU and not even in CPU"): the
                # previous logic attempted full GPU offload
                # (n_gpu_layers=-1) whenever ANY CUDA device was detected,
                # with no regard for whether that device actually has
                # enough VRAM for this model. Checked against the user's
                # own real numbers: an RTX 4050 Laptop (6.0GB VRAM) trying
                # to hold a model whose GGUF+mmproj files alone total
                # ~4.5-5GB on disk, PLUS this app's own KV cache (sized by
                # n_ctx, non-trivial even at the now-reduced 4096 -- see
                # GEN_PARAMS above) PLUS the runtime's own overhead,
                # comfortably exceeds 6GB. The result on a real
                # over-committed GPU is not always a clean "out of
                # memory" exception -- it can also manifest as severe
                # driver-level paging/thrashing that looks exactly like
                # an indefinite hang, which independently explains the
                # user's report even before the Bug #43 timeout fix
                # existed. Fixed by estimating the real memory
                # requirement from the model's actual on-disk size (a
                # reasonable proxy: a loaded GGUF's resident size tracks
                # its file size closely, quantisation format aside) plus
                # a documented KV-cache/overhead reserve, and skipping the
                # GPU attempt entirely -- going straight to CPU, with a
                # clear logged reason -- when the detected VRAM can't
                # plausibly hold it. This can never make GPU offload MORE
                # likely to fail than before; it only ever converts a
                # doomed, slow/hanging GPU attempt into an immediate,
                # working CPU load with an honest explanation of why.
                _KV_CACHE_OVERHEAD_GB = 1.5   # see docstring above
                _VRAM_SAFETY_FACTOR = 1.15    # 15% headroom over raw estimate
                present, model_gb = model_present_on_disk(self.device.model_id or MODEL_ID)
                required_gb = (model_gb + _KV_CACHE_OVERHEAD_GB) * _VRAM_SAFETY_FACTOR
                gpu_capable = self.device.has_cuda or self.device.has_mps
                # MPS (Apple unified memory) shares system RAM rather than
                # a separate VRAM pool -- already gated by
                # device.py's MIN_VIABLE_TOTAL_GB check, so only CUDA's
                # separate, fixed VRAM pool needs this extra headroom
                # check here.
                gpu_vram_sufficient = (
                    self.device.has_mps or
                    (self.device.has_cuda and self.device.vram_gb >= required_gb))
                n_gpu_layers = -1 if (gpu_capable and gpu_vram_sufficient) else 0
                if gpu_capable and not gpu_vram_sufficient:
                    self._gpu_skip_reason = (
                        f"GPU detected ({self.device.gpu_name}, "
                        f"{self.device.vram_gb:.1f}GB VRAM) but this model needs an "
                        f"estimated ~{required_gb:.1f}GB ({model_gb:.1f}GB model on disk + "
                        f"{_KV_CACHE_OVERHEAD_GB:.1f}GB KV-cache/runtime reserve, "
                        f"+{(_VRAM_SAFETY_FACTOR - 1) * 100:.0f}% safety margin) -- "
                        f"running on CPU instead to avoid an out-of-memory failure or "
                        f"driver-level paging that can look like an indefinite hang. "
                        f"A smaller context window or a smaller model tier would let "
                        f"GPU offload fit.")
                    log.warning(self._gpu_skip_reason)
                    self._progress = (
                        f"GPU present but insufficient VRAM ({self.device.vram_gb:.1f}GB "
                        f"< ~{required_gb:.1f}GB needed) -- using CPU")
                else:
                    self._gpu_skip_reason = None

                # v6 (Bug #32, PROJECT_MEMORY_v8.md, user-reported: "LLM:
                # ready" shown but a CPU/RAM spike instead of GPU usage):
                # `n_gpu_layers=-1` is silently IGNORED if the installed
                # llama-cpp-python build doesn't actually have CUDA/Metal
                # support compiled in (e.g. a CPU-only backend installed
                # automatically as a Bug #30 fallback) -- there is no
                # Python-level exception or return value that reveals
                # this. verbose=True (only for this internal call, output
                # captured and never shown to the user) lets llama.cpp's
                # own native load log be parsed for a real offload
                # confirmation instead of trusting the parameter alone.
                def _construct():
                    return Llama(
                        model_path=gguf_path,
                        chat_handler=chat_handler,
                        n_ctx=GEN_PARAMS["n_ctx"],
                        n_gpu_layers=n_gpu_layers,
                        verbose=True,
                    )

                # v7 FIX (Bug #43, PROJECT_MEMORY_v8.md): the actual
                # construction call now runs under a hard wall-clock
                # timeout instead of blocking this thread (and therefore
                # the reported `_status`) indefinitely -- see
                # `_run_with_timeout`'s docstring for the full reasoning.
                # A lightweight heartbeat thread updates `self._progress`
                # every 15s so the UI can show real elapsed time instead
                # of a static, indistinguishable-from-frozen "loading".
                stop_heartbeat = threading.Event()
                t0 = time.time()

                def _heartbeat():
                    while not stop_heartbeat.wait(15):
                        elapsed = int(time.time() - t0)
                        self._progress = (
                            f"still loading... {elapsed}s elapsed (a "
                            f"~{model_gb:.1f}GB model on "
                            f"{'GPU' if n_gpu_layers != 0 else 'CPU'} can "
                            f"take a few minutes, especially on the first "
                            f"load from disk)")
                hb = threading.Thread(target=_heartbeat, daemon=True,
                                      name="signxai-llm-heartbeat")
                hb.start()
                try:
                    if n_gpu_layers != 0:
                        finished, box, thread = _run_with_timeout(
                            lambda: _capture_native_output(_construct),
                            LLM_LOAD_TIMEOUT_SECONDS)
                        if not finished:
                            self._pending_construct = (thread, box)
                            raise _LoadTimeout(
                                f"Model construction did not finish within "
                                f"{LLM_LOAD_TIMEOUT_SECONDS:.0f}s (GPU path). "
                                f"This usually means the GPU doesn't actually "
                                f"have enough free VRAM despite being "
                                f"detected, or a driver-level issue is "
                                f"stalling native GPU initialisation. The "
                                f"attempt is still running in the background "
                                f"in case it finishes late; retrying now will "
                                f"pick that up if so, otherwise it starts a "
                                f"fresh CPU-only attempt is recommended via "
                                f"SIGNXAI_LLM_GPU=0.")
                        if "error" in box:
                            raise box["error"]
                        self._llm, captured = box["value"]
                        self._gpu_offload_confirmed = _parse_gpu_offload(captured)
                        if self._gpu_offload_confirmed is False:
                            log.warning(
                                "GPU hardware was detected, but llama.cpp's own load "
                                "log confirms 0 layers were actually offloaded -- the "
                                "installed backend likely doesn't have GPU support "
                                "compiled in (see PROJECT_MEMORY_v8.md Bug #32). "
                                "Falling back to CPU inference (slower, but correct); "
                                "reinstall a GPU-capable backend to fix this.")
                        elif self._gpu_offload_confirmed is None:
                            log.info(
                                "Could not confirm GPU offload from llama.cpp's load "
                                "log (pattern not recognized for this version) -- "
                                "status will report it as unconfirmed rather than "
                                "assuming success.")
                    else:
                        finished, box, thread = _run_with_timeout(
                            _construct, LLM_LOAD_TIMEOUT_SECONDS)
                        if not finished:
                            self._pending_construct = (thread, box)
                            raise _LoadTimeout(
                                f"Model construction did not finish within "
                                f"{LLM_LOAD_TIMEOUT_SECONDS:.0f}s (CPU path). "
                                f"This usually means available RAM is too "
                                f"tight for a ~{model_gb:.1f}GB model plus its "
                                f"KV cache, or the disk read (first load, no "
                                f"OS page-cache yet) is unusually slow. The "
                                f"attempt is still running in the background "
                                f"in case it finishes late -- retrying now "
                                f"will pick that up if so. You can also raise "
                                f"the ceiling with the SIGNXAI_LLM_LOAD_TIMEOUT "
                                f"environment variable (seconds).")
                        if "error" in box:
                            raise box["error"]
                        self._llm = box["value"]
                        self._gpu_offload_confirmed = False   # CPU-only was requested
                finally:
                    stop_heartbeat.set()
                self._status = "ready"
                self._progress = "ready"
                log.info("Vision LLM ready: %s (gpu_layers=%s, gpu_offload_confirmed=%s, "
                        "load_time=%.1fs)", self.device.model_id, n_gpu_layers,
                        self._gpu_offload_confirmed, time.time() - t0)
            except _LoadTimeout as exc:
                self._status = "error"
                self._error = str(exc)
                log.warning("LLM load timed out: %s", exc)
                raise
            except ModuleNotFoundError as exc:
                # Expected, common state: `llama-cpp-python` simply isn't
                # installed yet. This is NOT an error condition — it's
                # "not_installed", actionable via `python install.py --llm`.
                # Previously this fell into the generic except below and
                # permanently showed "LLM: error" in the UI even on a
                # perfectly normal machine (see PROJECT_MEMORY_v8.md Bug #9).
                self._status = "not_installed"
                self._error = f"{exc} — run: python install.py --llm"
                raise
            except FileNotFoundError as exc:
                # Dependency IS installed but weights aren't downloaded yet
                # — also not an error, just an actionable next step.
                self._status = "not_installed"
                self._error = str(exc)
                raise
            except (OSError, ImportError) as exc:
                # BUG #30 FIX (PROJECT_MEMORY_v8.md, user-reported real
                # Windows log + screenshot): `import llama_cpp` (or the
                # ctypes.CDLL call inside it) can fail with an OSError like
                # "Failed to load shared library '...\\llama.dll': Could
                # not find module '...\\llama.dll' (or one of its
                # dependencies)" -- this means the PACKAGE is present on
                # disk (pip installed it successfully) but its NATIVE
                # library can't actually load, almost always because a
                # CUDA-build's dependency (CUDA runtime DLLs, or the
                # Microsoft Visual C++ Redistributable) is missing on this
                # machine. install.py now catches this at install time
                # (Bug #30) and falls back to a CPU build automatically,
                # but a user who installed some other way (manually
                # copying a venv, an older install, pip by hand) can still
                # hit this at load time -- surface the SAME actionable
                # guidance here instead of a bare traceback string, so the
                # fix is discoverable no matter how they got into this
                # state.
                self._status = "error"
                hint = ""
                msg = str(exc)
                if "llama.dll" in msg or "dependencies" in msg.lower():
                    hint = (" — this usually means a CUDA-build dependency "
                           "(CUDA runtime DLLs, or the Microsoft Visual "
                           "C++ Redistributable) is missing on this "
                           "machine. Re-run `python install.py --llm`, "
                           "which will now detect this and fall back to "
                           "a CPU-only build automatically; or install "
                           "the Microsoft Visual C++ Redistributable "
                           "(x64) from "
                           "https://aka.ms/vs/17/release/vc_redist.x64.exe "
                           "and try again.")
                self._error = msg + hint
                raise
            except Exception as exc:
                # Genuinely unexpected failure (corrupt file, OOM, etc.) —
                # this is the only case that should show as "error".
                self._status = "error"
                self._error = str(exc)
                raise

    # -------------------------------------------------------- v5 autostart
    def load_now(self) -> dict:
        """Force a load attempt right now (blocking). Used by the explicit
        ``POST /api/llm/load`` retry button and by autostart()'s background
        thread. Never raises -- returns ok/False with the reason instead,
        since callers (a background thread, or a request that shouldn't
        503 just because the model isn't ready yet) both want a result,
        not an exception."""
        try:
            self._load()
            return {"ok": True, "status": self._status}
        except Exception as exc:
            return {"ok": False, "status": self._status, "reason": str(exc)}

    def autostart(self) -> None:
        """v5 (Bug #22, PROJECT_MEMORY_v8.md): the two checks the user asked
        for, run once at process start -- "[1] model path & size [2] gpu
        is available or not -- and start to load model on gpu" -- instead
        of waiting for a manual 'Download' click that a user who already
        has the weights on disk has no reason to make. refresh_device()
        does check [1] (model_present_on_disk) and [2] (detect_device's
        nvidia-smi probe); if both are satisfied this kicks off the actual
        GGUF load in a background thread so server startup itself is never
        blocked by however long that takes. Safe to call repeatedly (only
        starts a thread when there is actually something to load)."""
        self.refresh_device()
        if self._status != "downloaded":
            return
        def _bg() -> None:
            result = self.load_now()
            if result["ok"]:
                log.info("LLM autostart: model loaded and ready (%s)",
                         self.device.model_id)
            else:
                log.info("LLM autostart: load did not complete (%s) -- "
                         "falls back to template reports; retry via "
                         "POST /api/llm/load.", result.get("reason"))
        threading.Thread(target=_bg, daemon=True, name="signxai-llm-autostart").start()

    # ----------------------------------------------------------- generate
    def generate(self, images: list[str], prompt: str, system: str) -> str:
        """Run one multimodal chat completion.

        ``images`` are file paths or data-URLs (llama-cpp-python's
        Llava15ChatHandler accepts both via the ``image_url`` content type).
        Raises on any failure — caller (report.py) is responsible for the
        retry-then-template-fallback loop, this method stays simple.
        """
        self._load()
        content = [{"type": "text", "text": prompt}]
        for img in images:
            url = img if img.startswith("data:") else _path_to_file_uri(img)
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        out = self._llm.create_chat_completion(
            messages=messages,
            temperature=GEN_PARAMS["temperature"],
            top_p=GEN_PARAMS["top_p"],
            top_k=GEN_PARAMS["top_k"],
            repeat_penalty=GEN_PARAMS["repeat_penalty"],
            max_tokens=GEN_PARAMS["max_tokens"],
        )
        return out["choices"][0]["message"]["content"]


_manager: Optional[VisionLLMManager] = None


def get_manager() -> VisionLLMManager:
    global _manager
    if _manager is None:
        _manager = VisionLLMManager()
    return _manager
