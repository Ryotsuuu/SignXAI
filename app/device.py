"""Hardware / device detection for the (single) local vision-LLM model.

v4 CHANGE (PROJECT_MEMORY_v8.md, explicit user instruction): dropped the
multi-tier model system entirely. There is exactly ONE model this project
ever downloads. No tiers, no "pick a bigger model if you have more VRAM"
logic. This was requested directly after a real bug where a user's web UI
silently tried to download the 31B tier because their hardware happened to
qualify for it, wasting bandwidth and confusing the "which model am I
actually using" story. One model, one code path, no ambiguity. v8 keeps
this principle intact -- it changes WHICH single model that is, not the
one-model policy itself (see the v8 note immediately below).

v8 CHANGE (Bug #53, PROJECT_MEMORY_v8.md, user-requested model swap):
switched from google/gemma-4-E4B-it-qat-q4_0-gguf to
google/gemma-4-E2B-it-qat-q4_0-gguf. The user specifically asked for
google/gemma-4-12B-it-assistant instead -- checked directly (web search,
Hugging Face model cards) before making any change, and that exact model
turned out to be technically incompatible with this project's use case: it
is a "draft"/MTP (multi-token-prediction) assistant model, explicitly
documented by its own publishers as "not intended to be run as a
standalone chat model" -- it only works paired with a full-size
google/gemma-4-12B-it model for speculative-decoding acceleration, and
cannot generate a report on its own. Its real full-size counterpart
(google/gemma-4-12B-it) is also LARGER than the E4B model already in use
(~7.2GB vs ~5.7GB at Q4_0) -- switching to it would have made this
project's actual, repeatedly-reported problem (GPU offload failing on a
6GB card because the model needs ~8.3GB) measurably WORSE, not better.
google/gemma-4-E2B-it-qat-q4_0-gguf is the well-reasoned alternative:
same publisher, same naming convention, same QAT/Q4_0 format and the same
Gemma-4 multimodal (image) support as the model already in use, at
roughly half the size (~2.84GB vs ~5.7GB per the GGUF file sizes on
Hugging Face, checked directly, not assumed) -- comfortably fits a 6GB
GPU's VRAM even with realistic KV-cache/runtime overhead, which the E4B
model measurably did not (PROJECT_MEMORY_v8.md Bug #44's own VRAM-gating
math: E4B needed an estimated ~8.3GB against 6GB available). The actual
report-generation task this model performs -- writing a structured
forensic narrative from JSON measurements the classical engine already
computed, following a strict schema -- does not need frontier-scale
general reasoning capability, making a smaller model in the same family a
reasonable trade for a project whose primary tested hardware cannot fit
the previous choice on GPU at all.

Also fixes Bug #12 properly (installed-model-priority) by construction:
since there's only one possible model, "which model is recommended" and
"which model is installed" are now the same question by definition.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# The one and only model this project downloads. See v4 and v8 notes above.
MODEL_ID = "google/gemma-4-E2B-it-qat-q4_0-gguf"
MIN_VIABLE_TOTAL_GB = 3.0   # combined RAM+VRAM floor for this model (E2B, ~2.84GB GGUF per HF, +headroom)

# v5 (Bug #22, PROJECT_MEMORY_v8.md): user-specified disk-presence check.
# Previously the manager had NO code path that ever looked at what's
# actually on disk -- status started at "not_installed" and only ever
# changed via download_blocking(), so a model a user copied in by hand
# (from a previous install, or manually from another machine) was
# invisible to the app forever, no matter how long it sat in the right
# folder. Slightly below the real file size (not exactly on it) so minor
# version/quantization file-size differences don't cause a false
# "not installed" read on a genuinely-present model -- see
# model_present_on_disk() below. v8: lowered from 4.5 to 2.5 alongside the
# E2B model swap above (E2B's GGUF is ~2.84GB, not E4B's ~5GB+).
MODEL_PRESENT_MIN_GB = 2.5


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _nvidia_smi_probe() -> tuple[bool, Optional[str], float, list[dict]]:
    """GPU probe via nvidia-smi. Only ever lists NVIDIA GPUs -- an
    integrated GPU (Intel/AMD) never appears here, so it can never be
    mistaken for a usable CUDA device. On a multi-NVIDIA-GPU machine, picks
    the one with the most VRAM and returns the full list for diagnostics."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            gpus = []
            for line in out.stdout.strip().splitlines():
                name, mem_mb = line.rsplit(",", 1)
                gpus.append({"name": name.strip(),
                           "vram_gb": round(float(mem_mb.strip()) / 1024.0, 1)})
            if gpus:
                best = max(gpus, key=lambda g: g["vram_gb"])
                return True, best["name"], best["vram_gb"], gpus
    except Exception:
        pass
    return False, None, 0.0, []


def model_dir_for(model_id: str) -> Path:
    from .config import MODELS_DIR
    return MODELS_DIR / model_id.replace("/", "--")


def model_present_on_disk(model_id: str = MODEL_ID) -> tuple[bool, float]:
    """Sum the bytes actually sitting under this model's cache directory
    and compare against MODEL_PRESENT_MIN_GB. This is a real filesystem
    check every time it's called -- no cached in-memory flag -- so it
    correctly picks up a model directory that appeared after process
    start (e.g. a user copying weights in while the server is running, or
    a completed background download)."""
    d = model_dir_for(model_id)
    if not d.exists():
        return False, 0.0
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    gb = total / (1024 ** 3)
    return gb >= MODEL_PRESENT_MIN_GB, round(gb, 2)


def _apple_silicon_probe(os_name: str, arch: str) -> bool:
    """Apple Silicon (M-series) unified memory usable by llama.cpp's Metal
    backend for GPU offload -- no CUDA needed."""
    return os_name == "Darwin" and arch in ("arm64", "aarch64")


@dataclass
class DeviceProfile:
    """Snapshot of the machine relevant to whether the model can run."""
    os: str
    arch: str
    has_cuda: bool = False
    gpu_name: Optional[str] = None
    vram_gb: float = 0.0
    has_mps: bool = False
    ram_gb: float = 0.0
    total_gb: float = 0.0
    viable: bool = False           # can this machine run the model at all
    model_id: Optional[str] = None  # MODEL_ID if viable, else None
    notes: str = ""
    all_gpus: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_device() -> DeviceProfile:
    """Detect the current machine and decide if it can run MODEL_ID.

    No torch/CUDA import anywhere in this module -- nvidia-smi (if
    present) is the only optional external dependency, and its absence is
    handled gracefully (falls back to RAM-only budget).
    """
    os_name = platform.system()
    arch = platform.machine()
    has_cuda, gpu_name, vram, all_gpus = _nvidia_smi_probe()
    has_mps = _apple_silicon_probe(os_name, arch)
    ram = _ram_gb()
    total = vram + ram

    if total < MIN_VIABLE_TOTAL_GB:
        return DeviceProfile(
            os=os_name, arch=arch, has_cuda=has_cuda, gpu_name=gpu_name,
            vram_gb=vram, has_mps=has_mps, ram_gb=ram, total_gb=total,
            viable=False, model_id=None, all_gpus=all_gpus,
            notes=(f"Only ~{total:.1f}GB combined RAM/VRAM detected -- below "
                   f"the {MIN_VIABLE_TOTAL_GB}GB floor for {MODEL_ID}. "
                   "Reports will use the deterministic forensic template "
                   "engine instead of the AI narrative layer."),
        )

    if has_cuda:
        gpu_note = f" NVIDIA GPU detected ({gpu_name}, {vram:.1f}GB VRAM)."
        if os_name == "Windows":
            gpu_note += (" install.py will attempt the same official CUDA "
                        "prebuilt-wheel index used on any platform; if no "
                        "matching wheel is published for this exact Python "
                        "version, it falls back to a from-source build "
                        "(needs a C++ compiler) or the CPU wheel -- see "
                        "PROJECT_MEMORY_v8.md Bug #25.")
    elif has_mps:
        gpu_note = " Apple Silicon (Metal) GPU offload available."
    else:
        gpu_note = " No GPU detected -- CPU-only, expect slower generation."

    return DeviceProfile(
        os=os_name, arch=arch, has_cuda=has_cuda, gpu_name=gpu_name,
        vram_gb=vram, has_mps=has_mps, ram_gb=ram, total_gb=total,
        viable=True, model_id=MODEL_ID, all_gpus=all_gpus,
        notes=f"Hardware sufficient for {MODEL_ID}.{gpu_note}",
    )
