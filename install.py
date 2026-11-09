#!/usr/bin/env python3
"""SignXAI — one-command cross-platform installer.

BUG #2 FIX (PROJECT_MEMORY_v8.md): no installer existed before; this is why
the prototype was hard to run on Windows (nothing did dependency setup, and
the LLM stack — before Bug #1's fix — assumed CUDA anyway).

USAGE
-----
    python install.py                 # core engine only (web UI + CLI, no AI explanation layer)
    python install.py --llm           # + download & set up the local vision-AI model
    python install.py --check         # just print hardware detection, install nothing

This is a plain Python script — no shell/batch files — so the exact same
command works on Windows, macOS, and Linux (project rule: PROJECT_MEMORY_v8.md
process rule #2).
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_has_pip(py: Path) -> bool:
    try:
        result = subprocess.run([str(py), "-m", "pip", "--version"],
                                capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def _bootstrap_pip(py: Path) -> bool:
    """BUG #24 FIX (PROJECT_MEMORY_v8.md): confirmed directly against a real
    user report — `venv.EnvBuilder(with_pip=True)` is supposed to guarantee
    pip exists in the new venv, but on a real Windows machine it created
    the venv, printed no error, and the very next line
    (`D:\\...\\python.exe -m pip install -r requirements.txt`) failed with
    "No module named pip". `with_pip=True` runs ensurepip internally and
    does not reliably surface or raise when that bootstrap is silently
    incomplete -- common causes seen in the wild: antivirus/security
    software quarantining pip's bundled wheel files during creation, or a
    base Python whose own `ensurepip` module is itself incomplete (some
    minimal/corporate Windows Python installs). Rather than trust
    `with_pip=True` and let a broken venv fail on the very next command
    with a confusing error, this explicitly VERIFIES pip works right after
    venv creation and, if it doesn't, tries two independent repairs before
    giving up loudly with an actionable message. This also self-heals an
    already-broken .venv/ left over from a previous failed run (like the
    one in the bug report) -- it's checked unconditionally, not only right
    after creation.
    """
    if _venv_has_pip(py):
        return True
    print("    [WARN] This virtual environment's pip did not install "
         "correctly (common causes: antivirus quarantining pip's files "
         "during setup, or an incomplete base Python ensurepip). "
         "Attempting to repair it automatically...")

    # Repair attempt 1: ensurepip, explicitly, with output shown.
    result = subprocess.run([str(py), "-m", "ensurepip", "--upgrade", "--default-pip"],
                            capture_output=True, text=True)
    if _venv_has_pip(py):
        print("    [OK] Repaired via ensurepip.")
        return True
    if result.stderr:
        print(f"    ensurepip output: {result.stderr.strip()[:400]}")

    # Repair attempt 2: fetch the official get-pip.py bootstrapper and run
    # it with the venv's own interpreter. This works even when the BASE
    # Python's ensurepip module is itself missing/broken (ensurepip repair
    # attempt 1 can't fix that, since it uses the same broken module).
    print("    ensurepip repair did not work — trying a direct get-pip.py "
         "bootstrap (needs internet access to bootstrap.pypa.io)...")
    try:
        import urllib.request
        get_pip = VENV_DIR / "get-pip.py"
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", str(get_pip))
        result = subprocess.run([str(py), str(get_pip)], capture_output=True, text=True)
        get_pip.unlink(missing_ok=True)
        if _venv_has_pip(py):
            print("    [OK] Repaired via get-pip.py.")
            return True
        if result.stderr:
            print(f"    get-pip.py output: {result.stderr.strip()[:400]}")
    except Exception as exc:
        print(f"    get-pip.py bootstrap failed: {exc}")

    print(f"\n    [FATAL] Could not get a working pip inside {VENV_DIR} "
         "after two repair attempts. This is almost always either "
         "(a) antivirus/security software blocking pip's files — try "
         "temporarily disabling it, or add an exclusion for this project "
         "folder, and re-run `python install.py`; or "
         "(b) no internet access to https://bootstrap.pypa.io from this "
         "machine. As a manual fallback, delete the .venv folder entirely "
         "and run:\n"
         f"        {py} -m ensurepip --upgrade\n"
         "    then re-run `python install.py` from a normal (non-elevated) "
         "prompt.")
    return False


def _ensure_venv_and_reexec() -> None:
    """BUG #4 FIX (found during install.py smoke-test, PROJECT_MEMORY_v8.md):
    modern Linux distros (Debian/Ubuntu 23.04+, PEP 668) refuse `pip
    install` outside a virtual environment with an
    'externally-managed-environment' error — this broke the "one command"
    promise on a real, common setup, not just Windows. Auto-create a local
    .venv/ and re-exec this exact script inside it, so `python install.py`
    keeps working unmodified everywhere, including inside the venv itself
    (idempotent — running it again just reuses the existing venv)."""
    if sys.prefix != sys.base_prefix:
        return  # already inside a venv (this one or another) — proceed normally
    py = _venv_python()
    if not py.exists():
        print(f">>> Creating local virtual environment at {VENV_DIR} "
             "(your system Python blocks direct pip installs — this keeps "
             "SignXAI's dependencies isolated and avoids that restriction)")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    if not _bootstrap_pip(py):
        sys.exit(1)
    print(f">>> Re-running installer inside the virtual environment...")
    os.execv(str(py), [str(py), str(Path(__file__).resolve())] + sys.argv[1:])


def run(cmd: list[str], desc: str, env: dict | None = None) -> bool:
    print(f"\n>>> {desc}")
    print("    $ " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"    [FAILED] {desc} (exit code {result.returncode})")
        return False
    print(f"    [OK] {desc}")
    return True


def pip_install(args: list[str], desc: str) -> bool:
    return run([sys.executable, "-m", "pip", "install"] + args, desc)


def install_core() -> bool:
    return pip_install(["-r", str(ROOT / "requirements.txt")],
                       "Installing core engine dependencies (numpy, opencv, "
                       "scikit-image, fastapi, rich, ...)")


def _parse_nvidia_smi_cuda_version(smi_output: str) -> Optional[tuple[int, int]]:
    """Extract the driver's reported max-supported CUDA version (the
    "CUDA Version: X.Y" field in `nvidia-smi`'s header banner) as
    (major, minor), or None if not found/parseable."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_output)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _check_cuda_runtime_ready() -> tuple[bool, str]:
    """v8 FIX (Bug #52, PROJECT_MEMORY_v8.md, user-requested: "I want the
    --llm installation script check does cuda based utility are ready to
    use in system"): the previous check for whether to attempt the GPU
    ('cu121') wheel index was just "did `nvidia-smi` run successfully" --
    which only confirms an NVIDIA GPU and its driver are PRESENT, not
    that this specific machine can actually LOAD a CUDA-linked native
    binary. That gap is exactly what produced the real failure this
    project has already seen and documented (PROJECT_MEMORY_v8.md Bug
    #25's log): the cu121 wheel downloaded and INSTALLED successfully
    (pip has no way to know it won't load), and only failed at actual
    import time with `FileNotFoundError: Could not find module` --
    because the driver's CUDA runtime version was too old for the
    wheel's requirements, or the Microsoft Visual C++ Redistributable
    (a separate, commonly-missing prerequisite on Windows) wasn't
    installed. Both of those are things that CAN be checked cheaply
    ahead of time, without downloading anything.

    Two checks, both best-effort (a failure to check is not treated as
    a failure of the check -- it degrades to "unknown, proceed anyway
    with a caveat" rather than blocking the user on a check that itself
    couldn't run):

    1. **Driver CUDA-version floor.** `nvidia-smi`'s own header reports
       the MAXIMUM CUDA runtime version the installed driver supports
       (not the toolkit -- no separate CUDA Toolkit install is needed
       for this). The cu121 wheel needs a driver that supports CUDA
       12.1 or newer (a real, documented NVIDIA compatibility
       requirement: a driver only supports CUDA versions up to the one
       it shipped with, and llama-cpp-python's prebuilt cu121 wheel
       links against the CUDA 12.1 runtime). Parsed directly from
       `nvidia-smi`'s own output -- verified against this project's own
       real Windows log (PROJECT_MEMORY_v8.md Bug #25's install
       transcript), which is real evidence this parsing approach works
       against actual `nvidia-smi` output format, not a guess.
    2. **Windows VC++ Redistributable presence.** The exact real error
       this project has on record (Bug #25) explicitly names this as a
       likely cause. Checked heuristically by looking for
       `vcruntime140.dll` in `System32` -- the specific DLL the VC++
       Redistributable installs there, and the standard way to detect
       its presence without needing registry access or admin rights.

    Returns ``(ready, message)`` -- ``ready=False`` means "don't bother
    attempting the cu121 wheel, go straight to CPU", with an actionable
    message explaining why. This is a PRE-FLIGHT check to save the user
    a doomed ~450MB download and a confusing later failure -- it does
    not replace the existing post-install verification
    (`app/llm/manager.py`'s own load-time GPU-offload confirmation,
    Bug #32), which remains the authoritative check once the wheel is
    actually installed and loaded.
    """
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, text=True)
    except Exception as exc:
        return False, f"'nvidia-smi' could not be run ({exc}) -- no NVIDIA GPU/driver detected."
    if out.returncode != 0:
        return False, "'nvidia-smi' returned an error -- no usable NVIDIA GPU/driver detected."

    smi_text = out.stdout or ""
    cuda_ver = _parse_nvidia_smi_cuda_version(smi_text)
    _MIN_CUDA = (12, 1)   # cu121 wheel's own requirement
    if cuda_ver is not None and cuda_ver < _MIN_CUDA:
        return False, (
            f"NVIDIA driver reports support for CUDA {cuda_ver[0]}.{cuda_ver[1]} only, "
            f"but the 'cu121' prebuilt wheel needs a driver supporting CUDA "
            f"{_MIN_CUDA[0]}.{_MIN_CUDA[1]}+. Update the NVIDIA driver "
            f"(https://www.nvidia.com/drivers) to enable GPU offload, or continue "
            f"with the CPU backend for now.")
    if cuda_ver is None:
        # Couldn't parse it -- don't block on an inconclusive check, but
        # say so plainly rather than silently treating it as a pass.
        log_note = ("(could not parse a CUDA version from 'nvidia-smi' output -- "
                    "proceeding with the GPU wheel attempt anyway, since a GPU "
                    "was detected; this specific sub-check was inconclusive, "
                    "not failed)")
    else:
        log_note = f"(driver supports CUDA {cuda_ver[0]}.{cuda_ver[1]}, meets the {_MIN_CUDA[0]}.{_MIN_CUDA[1]}+ floor)"

    if platform.system() == "Windows":
        vcredist_dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vcruntime140.dll"
        if not vcredist_dll.exists():
            return False, (
                "NVIDIA GPU/driver looks fine "
                f"{log_note}, but the Microsoft Visual C++ Redistributable "
                f"does not appear to be installed (missing {vcredist_dll}) -- "
                "this is a separate, commonly-missed prerequisite for the "
                "cu121 wheel's native DLL to load on Windows (see "
                "PROJECT_MEMORY_v8.md Bug #25 for the exact real failure this "
                "causes). Install it from "
                "https://aka.ms/vs/17/release/vc_redist.x64.exe, or continue "
                "with the CPU backend for now.")

    return True, f"CUDA runtime pre-flight check passed {log_note}."


def _llm_backend() -> tuple[str, str]:
    """Pick the llama-cpp-python prebuilt-wheel INDEX BACKEND for this
    machine. Returns (backend_name, human-readable note).

    BUG #25 FIX (PROJECT_MEMORY_v8.md): the previous approach here
    (hand-building a direct GitHub release download URL, assuming a
    `llama_cpp_python-<ver>-py3-none-<platform>.whl` filename) was WRONG —
    caught by re-verifying against the project's own current README, PyPI
    page, and ReadTheDocs together (all independently agree, checked
    fresh for this fix): llama-cpp-python ships compiled C extensions, so
    its real wheel filenames are tagged per-Python-version
    (`cp310`/`cp311`/`cp312`/...), never a universal `py3-none` tag — the
    exact assumption the old code baked into every URL it built. The
    upstream release/tag naming scheme has also changed since this
    project last checked it directly (separate backend-suffixed tags like
    `v0.3.35-metal`/`v0.3.35-cu121`, not one tag with a full asset
    matrix). A hand-built filename guess is fragile against both facts
    and would very likely 404 on a real run — this was never actually
    exercised end-to-end against a real Windows/macOS machine with a
    genuine 5GB+ GPU, so the earlier "verified" claim documented a check
    that, in hindsight, wasn't checking the right thing.

    The fix: use what the project's own README documents and recommends
    instead — a PEP 503 "simple" package index per backend,
    `https://abetlen.github.io/llama-cpp-python/whl/<backend>`, which pip
    resolves against the CURRENT interpreter's actual Python version and
    platform automatically (the same mechanism PyPI itself uses). No
    filename-guessing, no version-tag guessing, and it keeps working
    as new Python versions and llama-cpp-python releases ship, unlike a
    hardcoded URL.
    """
    os_name = platform.system()
    arch = platform.machine().lower()

    if os_name == "Darwin" and arch in ("arm64", "aarch64"):
        return "metal", ("macOS (Apple Silicon): using the official 'metal' "
                         "prebuilt-wheel index (Metal GPU support included, "
                         "no compiler needed).")

    has_nvidia = False
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        has_nvidia = out.returncode == 0
    except Exception:
        pass
    if has_nvidia:
        # v8 FIX (Bug #52, PROJECT_MEMORY_v8.md, user-requested): a GPU
        # being PRESENT (what the check above confirms) is not the same
        # as this machine being READY to actually load a CUDA-linked
        # native binary -- see `_check_cuda_runtime_ready`'s docstring
        # for the real, on-record failure mode this closes. Runs first,
        # before committing to a ~450MB download that Bug #25's own log
        # shows can install successfully and still fail to load.
        cuda_ready, cuda_msg = _check_cuda_runtime_ready()
        if not cuda_ready:
            return "cpu", (
                f"{os_name}: NVIDIA GPU detected, but a CUDA-readiness "
                f"pre-flight check failed: {cuda_msg} Using the 'cpu' "
                f"prebuilt-wheel index for now -- re-run 'python install.py "
                f"--llm' after addressing this to retry GPU.")
        # cu121 (CUDA 12.1) is the default GPU backend picked here: it's
        # the example version used in the project's own README, and a
        # newer NVIDIA driver's CUDA runtime is backward compatible with
        # binaries built against this older toolkit version. Advanced
        # users on a specific newer CUDA toolkit can install a more exact
        # backend (cu122/cu124/cu130/...) manually afterwards if they want.
        return "cu121", (f"{os_name}: NVIDIA GPU detected -- {cuda_msg} Using the "
                         "official 'cu121' CUDA prebuilt-wheel index.")
    return "cpu", f"{os_name}: using the official 'cpu' prebuilt-wheel index (no compiler needed)."


def _wheel_only_pip_install(args: list[str], desc: str, env: dict | None = None) -> bool:
    """Like pip_install, but with --only-binary=:all: -- pip will FAIL
    instead of silently substituting a source distribution when no wheel
    matches. This is the key fix for Bug #28 (PROJECT_MEMORY_v8.md,
    user-reported real Windows log): pip's normal multi-index resolution
    does not strictly prefer an OLDER version that has a published wheel
    over a NEWER version that only has a source distribution -- on a real
    run, `pip install llama-cpp-python --extra-index-url .../cu121`
    silently picked the newest sdist-only release (0.3.35) instead of any
    older version that might have had a real cu121 wheel, then tried (and
    failed) to compile it. Wheel-only mode makes that failure immediate
    and clearly attributable, instead of a 10-minute detour into a doomed
    compile with a confusing low-level CMake error at the end of it.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--only-binary=:all:"] + args
    print(f"\n>>> {desc}")
    print("    $ " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"    [no matching prebuilt wheel] {desc}")
        return False
    print(f"    [OK] {desc}")
    return True


def _find_vswhere() -> Path | None:
    for candidate in (
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
    ):
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _find_vcvars64() -> Path | None:
    """Locate vcvars64.bat via vswhere -- Visual Studio's own official,
    version-independent installation-discovery tool (bundled with every
    VS/Build Tools install since VS2017) -- rather than guessing a
    version-specific install path."""
    vswhere = _find_vswhere()
    if vswhere is None:
        return None
    try:
        out = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        return None
    candidate = Path(lines[0].strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return candidate if candidate.exists() else None


def _msvc_build_env() -> dict | None:
    """BUG #28 FIX (PROJECT_MEMORY_v8.md): a real user's from-source build
    failed with `CMake Error ... 'nmake' ... no such file or directory` --
    NMake and the MSVC compiler (cl.exe) are only on PATH inside a
    "Developer Command Prompt for VS" that has run vcvars64.bat; a plain
    PowerShell/cmd window (what `python install.py` actually runs in) never
    has them, EVEN WITH Build Tools correctly installed. Rather than just
    tell the user to find and use that special prompt themselves, this
    locates vcvars64.bat automatically (see _find_vcvars64) and runs it in
    a child cmd.exe, capturing the resulting environment (PATH, INCLUDE,
    LIB, ...) so THIS PROCESS's own pip install can use it transparently.
    Returns None (never raises) if no MSVC install with the C++ workload
    can be found at all -- the caller is responsible for telling the user
    what to install in that case.
    """
    vcvars = _find_vcvars64()
    if vcvars is None:
        return None
    try:
        out = subprocess.run(f'"{vcvars}" && set', shell=True,
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    env = dict(os.environ)
    for line in out.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    return env


def _install_llm_from_source() -> bool:
    print("\n    Falling back to a from-source build (no matching prebuilt "
         "wheel was found for this exact Python version/platform).")
    if platform.system() == "Windows":
        # Detect a compiler on the CURRENT PATH first -- covers a Developer
        # Command Prompt, or a machine where vcvars was already sourced.
        have_cl = shutil.which("cl") is not None
        env = None
        if not have_cl:
            print("    No C++ compiler (cl.exe) found on PATH -- looking for "
                 "a Visual Studio Build Tools install to load automatically "
                 "(this is the actual fix for the 'nmake ... no such file' "
                 "error: Build Tools being installed is not enough, its "
                 "compiler environment also has to be loaded first, which a "
                 "plain terminal never does on its own)...")
            env = _msvc_build_env()
            if env is not None:
                print("    [OK] Found and loaded a Visual Studio C++ build "
                     "environment automatically.")
            else:
                print("\n    [FATAL] No Visual Studio C++ Build Tools "
                     "installation could be found on this machine, so a "
                     "from-source build is not possible right now.\n"
                     "    Install 'Build Tools for Visual Studio' "
                     "(https://visualstudio.microsoft.com/visual-cpp-build-tools/), "
                     "selecting the 'Desktop development with C++' workload, "
                     "then re-run `python install.py --llm`.\n"
                     "    The app works fully without this -- template "
                     "reports instead of AI-narrated ones -- if you'd "
                     "rather skip it.")
                return False
        return run([sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                   "Installing llama-cpp-python (source build)", env=env)

    # macOS / Linux: a missing compiler fails with pip's own reasonably
    # clear error, and there's no vcvars-style hidden-environment problem
    # to work around -- just make sure the guidance is visible either way.
    print("    This needs a C++ compiler + cmake installed (macOS: "
         "'xcode-select --install'; Linux: build-essential + cmake). If you "
         "don't have one, the app still works fully without the AI layer.")
    return pip_install(["llama-cpp-python"],
                       "Installing llama-cpp-python (source build, needs a compiler)")


def _verify_llama_cpp_importable() -> tuple[bool, str]:
    """Actually try to import llama_cpp in a fresh subprocess.

    BUG #30 FIX (PROJECT_MEMORY_v8.md, user-reported real Windows log,
    confirmed with a screenshot of the app's own error): pip reporting a
    successful install only means the wheel's files were unpacked
    correctly -- it says NOTHING about whether the native shared library
    inside it can actually be LOADED on this machine. A CUDA-enabled
    build's native ``llama.dll`` can depend on other DLLs (CUDA runtime
    components such as ``cudart64_12.dll``/``cublas64_12.dll``, or the
    Microsoft Visual C++ Redistributable) that aren't present, or aren't
    a version compatible with this machine's installed NVIDIA driver --
    and pip has no way to know about any of that at install time. This is
    exactly what happened on a real machine: `pip install` reported
    "Successfully installed ... llama-cpp-python-0.3.4" after downloading
    a real 448MB cu121 wheel, but the app's own first load attempt failed
    with "Failed to load shared library '...\\llama.dll': Could not find
    module '...\\llama.dll' (or one of its dependencies)". Running the
    actual import right after install, in this same installer, catches
    that gap immediately instead of leaving a broken backend installed
    for the user to discover later when they try to use it.
    """
    try:
        result = subprocess.run([sys.executable, "-c", "import llama_cpp"],
                                capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or result.stdout or "unknown import error").strip()


def _uninstall_llama_cpp() -> None:
    """Best-effort cleanup of a package that installed but can't load --
    never leave a broken backend sitting in .venv/ silently claiming to
    be usable (see _verify_llama_cpp_importable's docstring, Bug #30)."""
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
                   capture_output=True)


def install_llm_backend() -> bool:
    print("\n" + "=" * 60)
    print("Setting up the local vision-AI explanation layer")
    print("=" * 60)
    if not pip_install(["huggingface_hub>=0.24"], "Installing huggingface_hub"):
        return False

    backend, note = _llm_backend()
    print(f"    {note}")

    # BUG #28 FIX (PROJECT_MEMORY_v8.md, user-reported real Windows log):
    # try the detected backend wheel-only (fails fast instead of silently
    # falling back to a source-only newer release, see
    # _wheel_only_pip_install's docstring); if that specific backend has no
    # matching wheel yet for this exact Python version, retry with the
    # 'cpu' backend before ever attempting a compile -- 'cpu' has by far
    # the widest published wheel coverage across Python versions, and
    # still gives a fully working (if not GPU-accelerated) AI layer.
    candidates = [backend] + (["cpu"] if backend != "cpu" else [])
    for i, cand in enumerate(candidates):
        index_url = f"https://abetlen.github.io/llama-cpp-python/whl/{cand}"
        if not _wheel_only_pip_install(
                ["llama-cpp-python", "--extra-index-url", index_url],
                f"Installing llama-cpp-python (prebuilt '{cand}' wheel index)"):
            continue

        # BUG #30 FIX (PROJECT_MEMORY_v8.md, user-reported real Windows
        # log + screenshot): a successful pip install does NOT mean the
        # native library can actually load -- verify it actually can
        # before declaring success, and self-heal by trying the next,
        # more broadly-compatible candidate if it can't (see
        # _verify_llama_cpp_importable's docstring for the full story).
        ok, err = _verify_llama_cpp_importable()
        if ok:
            if cand != backend:
                print(f"    Note: no prebuilt '{backend}' wheel was published "
                     f"yet for this exact Python version, so the 'cpu' "
                     "backend was installed instead -- the AI layer still "
                     "works fully, just without GPU offload (slower "
                     "generation). You can retry the GPU backend later once "
                     "a matching wheel is published, by re-running "
                     "`python install.py --llm`.")
            return True

        print(f"\n    [WARN] '{cand}' installed successfully but its native "
             f"library failed to load:\n    {err[:500]}")
        is_last = (i == len(candidates) - 1)
        if "cu" in cand and platform.system() == "Windows":
            print("    This is a common Windows CUDA-wheel issue: the "
                 "installed package's native DLL depends on components "
                 "(CUDA runtime DLLs, or the Microsoft Visual C++ "
                 "Redistributable) that aren't present on this machine, "
                 "independent of anything pip can check. " +
                 ("Trying the CPU backend instead, which has no such "
                  "dependency..." if not is_last else
                  "If installing the CPU backend below also doesn't help, "
                  "installing the Microsoft Visual C++ Redistributable "
                  "(x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe "
                  "resolves this in the large majority of cases."))
        _uninstall_llama_cpp()   # never leave a broken backend installed

    print("\n    [WARN] No installed backend could actually be loaded. "
         "Falling back to a from-source build...")
    return _install_llm_from_source()


def download_model() -> None:
    """v4: single model, no tier system (see app/device.py). Downloads
    whichever single model app.device.MODEL_ID names (v8: switched to
    google/gemma-4-E2B-it-qat-q4_0-gguf, Bug #53) if (and only if) this
    machine has enough combined RAM+VRAM -- no other model is ever
    considered."""
    sys.path.insert(0, str(ROOT))
    from app.device import MODEL_ID, detect_device
    from app.llm.manager import get_manager

    device = detect_device()
    print(f"\nDetected: {device.os}/{device.arch}, "
         f"{'NVIDIA GPU: ' + str(device.gpu_name) if device.has_cuda else 'no NVIDIA GPU'}, "
         f"{device.vram_gb}GB VRAM, {device.ram_gb}GB RAM "
         f"({'Apple Silicon/Metal' if device.has_mps else 'no Metal'})")
    if not device.viable:
        print(f"\n[SKIP] {device.notes}")
        return
    print(f"Model: {MODEL_ID}")
    print(f"({device.notes})")

    mgr = get_manager()
    print("\nDownloading model weights now (this runs in THIS process, not a "
         "background thread, so it won't be silently killed when the script "
         "exits -- see PROJECT_MEMORY_v8.md Bug #17). huggingface_hub prints its "
         "own progress bars below; this can take several minutes depending on "
         "model size and connection speed.\n")
    result = mgr.download_blocking()
    if not result["ok"]:
        print(f"\n[FAILED] {result['reason']}")
        print("You can retry later from the web UI (AI Model tab) or CLI (option 4).")
    else:
        print(f"\n[OK] Model ready. Run 'python main.py' or 'python cli.py' to use it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true",
                    help="also install and download the local vision-AI model")
    ap.add_argument("--check", action="store_true",
                    help="only print hardware detection + recommendation, install nothing")
    args = ap.parse_args()

    print(f"SignXAI installer — Python {platform.python_version()} on "
         f"{platform.system()} {platform.machine()}")

    _ensure_venv_and_reexec()   # no-op if already inside a venv

    # Defensive re-check even when already inside a venv (e.g. someone
    # `python -m venv`'d their own, or activated a pre-existing one by
    # hand outside install.py's own creation path above) — same failure
    # mode, same fix, see _bootstrap_pip's docstring (Bug #24).
    if not _bootstrap_pip(Path(sys.executable)):
        sys.exit(1)

    if args.check:
        sys.path.insert(0, str(ROOT))
        from app.device import detect_device
        d = detect_device()
        print("\nHardware detection:")
        for k, v in d.to_dict().items():
            print(f"  {k}: {v}")
        return

    if not install_core():
        print("\n[FATAL] Core dependency install failed. See errors above.")
        sys.exit(1)

    if args.llm:
        if not install_llm_backend():
            print("\n[WARN] LLM backend install failed — the app still works "
                 "fully without it (deterministic template reports instead "
                 "of AI narrative). See errors above.")
        else:
            download_model()

    run_py = _venv_python()
    rel = run_py.relative_to(ROOT) if VENV_DIR.exists() else Path("python")
    print("\n" + "=" * 60)
    print("Setup complete. All dependencies are installed inside .venv/ — ")
    print("use THIS python (not your system python) to run the app:")
    print(f"  Web app:      {rel} main.py")
    print(f"  Terminal UI:  {rel} cli.py")
    if not args.llm:
        print("  (AI explanation layer not installed — re-run with --llm to add it)")
    print("=" * 60)


if __name__ == "__main__":
    main()
