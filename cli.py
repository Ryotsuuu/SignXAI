#!/usr/bin/env python3
"""SignXAI terminal interface — full functionality without the web server.

BUG #2 FIX (PROJECT_MEMORY_v8.md): no CLI existed before. Run with:

    python cli.py

Uses `rich` for colour + block-character progress bars, per the project
brief ("terminal in rich in color & cool texture visualization... Anime.js +
block characters" — no JS terminal exists, so this is the equivalent: rich's
block-gradient progress bars use the same ░▒▓█ glyphs).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console                              # noqa: E402
from rich.panel import Panel                                   # noqa: E402
from rich.progress import (BarColumn, Progress, SpinnerColumn,  # noqa: E402
                           TextColumn)
from rich.prompt import Confirm, Prompt                        # noqa: E402
from rich.table import Table                                   # noqa: E402
from rich.text import Text                                     # noqa: E402

from app.config import APP_TITLE, ensure_dirs                  # noqa: E402
from app.core import storage                                   # noqa: E402
from app.core.engine import VerificationEngine                 # noqa: E402
from app.device import detect_device                           # noqa: E402
from app.llm.manager import get_manager                        # noqa: E402
from app.llm import report as report_mod                       # noqa: E402

console = Console()
engine = VerificationEngine()

VERDICT_STYLE = {"GENUINE": "bold green", "FORGED": "bold red",
                 "INCONCLUSIVE": "bold yellow"}
BLOCKS = " ░▒▓█"


def block_bar(value: float, width: int = 30) -> str:
    """Render a 0..1 value as a block-gradient bar — ░▒▓█ style."""
    filled = value * width
    full = int(filled)
    frac = filled - full
    bar = BLOCKS[-1] * full
    if full < width:
        idx = min(int(frac * (len(BLOCKS) - 1)), len(BLOCKS) - 1)
        bar += BLOCKS[idx]
        bar += BLOCKS[0] * (width - full - 1)
    return bar


def banner() -> None:
    console.print(Panel.fit(
        Text(APP_TITLE, style="bold white on black", justify="center"),
        border_style="grey50"))


def main_menu() -> None:
    ensure_dirs()
    banner()
    while True:
        console.print()
        console.print("[bold]1[/] Enroll a signer")
        console.print("[bold]2[/] Verify a signature")
        console.print("[bold]3[/] Case log")
        console.print("[bold]4[/] AI model status / download")
        console.print("[bold]5[/] Device diagnostics")
        console.print("[bold]0[/] Exit")
        choice = Prompt.ask("Select", choices=["0", "1", "2", "3", "4", "5"], default="0")
        if choice == "0":
            console.print("[dim]Goodbye.[/]")
            return
        {"1": do_enroll, "2": do_verify, "3": do_case_log,
         "4": do_llm, "5": do_device}[choice]()


def do_enroll() -> None:
    name = Prompt.ask("Signer name")
    matches = storage.find_profiles_by_name(name)
    profile_id = None
    if matches:
        m = matches[0]
        console.print(f"[yellow]Found existing profile '{m['name']}' "
                      f"({m['n_references']} samples).[/]")
        if Confirm.ask("Add new samples to this existing profile?", default=True):
            profile_id = m["id"]
    if profile_id is None:
        profile = storage.create_profile(name)
        profile_id = profile["id"]
        console.print(f"[green]Created profile {profile_id}[/]")

    paths = []
    while True:
        p = Prompt.ask("Path to a signature image (blank to finish)", default="")
        if not p:
            break
        if not Path(p).exists():
            console.print(f"[red]File not found: {p}[/]")
            continue
        paths.append(p)
    if not paths:
        console.print("[yellow]No samples provided — nothing enrolled.[/]")
        return

    images = [Path(p).read_bytes() for p in paths]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                 console=console) as prog:
        prog.add_task("Extracting features & recalibrating profile…", total=None)
        stats = engine.enroll(profile_id, images)
    console.print(f"[green]Enrolled {len(paths)} sample(s).[/] "
                 f"Profile now has {stats['n_references']} reference(s), "
                 f"adaptive threshold {stats['threshold']}.")


def do_verify() -> None:
    profiles = storage.list_profiles()
    if not profiles:
        console.print("[yellow]No profiles enrolled yet.[/]")
        return
    table = Table(show_header=True)
    table.add_column("#"); table.add_column("Name"); table.add_column("Refs")
    for i, p in enumerate(profiles, 1):
        table.add_row(str(i), p["name"], str(p["n_references"]))
    console.print(table)
    idx = int(Prompt.ask("Profile #", choices=[str(i) for i in range(1, len(profiles) + 1)])) - 1
    profile_id = profiles[idx]["id"]

    q_path = Prompt.ask("Path to the questioned signature image")
    if not Path(q_path).exists():
        console.print(f"[red]File not found: {q_path}[/]")
        return
    query = Path(q_path).read_bytes()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                 console=console) as prog:
        prog.add_task("Aligning, scoring, generating evidence…", total=None)
        try:
            case = engine.verify(profile_id, query)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            return

    style = VERDICT_STYLE.get(case["verdict"], "white")
    console.print()
    console.print(Panel.fit(
        f"[{style}]{case['verdict']}[/]  score {case['score']} vs threshold {case['threshold']}\n"
        f"confidence {block_bar(case['confidence'])} {case['confidence']*100:.1f}%\n"
        f"compared against {case['aggregate']['n_references_compared']} reference sample(s) "
        f"+ 1 aggregate profile score",
        title="Verdict", border_style=style.split()[-1]))

    # v5: per-reference transparency in the terminal too, matching the web
    # UI's "reference_scores" table — the questioned signature really was
    # compared against every enrolled sample, not just one (Bug #21).
    rtable = Table(title="Score vs. each enrolled reference")
    rtable.add_column("Reference"); rtable.add_column("Score"); rtable.add_column("Transform applied")
    for r in case["reference_scores"]:
        # v7 (Bug #36, PROJECT_MEMORY_v8.md): relabeled from "Aligned" --
        # "no" here means identity was already the best fit (often the
        # BEST case, e.g. an exact or near-exact match), not that
        # alignment failed. See align.py's module docstring.
        rtable.add_row(r["ref_image"], str(r["score"]), "yes" if r["aligned_ok"] else "no")
    rtable.add_row(f"[bold]PROTOTYPE (all samples)[/]",
                   f"[bold]{case['aggregate']['prototype_score']}[/]", "")
    console.print(rtable)

    edge = case.get("edge_case_review") or {}
    if edge.get("triggered"):
        console.print(Panel(
            "\n".join(f"• {f}" for f in edge["flags"]),
            title=f"[yellow]Edge-case review flagged this verdict[/] "
                 f"(fused score alone said {case['original_verdict']})",
            border_style="yellow"))

    # v6 (Bug #31, PROJECT_MEMORY_v8.md): surface zonal (local) analysis
    # findings in the terminal too — the questioned signature's stroke
    # pattern was checked region-by-region, not just as one global score.
    zonal = case.get("zonal_analysis") or {}
    if zonal.get("flagged_zones"):
        ztable = Table(title=f"Zonal (local) stroke-pattern findings "
                             f"(aggregate zonal score: {zonal['aggregate_score']})")
        ztable.add_column("Zone"); ztable.add_column("Query corners")
        ztable.add_column("Reference corners"); ztable.add_column("Ink mass %")
        for z in zonal["flagged_zones"]:
            ztable.add_row(f"row {z['row']}, col {z['col']}", str(z["query_corners"]),
                          str(z["ref_corners"]), f"{z['ink_mass_fraction']*100:.1f}%")
        console.print(ztable)
        console.print("[dim]See the zone_grid.png asset for a visual breakdown "
                      "of every region.[/]")

    mtable = Table(title="Deviant features (|z| > 2)")
    mtable.add_column("Feature"); mtable.add_column("Category")
    mtable.add_column("Query"); mtable.add_column("Ref mean±std"); mtable.add_column("Z")
    for m in case["metrics"]:
        if m["deviation"]:
            mtable.add_row(m["label"], m["category"], str(m["query_value"]),
                          f"{m['ref_mean']}±{m['ref_std']}", str(m["z_score"]))
    console.print(mtable)

    console.print("\n[dim]Generating forensic report (LLM if available, "
                  "template otherwise)…[/]")
    report = report_mod.generate_report(case["case_id"])
    ex = report["examination"]
    console.print(Panel(ex["summary"] + "\n\n" + ex["conclusion"],
                        title=f"Explanation ({report['source']})", border_style="grey50"))
    console.print(f"Full report: [underline]{storage.case_dir(case['case_id'])}/report.pdf[/]")

    if Confirm.ask("Was this verdict correct?", default=True):
        add_ref = False
        if case["verdict"] == "GENUINE":
            add_ref = Confirm.ask(
                "  Add this signature to the profile as a new reference sample? "
                "(real test-time calibration — relearns the profile's statistics "
                "from it)", default=False)
        result = engine.feedback(case["case_id"], True, add_ref)
        _print_training_update(result)
    else:
        result = engine.feedback(case["case_id"], False)
        console.print("[yellow]Threshold adjusted based on your correction.[/]")
        _print_training_update(result)


def _print_training_update(result: dict) -> None:
    """v5 (Bug #23, PROJECT_MEMORY_v8.md): the terminal equivalent of the
    web UI's persistent training panel — previously the CLI's feedback
    flow (like the old web alert()) only printed one line and threw away
    everything else `engine.feedback()` returns."""
    console.print(Panel.fit(
        f"Adaptive threshold: {result['threshold']}\n"
        f"Profile maturity: {result['maturity']} "
        f"({result['feedback_count']} feedback event(s), "
        f"{result['n_references']} reference sample(s))\n"
        f"Promoted to new reference sample: "
        f"{'yes — profile statistics relearned' if result['promoted_to_reference'] else 'no'}",
        title="Training & adaptation update", border_style="grey50"))


def do_case_log() -> None:
    cases = storage.list_cases(30)
    table = Table(title="Recent cases")
    table.add_column("Profile"); table.add_column("Verdict"); table.add_column("Score")
    table.add_column("Confidence"); table.add_column("When")
    for c in cases:
        style = VERDICT_STYLE.get(c["verdict"], "white")
        table.add_row(c["profile_name"], f"[{style}]{c['verdict']}[/]",
                     str(c["score"]), f"{c['confidence']*100:.1f}%", c["created_at"])
    console.print(table if cases else "[dim]No cases yet.[/]")


def do_llm() -> None:
    mgr = get_manager()
    s = mgr.status()
    gpu_line = ""
    if s["gpu_detected"]:
        if s["gpu_offload_active"]:
            gpu_line = f"\nGPU offload: [green]confirmed active[/] ({s.get('gpu_name') or 'GPU'})"
        elif s.get("gpu_offload_unconfirmed"):
            gpu_line = "\nGPU offload: [yellow]unconfirmed[/] — check terminal for 'offloaded N/M layers to GPU'"
        elif s["status"] == "ready":
            gpu_line = ("\nGPU offload: [red]not active[/] — GPU hardware detected but llama.cpp's "
                       "own load log confirms 0 layers offloaded (likely a CPU-only backend; "
                       "see PROJECT_MEMORY_v8.md Bug #32)")
    console.print(Panel.fit(
        f"Status: {s['status']}\nModel: {s['model_id']}\n"
        f"Backend: {s['backend']}{gpu_line}\n{s['notes']}",
        title="Local vision-AI model", border_style="grey50"))
    if s["status"] == "not_installed" and Confirm.ask("Download this model now?", default=False):
        console.print("[dim]Downloading now (this blocks until done, so it "
                      "can't be silently interrupted if you exit the menu — "
                      "see PROJECT_MEMORY_v8.md Bug #17). This can take several "
                      "minutes.[/]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                     console=console) as prog:
            prog.add_task("Downloading model weights...", total=None)
            result = mgr.download_blocking()
        if result["ok"]:
            console.print("[green]Model ready.[/]")
        else:
            console.print(f"[red]Download failed: {result['reason']}[/]")


def do_device() -> None:
    d = detect_device()
    table = Table(title="Device diagnostics")
    for k, v in d.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
