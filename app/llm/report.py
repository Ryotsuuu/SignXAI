"""Forensic report generation.

Pipeline (loop-engineered):

1. Build the evidence pack (metrics + exhibit images).
2. If the vision LLM is ready, ask it for a structured examination,
   validating the JSON contract on every attempt and feeding validation
   errors back into the next attempt (max 3 attempts).
3. On LLM absence/failure, produce the same structured examination with a
   deterministic, metrics-driven template — the report is always complete.
4. Render deliverables: ``report.json``, ``report.html``, ``report.pdf``
   and the LLM/template-annotated exhibit image.
"""
from __future__ import annotations

import base64
import html
import json
import logging
import time
from pathlib import Path

import cv2
from pydantic import BaseModel, Field, field_validator

from ..core import explain, storage
from . import prompts
from .manager import get_manager

log = logging.getLogger("signxai.report")

MAX_LLM_ATTEMPTS = 3


# ------------------------------------------------------------- JSON contract
class Finding(BaseModel):
    region: list[int] = Field(min_length=4, max_length=4)
    title: str
    detail: str
    severity: str = "medium"

    @field_validator("region")
    @classmethod
    def _region_ok(cls, v: list[int]) -> list[int]:
        if not all(0 <= c <= 1000 for c in v):
            raise ValueError("region coords must be within 0..1000")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("region must satisfy x1<x2 and y1<y2")
        return v

    @field_validator("severity")
    @classmethod
    def _sev_ok(cls, v: str) -> str:
        v = v.lower()
        if v not in ("low", "medium", "high"):
            raise ValueError("severity must be low|medium|high")
        return v


class Examination(BaseModel):
    verdict_agreement: bool
    summary: str
    findings: list[Finding] = Field(default_factory=list, max_length=5)
    category_analysis: dict[str, str] = Field(default_factory=dict)
    conclusion: str
    methodology: str = ""              # v3: how the comparison was performed
    confidence_explanation: str = ""   # v3: what the confidence % actually means here
    recommendation: str = ""           # v3: concrete next step for the examiner/user


def _extract_json(text: str) -> dict:
    """Tolerantly pull the first JSON object out of an LLM response."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                t = p
                break
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object found in response")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unbalanced braces in response")


# ------------------------------------------------------ template fallback
_CATEGORY_FEATURES = {
    "Geometry": "overall proportions, placement and slant of the signature",
    "Stroke": "stroke width, pen-pressure cues and stroke construction",
    "Topology": "pen lifts, stroke connectivity, junctions and loops",
    "Shape": "global shape and contour structure",
    "Texture": "fine ink texture and local writing rhythm",
}


def template_examination(case: dict) -> Examination:
    """Deterministic forensic examination driven purely by the metrics."""
    deviants = [m for m in case["metrics"] if m["deviation"]]
    cats: dict[str, list] = {}
    for m in deviants:
        cats.setdefault(m["category"], []).append(m)

    verdict = case["verdict"]
    conf_pct = round(case["confidence"] * 100, 1)
    if verdict == "GENUINE":
        summary = (f"The questioned signature is consistent with the enrolled "
                   f"reference specimens of {case['profile_name']} within the "
                   f"signer's natural variation (fused similarity "
                   f"{case['score']:.3f} ≥ threshold {case['threshold']:.3f}, "
                   f"confidence {conf_pct}%).")
    elif verdict == "INCONCLUSIVE":
        summary = (f"The questioned signature shows both agreements and "
                   f"disagreements with the enrolled references (score "
                   f"{case['score']:.3f} vs threshold {case['threshold']:.3f}); "
                   f"the evidence is insufficient for a definitive opinion.")
    else:
        summary = (f"The questioned signature falls outside the natural "
                   f"variation envelope of the enrolled references "
                   f"(score {case['score']:.3f} < threshold "
                   f"{case['threshold']:.3f}, confidence {conf_pct}%). "
                   f"{len(deviants)} forensic features deviate significantly.")

    findings = []
    for f in case.get("findings", []):
        try:
            findings.append(Finding(**f))
        except Exception:
            continue

    category_analysis = {}
    for cat, desc in _CATEGORY_FEATURES.items():
        if cat in cats:
            worst = max(cats[cat], key=lambda m: abs(m["z_score"]))
            category_analysis[cat] = (
                f"Significant deviation in {desc}: '{worst['label']}' measured "
                f"{worst['query_value']} vs reference mean {worst['ref_mean']} "
                f"(z = {worst['z_score']}).")
        else:
            category_analysis[cat] = (
                f"No significant deviation observed in {desc} (all |z| ≤ 2).")

    if verdict == "GENUINE":
        conclusion = ("Based on the quantitative comparison of enrolled "
                      "exemplars and the questioned specimen, the observed "
                      "agreements in " +
                      (", ".join(c.lower() for c in _CATEGORY_FEATURES
                                 if c not in cats) or "all measured categories") +
                      " support the opinion that the signature is consistent "
                      "with the reference writer.")
    else:
        conclusion = ("The deviations identified in " +
                      (", ".join(c.lower() for c in cats) or "multiple categories") +
                      " exceed the range of natural variation established from the "
                      "enrolled exemplars. The evidence supports the opinion that "
                      "the questioned signature was not produced by the reference "
                      "writer." if verdict == "FORGED" else
                      "The evidence is mixed; additional reference exemplars are "
                      "recommended before a definitive opinion is offered.")
    n_refs = case.get("enrollment", {}).get("n_references", "unknown")
    methodology = (
        f"This examination compared the questioned signature against "
        f"{n_refs} enrolled genuine reference sample(s) for {case['profile_name']}. "
        f"Both the questioned and reference images were preprocessed identically: "
        f"grayscale conversion, Otsu adaptive thresholding to separate ink from "
        f"background, noise-speck removal, cropping to the ink bounding box, and "
        f"normalisation onto a fixed-size canvas so images of different original "
        f"sizes and resolutions are directly comparable. The questioned signature "
        f"was then rigidly aligned (translation/rotation/scale) to each reference "
        f"before comparison, to avoid penalising incidental differences in how the "
        f"page was photographed or scanned. Similarity was computed by fusing "
        f"seven independent comparison methods: structural similarity (SSIM) of "
        f"the aligned ink, ink-overlap ratio (IoU), two texture/gradient "
        f"descriptors (HOG and LBP cosine similarity), shape-invariant Hu moment "
        f"comparison, dynamic time warping of the stroke path, and a statistical "
        f"comparison of {len(case.get('metrics', []))} individual geometric/stroke/"
        f"topology features against this signer's own enrolled-sample statistics. "
        f"No external dataset or pretrained model determined this verdict — every "
        f"number above was computed fresh from this signer's own enrolled samples."
    )

    if n_refs and isinstance(n_refs, int) and n_refs <= 2:
        confidence_note = (
            f"This profile has only {n_refs} enrolled reference sample(s). With so "
            f"few samples, the system cannot reliably estimate how much this "
            f"signer's OWN signature naturally varies from one signing to the "
            f"next — so the decision threshold includes a conservative built-in "
            f"safety margin rather than a tightly measured one. Practically: "
            f"borderline cases are more likely to be marked INCONCLUSIVE (routed "
            f"for human review) rather than confidently GENUINE or FORGED. "
            f"Enrolling 3-5 reference samples materially improves this."
        )
    else:
        confidence_note = (
            f"This profile has {n_refs} enrolled reference samples, enough to "
            f"estimate this signer's natural signature variation directly from "
            f"their own data (not from any external population). The confidence "
            f"percentage reflects how far the fused similarity score sits from "
            f"this signer's personal decision threshold, not a general population "
            f"accuracy statistic."
        )

    if verdict == "GENUINE":
        recommendation = ("No further action indicated based on this comparison "
                          "alone. For high-stakes decisions, corroborate with "
                          "other verification factors per your organisation's "
                          "policy — this is decision support, not a standalone "
                          "legal determination.")
    elif verdict == "FORGED":
        recommendation = ("Flag for manual review before relying on this "
                          "signature. Consider requesting the signer provide a "
                          "fresh sample under supervision, and review the "
                          "evidence images above for the specific regions that "
                          "drove this result.")
    else:
        recommendation = ("Route to a human examiner — the evidence does not "
                          "clearly support GENUINE or FORGED. If this profile has "
                          "fewer than 3 enrolled references, enrolling more "
                          "genuine samples is the single most effective way to "
                          "get a more decisive result on future verifications.")

    return Examination(verdict_agreement=True, summary=summary,
                       findings=findings, category_analysis=category_analysis,
                       conclusion=conclusion, methodology=methodology,
                       confidence_explanation=confidence_note,
                       recommendation=recommendation)


# -------------------------------------------------------------- LLM attempt
def _llm_examination(case: dict, case_dir: Path) -> Examination:
    manager = get_manager()
    if manager.status()["status"] == "disabled":
        raise RuntimeError("LLM disabled on this device")
    # v7.1 FIX (Bug #45, PROJECT_MEMORY_v8.md, user-reported: "CPU fallback
    # loads it but it didn't make LLM to provide explanation" -- confirmed
    # from the user's own log: 'Image' object has no attribute 'startswith'):
    # `VisionLLMManager.generate()` has always expected `images: list[str]`
    # (file paths or data-URLs -- see its docstring), but this call site
    # opened PIL `Image` objects and passed those instead. This is a
    # pre-existing type mismatch that predates v7 -- it was simply never
    # caught before because the LLM itself could never get past "loading"
    # (Bug #43) to actually reach this call. Fixing Bug #43 (the load hang)
    # is what finally let a real run reach this code path for the first
    # time and surface this second, independent bug. Fixed by passing the
    # file PATHS directly (as plain strings) -- `generate()` already knows
    # how to wrap a bare path in a `file://` URL itself; no PIL round-trip
    # is needed at all, since llama.cpp's own image loader reads the file
    # directly.
    for name in ("query.png", "reference.png", "heatmap.png"):
        if not (case_dir / name).exists():
            raise FileNotFoundError(
                f"Expected evidence image not found for LLM prompt: "
                f"{case_dir / name}")
    images = [str(case_dir / "query.png"), str(case_dir / "reference.png"),
             str(case_dir / "heatmap.png")]
    prompt = prompts.build_user_prompt(case)
    last_err = "n/a"
    last_out = ""
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        log.info("LLM examination attempt %d/%d", attempt, MAX_LLM_ATTEMPTS)
        out = manager.generate(images, prompt, prompts.FORENSIC_SYSTEM_PROMPT)
        last_out = out
        try:
            return Examination(**_extract_json(out))
        except Exception as exc:
            last_err = str(exc)
            prompt = prompts.repair_prompt(out, last_err)
    raise RuntimeError(f"LLM failed validation after {MAX_LLM_ATTEMPTS} "
                       f"attempts: {last_err}; last output: {last_out[:200]}")


# --------------------------------------------------------------- rendering
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _render_html(case: dict, ex: Examination, source: str,
                 annotated_llm: str | None) -> str:
    def img(uri_name: str, caption: str) -> str:
        p = storage.case_dir(case["case_id"]) / Path(case["assets"][uri_name]).name
        if not p.exists():
            return ""
        return (f'<figure><img src="data:image/png;base64,{_b64(p)}"/>'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")
    rows = "".join(
        f"<tr class={'dev' if m['deviation'] else ''}><td>{html.escape(m['label'])}</td>"
        f"<td>{html.escape(m['category'])}</td><td>{m['query_value']}</td>"
        f"<td>{m['ref_mean']}</td><td>{m['z_score']:+.2f}</td></tr>"
        for m in case["metrics"])
    findings = "".join(
        f"<li><b>{html.escape(f.title)}</b> "
        f"<span class='sev {f.severity}'>{f.severity.upper()}</span><br/>"
        f"{html.escape(f.detail)}</li>" for f in ex.findings) or "<li>None.</li>"
    cats = "".join(
        f"<li><b>{html.escape(k)}:</b> {html.escape(v)}</li>"
        for k, v in ex.category_analysis.items())
    shap_rows = "".join(
        f"<tr><td>{html.escape(s['label'])}</td><td>{html.escape(s['category'])}</td>"
        f"<td>{s['shap_value']:+.4f}</td><td>{html.escape(s['direction'])}</td></tr>"
        for s in case.get("shap_findings", [])) or "<tr><td colspan=4>Not available.</td></tr>"
    vclass = {"GENUINE": "g", "INCONCLUSIVE": "i", "FORGED": "f"}[case["verdict"]]
    llm_img = ""
    if annotated_llm:
        p = storage.case_dir(case["case_id"]) / annotated_llm
        if p.exists():
            llm_img = (f'<figure><img src="data:image/png;base64,{_b64(p)}"/>'
                       f"<figcaption>Examiner-annotated regions</figcaption></figure>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1321;color:#e6edf7;
max-width:960px;margin:24px auto;padding:0 20px}}
h1{{color:#22d3ee;font-size:22px}} h2{{color:#7dd3fc;font-size:16px;
border-bottom:1px solid #243047;padding-bottom:4px}}
.verdict{{font-size:26px;font-weight:700;padding:12px 18px;border-radius:8px;
display:inline-block;margin:8px 0}}
.g{{background:#052e16;color:#4ade80}} .i{{background:#3a2b05;color:#fbbf24}}
.f{{background:#3b0a0a;color:#f87171}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #243047;padding:5px 8px;text-align:left}}
tr.dev td{{background:#2a1215;color:#fca5a5}}
figure{{display:inline-block;margin:8px}} img{{max-width:440px;border:1px solid
#243047;border-radius:6px}} figcaption{{font-size:12px;color:#8b9bb8}}
.sev{{font-size:11px;padding:1px 6px;border-radius:4px}}
.sev.low{{background:#052e16;color:#4ade80}} .sev.medium{{background:#3a2b05;
color:#fbbf24}} .sev.high{{background:#3b0a0a;color:#f87171}}
.meta td{{background:#111a2e}} .disc{{font-size:11px;color:#64748b;
margin-top:24px}}</style></head><body>
<h1>Forensic Signature Examination Report</h1>
<table class="meta">
<tr><td>Case ID</td><td>{case['case_id']}</td><td>Date</td><td>{case['created_at']}</td></tr>
<tr><td>Subject</td><td>{html.escape(case['profile_name'])}</td>
<td>Analysis source</td><td>{'Vision LLM (local)' if source == 'llm' else 'Deterministic forensic engine'}</td></tr>
<tr><td>Fused score</td><td>{case['score']:.4f}</td>
<td>Decision threshold</td><td>{case['threshold']:.4f}</td></tr>
</table>
<div class="verdict {vclass}">{case['verdict']} — {round(case['confidence'] * 100, 1)}% confidence</div>
<h2>Executive summary</h2><p>{html.escape(ex.summary)}</p>
<h2>Methodology</h2><p>{html.escape(ex.methodology)}</p>
<h2>Exhibits</h2>
{img('query', 'Questioned signature')}{img('reference', 'Enrolled reference')}
{img('heatmap', 'Divergence heatmap')}{img('overlay', 'Ink agreement overlay')}
{img('skeleton', 'Stroke skeleton comparison')}{img('annotated', 'Engine-annotated divergences')}
{img('zone_grid', 'Zonal (local) stroke-pattern analysis')}
{llm_img}
<h2>Quantitative feature analysis</h2>
<table><tr><th>Feature</th><th>Category</th><th>Questioned</th>
<th>Reference mean</th><th>z-score</th></tr>{rows}</table>
<h2>SHAP attribution (Shapley-value analysis)</h2>
<p>Real Shapley-value attribution — how much each measurement individually
pushed the fused score toward or away from a match, computed via
KernelSHAP over the classical features (not a heuristic ranking).</p>
<table><tr><th>Feature</th><th>Category</th><th>SHAP value</th><th>Direction</th></tr>{shap_rows}</table>
<h2>Findings</h2><ul>{findings}</ul>
<h2>Category analysis</h2><ul>{cats}</ul>
<h2>What this confidence level means</h2><p>{html.escape(ex.confidence_explanation)}</p>
<h2>Conclusion</h2><p>{html.escape(ex.conclusion)}</p>
<h2>Recommendation</h2><p>{html.escape(ex.recommendation)}</p>
<p class="disc">This report was generated by SignXAI, an automated explainable-AI
system. It is intended as decision support and does not constitute a legally
admissible forensic opinion. Examination by a certified human forensic
document examiner is recommended for legal proceedings.</p>
</body></html>"""


def _render_pdf(case: dict, ex: Examination, source: str, out: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm)
    story = [Paragraph("Forensic Signature Examination Report", styles["Title"]),
             Spacer(1, 6)]
    meta = [["Case ID", case["case_id"], "Date", case["created_at"]],
            ["Subject", case["profile_name"], "Source",
             "Vision LLM" if source == "llm" else "Forensic engine"],
            ["Score", f"{case['score']:.4f}", "Threshold", f"{case['threshold']:.4f}"]]
    t = Table(meta, colWidths=[25 * mm, 60 * mm, 30 * mm, 55 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [t, Spacer(1, 10)]
    vcol = {"GENUINE": colors.green, "INCONCLUSIVE": colors.orange,
            "FORGED": colors.red}[case["verdict"]]
    vt = Table([[f"{case['verdict']} — {round(case['confidence'] * 100, 1)}% confidence"]])
    vt.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (0, 0), vcol),
                            ("FONTSIZE", (0, 0), (0, 0), 14),
                            ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold")]))
    story += [vt, Spacer(1, 10),
              Paragraph("Executive summary", styles["Heading2"]),
              Paragraph(ex.summary, styles["BodyText"]), Spacer(1, 8),
              Paragraph("Methodology", styles["Heading2"]),
              Paragraph(ex.methodology, styles["BodyText"]), Spacer(1, 8)]
    cdir = storage.case_dir(case["case_id"])
    imgs = []
    for key, cap in (("query", "Questioned"), ("reference", "Reference"),
                     ("heatmap", "Heatmap"), ("annotated", "Annotated"),
                     ("zone_grid", "Zonal analysis")):
        p = cdir / Path(case["assets"][key]).name
        if p.exists():
            imgs.append(RLImage(str(p), width=80 * mm, height=40 * mm))
    for i in range(0, len(imgs), 2):
        story.append(Table([imgs[i:i + 2]]))
        story.append(Spacer(1, 4))
    story += [Spacer(1, 8), Paragraph("Quantitative feature analysis",
                                      styles["Heading2"])]
    rows = [["Feature", "Category", "Questioned", "Ref mean", "z"]]
    for m in case["metrics"]:
        rows.append([m["label"], m["category"], str(m["query_value"]),
                     str(m["ref_mean"]), f"{m['z_score']:+.2f}"])
    ft = Table(rows, colWidths=[52 * mm, 28 * mm, 32 * mm, 32 * mm, 20 * mm],
               repeatRows=1)
    ft.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    story += [ft, Spacer(1, 8), Paragraph("SHAP attribution (Shapley-value analysis)",
                                          styles["Heading2"]),
              Paragraph("Real Shapley-value attribution — how much each measurement "
                       "individually pushed the fused score toward or away from a "
                       "match, computed via KernelSHAP over the classical features.",
                       styles["BodyText"])]
    shap_rows = [["Feature", "Category", "SHAP value", "Direction"]]
    for s in case.get("shap_findings", []):
        shap_rows.append([s["label"], s["category"], f"{s['shap_value']:+.4f}", s["direction"]])
    st = Table(shap_rows, colWidths=[52 * mm, 28 * mm, 30 * mm, 40 * mm], repeatRows=1)
    st.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    story += [st, Spacer(1, 8), Paragraph("Findings", styles["Heading2"])]
    for f in ex.findings:
        story.append(Paragraph(f"<b>{f.title}</b> [{f.severity.upper()}] — "
                               f"{f.detail}", styles["BodyText"]))
    story += [Spacer(1, 6), Paragraph("Category analysis", styles["Heading2"])]
    for k, v in ex.category_analysis.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["BodyText"]))
    story += [Spacer(1, 6), Paragraph("What this confidence level means", styles["Heading2"]),
              Paragraph(ex.confidence_explanation, styles["BodyText"]),
              Spacer(1, 6), Paragraph("Conclusion", styles["Heading2"]),
              Paragraph(ex.conclusion, styles["BodyText"]), Spacer(1, 6),
              Paragraph("Recommendation", styles["Heading2"]),
              Paragraph(ex.recommendation, styles["BodyText"]), Spacer(1, 12),
              Paragraph("Generated by SignXAI (automated explainable-AI system). "
                        "Decision support only — not a legally admissible "
                        "forensic opinion.", styles["Italic"])]
    doc.build(story)


# ------------------------------------------------------------------- public
def generate_report(case_id: str, use_llm: bool = True) -> dict:
    """Generate all report artefacts for a verification case."""
    case = storage.load_case(case_id)
    cdir = storage.case_dir(case_id)
    source, ex = "template", None
    if use_llm:
        from .manager import get_manager
        llm_status = get_manager().status()["status"]
        # v5 (Bug #22, PROJECT_MEMORY_v8.md): this gate used to only try the
        # LLM when status was ALREADY "ready" or "loading" -- but nothing
        # ever set status to "ready" except a load that only happened
        # INSIDE generate(), which only runs past this same gate. A model
        # sitting correctly on disk, on viable hardware, could never
        # transition out of "not_installed" through this path -- a
        # deadlock. "downloaded" (weights present, not yet loaded into
        # memory -- see device.py/manager.py) now also attempts the LLM;
        # generate() call below triggers the actual (lazy, thread-safe)
        # load, so the first request after startup may simply pay that
        # one-time cost if autostart's background thread hasn't finished
        # yet.
        if llm_status in ("ready", "loading", "downloaded"):
            try:
                ex = _llm_examination(case, cdir)
                source = "llm"
            except Exception as exc:
                log.warning("LLM examination unavailable, using template: %s", exc)
        else:
            log.info("LLM status is '%s' — skipping LLM attempt, using template "
                     "directly (avoids retrying an import that will fail "
                     "identically 3 times).", llm_status)
    if ex is None:
        ex = template_examination(case)

    annotated_llm = None
    qimg = cv2.imread(str(cdir / "query.png"), cv2.IMREAD_GRAYSCALE)
    if qimg is not None and ex.findings:
        annotated_llm = explain.annotate_regions(
            qimg, [f.model_dump() for f in ex.findings], cdir)
        case["assets"]["annotated_examiner"] = f"{case_id}/{annotated_llm}"

    report = {
        "case_id": case_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "examination": ex.model_dump(),
        "verdict": case["verdict"],
        "confidence": case["confidence"],
    }
    (cdir / "report.json").write_text(json.dumps(report, indent=2),
                                      encoding="utf-8")
    (cdir / "report.html").write_text(_render_html(case, ex, source,
                                                   annotated_llm),
                                      encoding="utf-8")
    try:
        _render_pdf(case, ex, source, cdir / "report.pdf")
    except Exception as exc:                        # pragma: no cover
        log.exception("PDF rendering failed: %s", exc)
    storage.save_case(case)
    report["urls"] = {
        "json": f"/api/reports/{case_id}",
        "html": f"/api/reports/{case_id}/html",
        "pdf": f"/api/reports/{case_id}/pdf",
    }
    return report
