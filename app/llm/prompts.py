"""Prompts and the structured output contract for the forensic vision LLM."""
from __future__ import annotations

import json

FORENSIC_SYSTEM_PROMPT = (
    "You are Dr. Sigma, a certified forensic document examiner specialising in "
    "handwritten signature verification, writing a report for a non-technical "
    "reader (a bank officer, HR staff, or the signer themselves) who needs to "
    "understand not just WHAT the verdict is but WHY, in plain language, without "
    "losing forensic rigor. You examine signature evidence with laboratory "
    "rigour: stroke construction, pen-lift patterns, proportions, slant, "
    "pressure cues, spacing and overall pictorial gestalt. You are "
    "conservative: you only agree with a GENUINE verdict when the evidence is "
    "consistent, and you say so plainly when evidence is genuinely ambiguous "
    "rather than forcing false confidence. Write substantively — each section "
    "should be a real paragraph a human would find informative, not a one-line "
    "placeholder. You ALWAYS answer with strict JSON only — no markdown, no "
    "commentary outside the JSON object."
)

OUTPUT_CONTRACT = """{
  "verdict_agreement": true or false,
  "summary": "3-5 sentence executive summary of the examination, written for a non-expert reader",
  "findings": [
    {"region": [x1, y1, x2, y2],
     "title": "short finding title",
     "detail": "what is observed in this region and why it matters forensically -- be specific about the visual evidence, not just the number",
     "severity": "low" | "medium" | "high"}
  ],
  "category_analysis": {
    "geometry": "...", "stroke": "...", "shape": "...",
    "texture": "...", "topology": "..."
  },
  "methodology": "2-4 sentences in plain language explaining HOW this comparison was performed -- preprocessing, alignment, what was measured -- so a non-technical reader trusts the process, not just the number",
  "confidence_explanation": "2-3 sentences explaining what the confidence percentage means IN THIS SPECIFIC CASE -- reference how many enrolled samples this profile has and what that does/doesn't allow you to be certain about",
  "recommendation": "1-2 sentences of concrete, actionable next steps for the reader given this verdict",
  "conclusion": "one full paragraph formal conclusion, lab-report tone, that a reader could act on without needing to re-read the rest"
}"""


def build_user_prompt(case: dict) -> str:
    """Assemble the evidence prompt from a verification case."""
    metrics_brief = [
        {"label": m["label"], "category": m["category"], "z": m["z_score"],
         "deviation": m["deviation"]}
        for m in case["metrics"][:12]
    ]
    payload = {
        "engine_verdict": case["verdict"],
        "engine_confidence": case["confidence"],
        "fused_score": case["score"],
        "decision_threshold": case["threshold"],
        "component_similarities": case["components"],
        "most_deviant_features": metrics_brief,
        "enrollment": case["enrollment"],
    }
    return (
        "Examine the attached exhibits in order: (1) the QUESTIONED signature, "
        "(2) the enrolled REFERENCE signature, (3) a divergence heatmap where "
        "warm colours mark regions that differ after alignment.\n\n"
        f"Quantitative engine measurements:\n{json.dumps(payload, indent=2)}\n\n"
        "Respond with ONE JSON object exactly matching this contract "
        "(regions are integer pixel coordinates normalised to a 1000x1000 canvas, "
        "x1<x2 and y1<y2, at most 5 findings):\n"
        f"{OUTPUT_CONTRACT}"
    )


def repair_prompt(prev_output: str, error: str) -> str:
    """Feedback prompt for the validation-retry loop."""
    return (
        "Your previous answer failed validation. Error: "
        f"{error}\n\nPrevious answer:\n{prev_output[:1500]}\n\n"
        "Return a corrected JSON object only — strict JSON, valid regions, "
        "no commentary."
    )
