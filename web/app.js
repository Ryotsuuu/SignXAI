/* SignXAI frontend — vanilla JS, no build step, so `python main.py` is
   genuinely the only command needed (see PROJECT_MEMORY_v6.md process rule #2:
   no bash-only setup; this extends the spirit to "no npm build step" too). */

const API = "/api";

// ---------------------------------------------------------------- routing
document.querySelectorAll(".navlink").forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const view = link.dataset.view;
    document.querySelectorAll(".navlink").forEach(l => l.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    link.classList.add("active");
    document.getElementById(`view-${view}`).classList.add("active");
    if (view === "verify") loadProfiles();
    if (view === "cases") loadCases();
    if (view === "llm") loadLLMStatus();
  });
});

// ------------------------------------------------------------- sign pads
function setupPad(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#121212"; ctx.lineWidth = 2.2; ctx.lineCap = "round";
  let drawing = false, last = null;

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return { x: (t.clientX - r.left) * (canvas.width / r.width),
             y: (t.clientY - r.top) * (canvas.height / r.height) };
  }
  function start(e) { drawing = true; last = pos(e); e.preventDefault(); }
  function move(e) {
    if (!drawing) return;
    const p = pos(e);
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    last = p; e.preventDefault();
  }
  function end() { drawing = false; }

  canvas.addEventListener("mousedown", start);
  canvas.addEventListener("mousemove", move);
  window.addEventListener("mouseup", end);
  canvas.addEventListener("touchstart", start, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  canvas.addEventListener("touchend", end);

  return {
    clear: () => { ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height); },
    dataURL: () => canvas.toDataURL("image/png"),
  };
}

const padEnroll = setupPad(document.getElementById("padEnroll"));
const padVerify = setupPad(document.getElementById("padVerify"));
document.getElementById("clearEnroll").onclick = padEnroll.clear;
document.getElementById("clearVerify").onclick = padVerify.clear;

// -------------------------------------------------------------- enrollment
let pendingSamples = [];
let activeProfileId = null;

// v6 fix (Bug #29, PROJECT_MEMORY_v6.md, user-reported): thumbnails had no
// way to remove a sample once added -- a real UX gap, not just a missing
// nicety, since the only recovery from a bad sample was clearing the
// whole form. Renders from `pendingSamples` every time so the displayed
// thumbnails and the array actually being submitted can never drift out
// of sync with each other.
function renderThumbs() {
  const row = document.getElementById("enrollThumbs");
  row.innerHTML = pendingSamples.map((d, i) => `
    <div class="thumb-item">
      <img src="${d}">
      <button type="button" class="thumb-remove" title="Remove this sample"
              onclick="removeThumb(${i})">&times;</button>
    </div>`).join("");
}

function removeThumb(i) {
  pendingSamples.splice(i, 1);
  renderThumbs();
}

document.getElementById("addFromPad").onclick = () => {
  const d = padEnroll.dataURL();
  pendingSamples.push(d);
  renderThumbs();
  padEnroll.clear();
};

document.getElementById("fileEnroll").addEventListener("change", async (e) => {
  for (const file of e.target.files) {
    const d = await fileToDataURL(file);
    pendingSamples.push(d);
  }
  renderThumbs();
});

function fileToDataURL(file) {
  return new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(file);
  });
}

let nameCheckTimer = null;
document.getElementById("enrollName").addEventListener("input", (e) => {
  clearTimeout(nameCheckTimer);
  const name = e.target.value.trim();
  const box = document.getElementById("nameMatchWarning");
  if (!name) { box.classList.add("hidden"); activeProfileId = null; return; }
  nameCheckTimer = setTimeout(async () => {
    const res = await fetch(`${API}/profiles/check-name`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (data.needs_disambiguation) {
      const m = data.matches[0];
      activeProfileId = m.id;
      box.innerHTML = `Found an existing profile named "<b>${m.name}</b>" ` +
        `(${m.n_references} sample(s) enrolled). New samples will be added ` +
        `to <b>that</b> profile. If this is a different person, use a more ` +
        `specific name (e.g. add an initial or ID).`;
      box.classList.remove("hidden");
    } else {
      activeProfileId = null;
      box.classList.add("hidden");
    }
  }, 400);
});

document.getElementById("submitEnroll").onclick = async () => {
  const name = document.getElementById("enrollName").value.trim();
  const resultBox = document.getElementById("enrollResult");
  if (!name) { alert("Enter a signer name."); return; }
  if (pendingSamples.length === 0) { alert("Add at least one signature (draw or upload)."); return; }

  try {
    let profileId = activeProfileId;
    if (!profileId) {
      const res = await fetch(`${API}/profiles`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (res.status === 409) {
        const err = await res.json();
        resultBox.textContent = err.detail;
        resultBox.classList.remove("hidden");
        return;
      }
      const data = await res.json();
      profileId = data.profile_id;
    }
    const enrollRes = await fetch(`${API}/profiles/${profileId}/enroll`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: pendingSamples }),
    });
    const stats = await enrollRes.json();
    const src = stats.threshold_std_source || "";
    const srcNote = src.startsWith("blended") ?
      ` (threshold uses a conservative safety margin — ${src}; enroll more samples to tighten it)` :
      src === "observed" ? " (threshold calibrated from your enrolled samples)" : "";
    resultBox.innerHTML = `Profile <b>${name}</b> (${profileId}) now has ` +
      `${stats.n_references} reference sample(s). Adaptive threshold: ` +
      `${stats.threshold}${srcNote}. ${stats.n_references < 2 ?
        "Add at least one more sample for a personalised threshold." : ""}`;
    resultBox.classList.remove("hidden");
    pendingSamples = []; activeProfileId = profileId;
    renderThumbs();
  } catch (err) {
    resultBox.textContent = "Error: " + err;
    resultBox.classList.remove("hidden");
  }
};

// ---------------------------------------------------------------- verify
async function loadProfiles() {
  const res = await fetch(`${API}/profiles`);
  const profiles = await res.json();
  const sel = document.getElementById("verifyProfile");
  sel.innerHTML = profiles.map(p =>
    `<option value="${p.id}">${p.name} — ${p.n_references} sample(s)</option>`).join("");
}

document.getElementById("fileVerify").addEventListener("change", async (e) => {
  if (e.target.files[0]) {
    const d = await fileToDataURL(e.target.files[0]);
    const img = new Image();
    img.onload = () => {
      const cv = document.getElementById("padVerify");
      const ctx = cv.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.drawImage(img, 0, 0, cv.width, cv.height);
    };
    img.src = d;
  }
});

document.getElementById("submitVerify").onclick = async () => {
  const profileId = document.getElementById("verifyProfile").value;
  if (!profileId) { alert("Enroll a profile first."); return; }
  const image = padVerify.dataURL();
  const panel = document.getElementById("verifyResult");
  panel.classList.remove("hidden");
  panel.innerHTML = "<p>Analyzing…</p>";

  try {
    const res = await fetch(`${API}/verify`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_id: profileId, image }),
    });
    const data = await res.json();
    if (!res.ok) { panel.innerHTML = `<p>Error: ${data.detail}</p>`; return; }
    renderVerifyResult(data.case, data.report);
  } catch (err) {
    panel.innerHTML = `<p>Error: ${err}</p>`;
  }
};

function renderVerifyResult(c, report) {
  const panel = document.getElementById("verifyResult");
  const cls = c.verdict.toLowerCase();
  const assets = c.assets;
  const metricsRows = c.metrics.slice(0, 21).map((m, i) => `
    <tr class="${m.deviation ? "deviant" : ""} metric-row" onclick="toggleMetric(${i})">
      <td>${m.label} <span class="expand-hint">▸</span></td><td>${m.category}</td>
      <td>${m.query_value}</td><td>${m.ref_mean} ± ${m.ref_std}</td>
      <td>${m.z_score}</td>
    </tr>
    <tr id="metric-detail-${i}" class="metric-detail hidden">
      <td colspan="5">
        <b>What this measures:</b> ${m.description}<br><br>
        <b>This signature:</b> ${m.interpretation}
      </td>
    </tr>`).join("");

  const ex = report.examination;

  // v5: per-reference transparency -- "goes through each sample image"
  const refRows = (c.reference_scores || []).map(r => `
    <tr class="${r.score >= c.threshold ? "" : "deviant"}">
      <td>${r.ref_image}</td><td>${r.score}</td>
      <td>${r.aligned_ok ? "yes" : "no (identity fallback)"}</td>
    </tr>`).join("");
  const agg = c.aggregate || {};
  const edge = c.edge_case_review || {};
  const vr = c.verdict_reason || null;

  // v6 (Bug #29, PROJECT_MEMORY_v6.md): the results view is now split into
  // three explicitly labeled sections instead of one long undifferentiated
  // scroll -- (1) the engine's own analysis (verdict, per-reference scores,
  // metrics, SHAP -- "our own codebase response"), (2) a High-Level
  // Visualization section grouping every evidence image together, and
  // (3) a Local AI Explanation section holding ONLY the narrative model's
  // (or template's) prose, visually distinguished (sky-tint card) so it's
  // never confused with the engine's own decision evidence above it.
  panel.innerHTML = `
    <div class="verdict-banner ${cls}">
      <div>
        <div class="verdict-title">${c.verdict}</div>
        <div class="verdict-confidence">Fused score ${c.score} vs. threshold ${c.threshold} ·
          best-match ${(agg.n_references_compared || 0)} of ${c.enrollment.n_references} reference sample(s) compared,
          prototype score ${agg.prototype_score}</div>
      </div>
    </div>

    ${edge.triggered ? `
    <div class="warning-box">
      <b>Edge-case review flagged this verdict.</b> The fused score alone pointed to
      "${c.original_verdict}", but the independent re-check below found the individual
      per-reference comparisons don't fully support that call, so this was routed to
      INCONCLUSIVE / human review instead:
      <ul>${edge.flags.map(f => `<li>${f}</li>`).join("")}</ul>
      Reference agreement: ${Math.round((edge.reference_agreement || 0) * 100)}% ·
      Alignment reliability: ${Math.round((edge.alignment_reliability || 0) * 100)}%
    </div>` : ""}

    ${vr ? `
    <div class="reason-box reason-${vr.reason_code}">
      <b>${({clear_margin: "Why this verdict:", borderline_band: "Why INCONCLUSIVE — borderline score:",
             edge_case_downgrade: "Why INCONCLUSIVE — evidence disagreement:"})[vr.reason_code] || "Why this verdict:"}</b>
      <p>${vr.explanation}</p>
      ${vr.reason_code !== "clear_margin" ? `
      <div class="reason-diagnostics">
        Reference agreement: ${Math.round((edge.reference_agreement ?? 1) * 100)}% ·
        Alignment reliability: ${Math.round((edge.alignment_reliability ?? 1) * 100)}% ·
        Opposing evidence: ${Math.round((edge.opposing_evidence_fraction ?? 0) * 100)}%
        ${(edge.informational_flags || []).length ? `<ul>${edge.informational_flags.map(f => `<li>${f}</li>`).join("")}</ul>` : ""}
      </div>` : ""}
    </div>` : ""}

    <div class="section-block">
      <div class="section-eyebrow">Engine analysis</div>
      <div class="section-heading">What the codebase itself found</div>
      <p class="subsection-sub">Classical, per-person-calibrated comparison — this is what actually
        decided the verdict above. Never influenced by the AI explanation section below.</p>

      <table class="metrics-table">
        <thead><tr><th>Reference sample</th><th>Fused score</th><th>Alignment</th></tr></thead>
        <tbody>${refRows}</tbody>
      </table>

      <div style="height:16px"></div>
      <table class="metrics-table">
        <thead><tr><th>Feature</th><th>Category</th><th>Query</th><th>Reference (mean±std)</th><th>Z-score</th></tr></thead>
        <tbody>${metricsRows}</tbody>
      </table>

      <div style="height:16px"></div>
      <div class="narrative-card">
        <div class="narrative-source">SHAP attribution — which measurements drove this verdict (real Shapley-value analysis, not a heuristic)</div>
        ${c.shap_findings.map(f => `<div style="display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid var(--hairline);">
          <span>${f.label}</span>
          <span style="color:${f.shap_value > 0 ? 'var(--genuine)' : 'var(--forged)'}">${f.shap_value > 0 ? '+' : ''}${f.shap_value} (${f.direction})</span>
        </div>`).join("")}
      </div>
    </div>

    <div class="section-block">
      <div class="section-eyebrow">High-level visualization</div>
      <div class="section-heading">Visual evidence</div>
      <p class="subsection-sub">Heatmap, ink agreement, stroke skeleton, and per-reference score chart —
        generated from the comparison above, for human inspection.</p>
      <div class="evidence-grid">
        <figure><img src="${API}/cases/${c.case_id}/asset/${assets.heatmap.split('/').pop()}"><figcaption>Divergence heatmap (best-matching reference)</figcaption></figure>
        <figure><img src="${API}/cases/${c.case_id}/asset/${assets.overlay.split('/').pop()}"><figcaption>Ink agreement (green=match, red=query-only, blue=ref-only)</figcaption></figure>
        <figure><img src="${API}/cases/${c.case_id}/asset/${assets.skeleton.split('/').pop()}"><figcaption>Stroke skeleton comparison</figcaption></figure>
      </div>
      <div style="height:16px"></div>
      <div class="evidence-grid single-col">
        <figure><img src="${API}/cases/${c.case_id}/asset/${assets.zone_grid.split('/').pop()}"><figcaption>Zonal (local) stroke-pattern analysis — image divided into content-adaptive regions; red boxes show a flagged local structural mismatch (query/reference corner-count shown), green/amber show local agreement</figcaption></figure>
      </div>
      <div style="height:16px"></div>
      <div class="evidence-grid single-col">
        <figure><img src="${API}/cases/${c.case_id}/asset/${assets.agreement_chart.split('/').pop()}"><figcaption>Score vs. every enrolled reference + aggregate prototype</figcaption></figure>
      </div>
    </div>

    <div class="section-block">
      <div class="section-eyebrow">Local AI explanation model</div>
      <div class="section-heading">Plain-language narrative</div>
      <p class="subsection-sub">${report.source === "llm" ? "Written by the local vision AI model, grounded strictly in the metrics above." : "No AI model installed — this is a deterministic template built from the same metrics above."}</p>
      <div class="narrative-card ai-narrative">
        <div class="narrative-source">Source: ${report.source === "llm" ? "local vision AI model" : "deterministic template"}</div>
        <b>${ex.summary}</b>
        <p><b>Methodology:</b> ${ex.methodology || ""}</p>
        <p>${ex.conclusion}</p>
        <p><b>What this confidence means:</b> ${ex.confidence_explanation || ""}</p>
        <p><b>Recommendation:</b> ${ex.recommendation || ""}</p>
      </div>
    </div>

    <a class="btn-dark" style="width:fit-content" href="${API}/reports/${c.case_id}/pdf" target="_blank">Download full PDF report</a>

    <div class="section-block">
      <div class="section-eyebrow">Examiner feedback</div>
      <div class="section-heading">Confirm or correct this verdict</div>
      <div class="card">
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
          <button class="btn-ghost" id="fbCorrect-${c.case_id}" onclick="selectFeedback('${c.case_id}', true)">Correct</button>
          <button class="btn-ghost" id="fbWrong-${c.case_id}" onclick="selectFeedback('${c.case_id}', false)">Wrong</button>
        </div>
        <label style="display:flex;align-items:center;gap:8px;font-size:13px;margin-top:14px;">
          <input type="checkbox" id="addAsRef-${c.case_id}" ${c.verdict === "GENUINE" ? "" : "disabled"}>
          Add to profile if confirmed genuine
        </label>
        <label class="field-label">Remark (optional)</label>
        <textarea id="remark-${c.case_id}" class="text-input remark-block" rows="3"
                  placeholder="e.g. the tail flourish was cut off in this photo"></textarea>
        <div class="remark-actions">
          <button class="btn-dark" id="submitFb-${c.case_id}" disabled onclick="submitFeedback('${c.case_id}')">Submit feedback</button>
          <span id="fbHint-${c.case_id}" style="font-size:13px;color:var(--graphite)">Choose Correct or Wrong first.</span>
        </div>
      </div>
      <div class="card" id="trainingPanel-${c.case_id}" style="display:none;"></div>
    </div>
  `;
}

const _feedbackChoice = {};

function selectFeedback(caseId, correct) {
  _feedbackChoice[caseId] = correct;
  const okBtn = document.getElementById(`fbCorrect-${caseId}`);
  const noBtn = document.getElementById(`fbWrong-${caseId}`);
  okBtn.style.background = correct ? "var(--ink-black)" : "";
  okBtn.style.color = correct ? "#fff" : "";
  noBtn.style.background = !correct ? "var(--ink-black)" : "";
  noBtn.style.color = !correct ? "#fff" : "";
  document.getElementById(`submitFb-${caseId}`).disabled = false;
  document.getElementById(`fbHint-${caseId}`).textContent =
    "Add an optional remark, then submit.";
}

function toggleMetric(i) {
  document.getElementById(`metric-detail-${i}`).classList.toggle("hidden");
}

// v6 (Bug #29, PROJECT_MEMORY_v6.md, user-reported): feedback now requires
// an explicit choice (Correct/Wrong) plus an explicit Submit click, with
// an optional remark carried along -- instead of the previous instant-fire
// Confirm/Mark-wrong buttons with no way to add context.
async function submitFeedback(caseId) {
  const correct = _feedbackChoice[caseId];
  if (correct === undefined) return;
  const cb = document.getElementById(`addAsRef-${caseId}`);
  const addAsReference = !!(cb && cb.checked && correct);
  const remark = (document.getElementById(`remark-${caseId}`).value || "").trim();
  const submitBtn = document.getElementById(`submitFb-${caseId}`);
  submitBtn.disabled = true;
  submitBtn.textContent = "Submitting…";

  const res = await fetch(`${API}/cases/${caseId}/feedback`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ correct, add_as_reference: addAsReference, remark: remark || null }),
  });
  const data = await res.json();
  // v5 (Bug #23, PROJECT_MEMORY_v6.md): a visible, persistent training/TTC
  // panel replaces the old alert()-only popup, so the examiner can see
  // exactly what the feedback changed (threshold, weights, whether the
  // sample was promoted to a new reference) instead of a message that
  // disappears the instant it's dismissed.
  const target = document.getElementById(`trainingPanel-${caseId}`);
  if (target) {
    target.style.display = "";
    target.innerHTML = `
      <div class="narrative-source">Training &amp; adaptation update</div>
      <p><b>Feedback recorded:</b> ${correct ? "verdict confirmed correct" : "verdict marked wrong"}${remark ? ` \u2014 "${remark}"` : ""}</p>
      <p><b>Adaptive threshold:</b> now ${data.threshold}</p>
      <p><b>Profile maturity:</b> ${data.maturity} (${data.feedback_count} feedback event(s), ${data.n_references} reference sample(s))</p>
      <p><b>Promoted to new reference sample:</b> ${data.promoted_to_reference ? "yes \u2014 profile statistics were relearned" : "no"}</p>
    `;
  }
  submitBtn.textContent = "Feedback submitted";
}

// ------------------------------------------------------------------ cases
async function loadCases() {
  const res = await fetch(`${API}/cases`);
  const cases = await res.json();
  document.getElementById("caseTable").innerHTML = cases.map(c => `
    <div class="case-row">
      <div>${c.profile_name}</div>
      <div>${new Date(c.created_at).toLocaleString()}</div>
      <div>${c.score}</div>
      <div>${c.confidence}</div>
      <div><span class="badge ${c.verdict}">${c.verdict}</span></div>
    </div>`).join("") || "<p>No cases yet.</p>";
}

// -------------------------------------------------------------------- llm
let _llmPollTimer = null;

async function loadLLMStatus() {
  const res = await fetch(`${API}/llm/status`);
  const s = await res.json();
  const card = document.getElementById("llmCard");
  card.innerHTML = `
    <p><b>Status:</b> ${s.status}</p>
    <p><b>Model found on disk:</b> ${s.model_present_on_disk ? `yes (${s.model_size_on_disk_gb} GB)` : "no"}</p>
    <p><b>Hardware viable:</b> ${s.status !== 'disabled' ? 'yes' : 'no'}</p>
    <p><b>GPU detected:</b> ${s.gpu_detected ? `yes — ${s.gpu_name || 'unnamed'} (${s.gpu_vram_gb ?? '?'} GB VRAM)` : 'no'}</p>
    <p><b>GPU offload active:</b> ${s.gpu_offload_active ? 'yes' : (s.gpu_offload_unconfirmed ? 'unconfirmed — see notes below' : 'no')}</p>
    ${s.gpu_skip_reason ? `<p class="warning-box"><b>Why CPU, not GPU:</b> ${s.gpu_skip_reason}</p>` : ''}
    ${s.gpu_offload_unconfirmed ? '<p class="lede">GPU hardware was detected and the model loaded, but llama.cpp\'s own load log did not give a definitive confirmation either way for this backend version — check your terminal output for a line like "offloaded N/M layers to GPU" to be certain.</p>' : ''}
    ${(s.gpu_detected && !s.gpu_offload_active && !s.gpu_offload_unconfirmed && !s.gpu_skip_reason && s.status === 'ready') ? '<p class="warning-box">GPU hardware was detected, but llama.cpp\'s own load log confirms 0 layers were actually offloaded — the installed backend likely does not have GPU support compiled in (a common result of the automatic CPU-only fallback when a GPU wheel fails to load, see PROJECT_MEMORY_v7.md Bug #30/#32). Generation will work, just slower (CPU inference) than a real GPU build would be.</p>' : ''}
    <p><b>Recommended model:</b> ${s.model_id || "none — hardware below minimum"}</p>
    <p><b>Backend:</b> ${s.backend}</p>
    <p><b>Context window (n_ctx):</b> ${s.n_ctx ?? '?'} tokens &nbsp; <b>Load timeout:</b> ${s.load_timeout_seconds ?? '?'}s</p>
    <p>${s.notes}</p>
    ${s.status === "not_installed" ? '<button class="btn-dark" onclick="downloadLLM()">Download model</button>' : ""}
    ${s.status === "downloaded" ? '<button class="btn-dark" onclick="loadLLM()">Load model now</button><p class="lede">Model is on disk but not loaded into memory yet — this normally happens automatically a few seconds after the server starts; click if it hasn\'t.</p>' : ""}
    ${s.status === "loading" ? `<p>${s.progress || "Loading into memory..."}</p><p class="lede">This panel refreshes itself every few seconds while loading.</p>` : ""}
    ${s.status === "error" ? `<button class="btn-dark" onclick="loadLLM()">Retry load</button>` : ""}
    ${s.error ? `<p style="color:#b23b2e">${s.error}</p>` : ""}
  `;
  refreshPill(s);
  // v7.1 (Bug #46, PROJECT_MEMORY_v7.md, user-reported: "show me its
  // reasoning" -- a static "loading" with no auto-refresh looked
  // indistinguishable from frozen, which is exactly what the backend's
  // heartbeat progress message (Bug #43) exists to prevent. Poll while
  // loading so the progress text (and eventual ready/error transition)
  // actually appears without the user manually refreshing the page.
  clearTimeout(_llmPollTimer);
  if (s.status === "loading") {
    _llmPollTimer = setTimeout(loadLLMStatus, 4000);
  }
}

async function downloadLLM() {
  await fetch(`${API}/llm/download`, { method: "POST" });
  alert("Download started in the background. This can take several minutes " +
        "depending on model size and connection. Refresh this page to check progress.");
}

async function loadLLM() {
  await fetch(`${API}/llm/load`, { method: "POST" });
  alert("Loading started in the background. Refresh this page in a few " +
        "seconds to check progress.");
}

async function refreshPill(s) {
  const pill = document.getElementById("llmPill");
  pill.textContent = `LLM: ${s ? s.status : "…"}`;
}

fetch(`${API}/llm/status`).then(r => r.json()).then(refreshPill).catch(() => {});
