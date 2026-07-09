// View: functions that turn job state into DOM

import { bulletList, el } from "./dom";
import { fmtDate, sourcesLabel } from "./format";
import { RefinementStatus } from "./types";
import type {
  ApproveHandler,
  ExecutionEvent,
  HallucinationAnalysis,
  JobResponse,
  RefineHandler,
  RelatedThesisResponse,
  ResumeHandler,
  SourceResponse,
  ThesisResponse,
  ThesisSummaryResponse,
} from "./types";

// Display cap for the refinement counters. Must match MAX_REFINEMENTS in the
// backend's refinement_graph.py; the server enforces the real limit and signals
// exhaustion via RefinementStatus.Escalated, which the UI honors regardless.
const MAX_REFINEMENTS = 3;

// --- Source articles ---

function renderSourceItem(s: SourceResponse): HTMLElement {
  const li = el("li");
  if (s.url) {
    const a = el("a", s.title);
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    li.append(a);
  } else {
    li.textContent = s.title;
  }
  return li;
}

function renderSources(sources: SourceResponse[]): HTMLElement | null {
  if (sources.length === 0) return null;
  const details = el("details");
  details.append(el("summary", sourcesLabel(sources)));
  const ul = el("ul");
  for (const s of sources) {
    // Skip a single malformed source rather than failing the whole list,
    // mirroring _sources_from_docs on the backend.
    try {
      ul.append(renderSourceItem(s));
    } catch (err) {
      console.warn("Skipping unrenderable source", s, err);
    }
  }
  details.append(ul);
  return details;
}

// --- Structured thesis ---

function renderThesis(thesis: ThesisResponse): HTMLElement {
  const section = el("section");

  const metrics = el("div");
  metrics.className = "metrics";
  metrics.append(el("div", `Investment Score: ${thesis.opportunity_score}/5`));

  const confidence = el("div");
  confidence.append(
    el("div", `Confidence Level: ${Math.trunc(thesis.confidence_level * 100)}%`),
  );
  if (thesis.confidence_as_of) {
    confidence.append(el("small", `trends as of ${thesis.confidence_as_of}`));
  }
  metrics.append(confidence);

  // data-rec lets CSS distinguish Pursue / Investigate / Skip without icons.
  const rec = el("div", `Recommendation: ${thesis.recommendation}`);
  rec.dataset["rec"] = thesis.recommendation;
  metrics.append(rec);
  section.append(metrics);

  section.append(el("h3", "Key Themes"));
  section.append(bulletList(thesis.key_themes, "No themes found."));

  section.append(el("h3", "Risks"));
  section.append(bulletList(thesis.risks, "No risks found."));
  if (thesis.key_risk_factors.length > 0) {
    section.append(
      el("p", `Key Risk Factors: ${thesis.key_risk_factors.join(", ")}`),
    );
  }

  section.append(el("h3", "Investment Signals"));
  section.append(bulletList(thesis.investment_signals, "No signals found."));

  return section;
}

// --- Related past theses (episodic recall) ---

function renderRelated(related: RelatedThesisResponse[]): HTMLElement | null {
  if (related.length === 0) return null;
  const details = el("details");
  details.append(el("summary", `Related past theses (${related.length})`));
  for (const r of related) {
    const item = el("div");
    const link = el("a", r.query);
    link.href = `?job_id=${encodeURIComponent(r.job_id)}`;
    item.append(link);

    const date = (r.created_at ?? "").slice(0, 10);
    const parts = [`score ${r.score}/5`, r.recommendation, date].filter(Boolean);
    let meta = parts.join(" - ");
    if (r.approved) meta += " - approved";
    item.append(el("p", meta));
    item.append(el("small", `similarity ${r.similarity}`));
    details.append(item);
  }
  return details;
}

// --- Approval ---

function renderApproval(job: JobResponse, onApprove: ApproveHandler): HTMLElement {
  const wrap = el("div");
  wrap.className = "approval";
  const button = el("button", "Approve");
  button.addEventListener("click", () => {
    button.disabled = true; // prevent double-submit; re-render replaces it
    onApprove(job.job_id);
  });
  wrap.append(button);
  return wrap;
}

// --- Refinement controls / escalation ---

function renderRefinementPanel(
  job: JobResponse,
  feedbackOptions: string[],
  onRefine: RefineHandler,
): HTMLElement {
  const section = el("section");
  section.className = "refinement";

  if (job.refinement_status === RefinementStatus.Escalated) {
    section.append(
      el(
        "p",
        `Max refinements reached (${MAX_REFINEMENTS}/${MAX_REFINEMENTS}). ` +
          "Please refine your original query for a fresh analysis.",
      ),
    );
    return section;
  }

  section.append(
    el("h3", `Refine Thesis (refinement ${job.refinement_count}/${MAX_REFINEMENTS})`),
  );

  const options = el("div");
  options.className = "feedback-options";
  const boxes: HTMLInputElement[] = [];
  for (const opt of feedbackOptions) {
    const label = el("label");
    const box = el("input");
    box.type = "checkbox";
    box.value = opt;
    label.append(box, document.createTextNode(` ${opt}`));
    options.append(label);
    boxes.push(box);
  }
  if (boxes.length === 0) {
    options.append(el("small", "Feedback options unavailable."));
  }
  section.append(options);

  const button = el("button", "Refine Thesis");
  button.disabled = true;
  const syncButton = () => {
    button.disabled = !boxes.some((b) => b.checked);
  };
  for (const b of boxes) b.addEventListener("change", syncButton);

  button.addEventListener("click", () => {
    const selected = boxes.filter((b) => b.checked).map((b) => b.value);
    if (selected.length === 0) return;
    button.disabled = true; // prevent double-submit; re-render replaces it
    onRefine(job.job_id, selected);
  });

  section.append(button);
  section.append(
    el("small", `Refinements left: ${MAX_REFINEMENTS - job.refinement_count}`),
  );
  return section;
}

// --- Version history (end-anchored feedback pairing) ---

function renderHistory(
  history: ThesisResponse[],
  feedbackHistory: string[][],
): HTMLElement | null {
  if (history.length === 0) return null;
  const details = el("details");
  details.append(el("summary", `Previous versions (${history.length})`));
  history.forEach((prev, i) => {
    const item = el("div");
    item.append(el("p", `Version ${i + 1}`));
    item.append(
      el("p", `Score: ${prev.opportunity_score}/5 | Recommendation: ${prev.recommendation}`),
    );
    // Pair from the end so a longer feedback_history can't shift annotations.
    const j = i + (feedbackHistory.length - history.length);
    const feedback = j >= 0 && j < feedbackHistory.length ? feedbackHistory[j] : [];
    if (feedback && feedback.length > 0) {
      item.append(el("small", `Refined with: ${feedback.join(", ")}`));
    }
    details.append(item);
  });
  return details;
}

// --- Hallucination analysis (only when invalid tools were found) ---

function renderHallucination(raw: JobResponse["hallucination"]): HTMLElement | null {
  const h = raw as HallucinationAnalysis | null | undefined;
  if (!h || !h.invalid_tools || h.invalid_tools.length === 0) return null;
  const section = el("section");
  section.className = "hallucination";
  section.append(el("p", "Hallucinations Detected"));
  const details = el("details");
  details.open = true;
  details.append(el("summary", "Tool Call Analysis"));
  if (h.summary) details.append(el("pre", h.summary));
  details.append(el("p", `Invalid tools (do not exist): ${h.invalid_tools.join(", ")}`));
  section.append(details);
  return section;
}

// --- Execution trace ---

function renderExecutionTrace(log: unknown[]): HTMLElement | null {
  if (log.length === 0) return null;
  const details = el("details");
  details.append(el("summary", "Execution Trace"));
  details.append(el("p", "Tools that actually executed during refinement:"));
  log.forEach((raw, idx) => {
    const event = raw as ExecutionEvent;
    const item = el("div");
    item.append(
      el("p", `${idx + 1}. ${event.tool_name ?? "unknown"} - ${event.status ?? "unknown"}`),
    );
    if (event.refinement_number) {
      item.append(el("small", `Refinement #${event.refinement_number}`));
    }
    if (event.reason) item.append(el("small", `Reason: ${event.reason}`));
    for (const change of event.changes ?? []) item.append(el("small", change));
    details.append(item);
  });
  return details;
}

// --- Resume picker (controller filters the list; view builds the widget) ---

export function renderResumePicker(
  jobs: ThesisSummaryResponse[],
  onResume: ResumeHandler,
): HTMLElement {
  const details = el("details");
  details.append(el("summary", `Resume a previous session (${jobs.length} available)`));

  const select = el("select");
  for (const j of jobs) {
    const created = (j.created_at ?? "").slice(0, 19);
    const option = el(
      "option",
      `${j.query} - round ${j.refinement_count}/3 (${j.refinement_status}) - ${created}`,
    );
    option.value = j.job_id;
    select.append(option);
  }

  const button = el("button", "Resume");
  button.addEventListener("click", () => {
    const jobId = select.value;
    if (!jobId) return;
    button.disabled = true;
    // The view owns its button state; the controller owns network + URL.
    void onResume(jobId).finally(() => {
      button.disabled = false;
    });
  });

  details.append(select, button);
  return details;
}

// --- Full-job composition ---

export function renderJob(
  container: HTMLElement,
  job: JobResponse,
  feedbackOptions: string[],
  onRefine: RefineHandler,
  onApprove: ApproveHandler,
): void {
  container.replaceChildren();

  const sources = renderSources(job.sources);
  if (sources) container.append(sources);

  const thesis = job.thesis;
  if (!thesis) {
    container.append(el("p", "No thesis was returned."));
    return;
  }

  if (thesis.raw_output) {
    container.append(el("h3", "Raw Summary"));
    container.append(el("pre", thesis.raw_output));
  }

  if (thesis.key_themes.length === 0) {
    container.append(
      el("p", "Could not parse structured output. See raw output above."),
    );
    return;
  }

  container.append(el("p", "Structured thesis generated successfully"));
  container.append(renderThesis(thesis));

  const related = renderRelated(job.related_theses);
  if (related) container.append(related);

  // Approval first (matches app.py). When approved, no refinement controls.
  if (job.approved_at) {
    container.append(
      el("p", "This thesis has been approved. No further refinements needed."),
    );
  } else {
    container.append(renderApproval(job, onApprove));
    container.append(renderRefinementPanel(job, feedbackOptions, onRefine));
  }

  const history = renderHistory(job.thesis_history, job.feedback_history);
  if (history) container.append(history);

  const hallucination = renderHallucination(job.hallucination);
  if (hallucination) container.append(hallucination);

  const trace = renderExecutionTrace(job.execution_log);
  if (trace) container.append(trace);
}