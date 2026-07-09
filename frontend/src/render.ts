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

// --- Shared bits ---

// Collapsible <details>/<summary> styled as a daisyUI collapse. The body is
// wrapped in .collapse-content, matching daisyUI's expected structure.
// `boxed` controls whether it gets its own card background/border (top-level
// sections like Sources/Key Themes) or sits flush against a parent card,
// separated by a top border only (e.g. "Previous versions" nested in the
// action bar).
function collapsible(summaryText: string, body: HTMLElement, open = false, boxed = true): HTMLElement {
  const details = el(
    "details",
    undefined,
    boxed
      ? "collapse collapse-arrow bg-base-200 border border-base-300 rounded-box"
      : "collapse collapse-arrow border-t border-base-300 rounded-none",
  );
  details.open = open;
  details.append(
    el(
      "summary",
      summaryText,
      boxed
        ? "collapse-title text-xs font-semibold uppercase tracking-widest text-base-content/60 hover:text-base-content transition-colors"
        : "collapse-title text-xs text-base-content/60 min-h-0 py-0",
    ),
  );
  const content = el("div", undefined, "collapse-content text-sm");
  content.append(body);
  details.append(content);
  return details;
}

function recommendationBadgeClass(rec: string): string {
  const base = "inline-flex items-center px-2.5 py-1 rounded-md text-xs font-semibold border";
  if (rec === "Pursue") return `${base} bg-primary/15 text-primary border-primary/30`;
  if (rec === "Skip") return `${base} bg-error/15 text-error border-error/30`;
  return `${base} bg-accent/15 text-accent border-accent/30`; // Investigate and any other value
}

// --- Source articles ---

function renderSourceItem(s: SourceResponse): HTMLElement {
  const li = el("li");
  if (s.url) {
    const a = el(
      "a",
      s.title,
      "text-xs text-primary hover:text-primary/80 underline underline-offset-2 decoration-primary/30 leading-relaxed transition-colors",
    );
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
  const ul = el("ul", undefined, "space-y-1.5");
  for (const s of sources) {
    // Skip a single malformed source rather than failing the whole list,
    // mirroring _sources_from_docs on the backend.
    try {
      ul.append(renderSourceItem(s));
    } catch (err) {
      console.warn("Skipping unrenderable source", s, err);
    }
  }
  return collapsible(sourcesLabel(sources), ul, true);
}

// --- Structured thesis ---

// Score / confidence / recommendation strip. Rendered first (right after the
// job-id line), matching the design's placement ahead of Sources/Raw Summary.
function renderMetricsStrip(thesis: ThesisResponse): HTMLElement {
  const metrics = el(
    "div",
    undefined,
    "border-b border-base-300 pb-4 flex flex-wrap items-center gap-6",
  );

  const scoreBox = el("div", undefined, "flex-1 min-w-0");
  scoreBox.append(el("p", "Investment Score", "text-xs text-base-content/60 font-mono uppercase tracking-wider mb-1"));
  const scoreValue = el("div", undefined, "flex items-end gap-2");
  scoreValue.append(
    el("span", `${thesis.opportunity_score}`, "text-3xl font-bold font-mono leading-none"),
    el("span", "/5", "text-sm text-base-content/60 mb-0.5 font-mono"),
  );
  scoreBox.append(scoreValue);
  metrics.append(scoreBox);

  const confidenceBox = el("div", undefined, "flex-1 min-w-0");
  confidenceBox.append(el("p", "Confidence", "text-xs text-base-content/60 font-mono uppercase tracking-wider mb-1"));
  const pct = Math.trunc(thesis.confidence_level * 100);
  const confidenceValue = el("div", undefined, "flex items-end gap-2 mb-1.5");
  confidenceValue.append(
    el("span", `${pct}`, "text-3xl font-bold font-mono leading-none"),
    el("span", "%", "text-sm text-base-content/60 mb-0.5 font-mono"),
  );
  confidenceBox.append(confidenceValue);
  const barTrack = el("div", undefined, "h-1.5 w-full bg-base-300 rounded-full overflow-hidden");
  const barFill = el("div", undefined, "h-full rounded-full bg-accent");
  barFill.style.width = `${pct}%`;
  barTrack.setAttribute("role", "progressbar");
  barTrack.setAttribute("aria-valuenow", String(pct));
  barTrack.setAttribute("aria-valuemax", "100");
  barTrack.append(barFill);
  confidenceBox.append(barTrack);
  if (thesis.confidence_as_of) {
    confidenceBox.append(
      el(
        "p",
        `trends as of ${fmtDate(thesis.confidence_as_of)}`,
        "text-[10px] text-base-content/60 font-mono mt-1",
      ),
    );
  }
  metrics.append(confidenceBox);

  const recBox = el("div", undefined, "flex flex-col items-start gap-2");
  recBox.append(el("p", "Recommendation", "text-xs text-base-content/60 font-mono uppercase tracking-wider"));
  recBox.append(el("span", thesis.recommendation, recommendationBadgeClass(thesis.recommendation)));
  metrics.append(recBox);

  return metrics;
}

// Key Themes / Risks / Investment Signals. Rendered after Sources/Raw Summary.
function renderThesisDetails(thesis: ThesisResponse): HTMLElement {
  const section = el("section", undefined, "space-y-4");

  section.append(
    collapsible("Key Themes", bulletList(thesis.key_themes, "No themes found.", "bg-primary"), true),
  );

  const risksBody = el("div");
  risksBody.append(bulletList(thesis.risks, "No risks found.", "bg-error"));
  if (thesis.key_risk_factors.length > 0) {
    risksBody.append(
      el(
        "p",
        `Key risk factors: ${thesis.key_risk_factors.join(", ")}`,
        "text-xs text-base-content/60 font-mono mt-3",
      ),
    );
  }
  section.append(collapsible("Risks", risksBody, true));

  section.append(
    collapsible(
      "Investment Signals",
      bulletList(thesis.investment_signals, "No signals found.", "bg-accent"),
      true,
    ),
  );

  return section;
}

// --- Related past theses (episodic recall) ---

function renderRelated(related: RelatedThesisResponse[]): HTMLElement | null {
  if (related.length === 0) return null;
  const wrap = el("div", undefined, "space-y-2");
  for (const r of related) {
    const item = el(
      "div",
      undefined,
      "flex items-center justify-between py-2.5 px-3 rounded-field bg-base-300/50 hover:bg-base-300",
    );
    const left = el("div", undefined, "flex flex-col gap-0.5 min-w-0");
    const link = el("a", r.query, "text-xs text-primary hover:text-primary/80 font-medium truncate");
    link.href = `?job_id=${encodeURIComponent(r.job_id)}`;
    left.append(link);

    const date = r.created_at ? fmtDate(r.created_at) : "";
    const parts = [`score ${r.score}/5`, date].filter(Boolean);
    let meta = parts.join(" · ");
    if (r.approved) meta += " · approved";
    left.append(el("span", meta, "text-[10px] text-base-content/60 font-mono"));
    item.append(left);

    const right = el("div", undefined, "flex items-center gap-3 flex-shrink-0 ml-4");
    right.append(el("span", `sim ${r.similarity}`, "text-[10px] text-base-content/60 font-mono"));
    right.append(el("span", r.recommendation, recommendationBadgeClass(r.recommendation)));
    item.append(right);

    wrap.append(item);
  }
  return collapsible(`Related past theses (${related.length})`, wrap);
}

// --- Action bar: Approve + Refine live together ---

function renderActionBar(
  job: JobResponse,
  feedbackOptions: string[],
  onRefine: RefineHandler,
  onApprove: ApproveHandler,
): HTMLElement {
  const section = el(
    "section",
    undefined,
    "bg-base-200 border border-base-300 rounded-box px-6 py-5 space-y-4",
  );

  const escalated = job.refinement_status === RefinementStatus.Escalated;

  const headerRow = el("div", undefined, "flex items-center justify-between");
  const headerText = el("div");
  const titleLine = el("p", undefined, "text-sm font-semibold");
  titleLine.append(
    document.createTextNode("Refine Thesis "),
    el(
      "span",
      `(refinement ${job.refinement_count}/${MAX_REFINEMENTS})`,
      "font-normal text-xs text-base-content/60",
    ),
  );
  headerText.append(
    titleLine,
    el(
      "p",
      "Select reasons to guide the next iteration, or approve to finalize.",
      "text-xs text-base-content/60 mt-0.5",
    ),
  );
  headerRow.append(
    headerText,
    el(
      "span",
      `${MAX_REFINEMENTS - job.refinement_count} left`,
      "text-xs font-mono text-base-content/60",
    ),
  );
  section.append(headerRow);

  const approveButton = el(
    "button",
    "Approve",
    "btn btn-primary btn-sm disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-primary! disabled:text-primary-content! disabled:border-primary! disabled:opacity-40!",
  );
  approveButton.addEventListener("click", () => {
    approveButton.disabled = true; // prevent double-submit; re-render replaces it
    onApprove(job.job_id);
  });

  if (escalated) {
    section.append(
      el(
        "p",
        `Max refinements reached (${MAX_REFINEMENTS}/${MAX_REFINEMENTS}). ` +
          "Please refine your original query for a fresh analysis.",
        "bg-accent/10 border border-accent/30 rounded-field px-4 py-3 text-xs text-accent",
      ),
    );
    section.append(approveButton);
  } else {
    const options = el("div", undefined, "grid grid-cols-1 sm:grid-cols-2 gap-2");
    const boxes: HTMLInputElement[] = [];
    for (const opt of feedbackOptions) {
      const label = el(
        "label",
        undefined,
        "flex items-center gap-2.5 px-3 py-2.5 rounded-field border border-base-300 bg-base-300/30 hover:bg-base-300 cursor-pointer text-xs select-none",
      );
      const box = el("input", undefined, "checkbox checkbox-primary checkbox-xs flex-shrink-0");
      box.type = "checkbox";
      box.value = opt;
      label.append(box, document.createTextNode(` ${opt}`));
      options.append(label);
      boxes.push(box);
    }
    if (boxes.length === 0) {
      options.append(el("small", "Feedback options unavailable.", "text-base-content/60"));
    }
    section.append(options);

    const refineButton = el(
      "button",
      "Refine Thesis",
      "btn btn-outline btn-sm disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-transparent! disabled:text-base-content! disabled:border-base-content! disabled:opacity-40!",
    );
    refineButton.disabled = true;
    const syncButton = () => {
      refineButton.disabled = !boxes.some((b) => b.checked);
    };
    for (const b of boxes) b.addEventListener("change", syncButton);

    refineButton.addEventListener("click", () => {
      const selected = boxes.filter((b) => b.checked).map((b) => b.value);
      if (selected.length === 0) return;
      refineButton.disabled = true; // prevent double-submit; re-render replaces it
      onRefine(job.job_id, selected);
    });

    const actions = el("div", undefined, "flex gap-3");
    actions.append(refineButton, approveButton);
    section.append(actions);
  }

  const history = renderHistory(job.thesis_history, job.feedback_history);
  if (history) section.append(history);

  return section;
}

// --- Version history (end-anchored feedback pairing) ---
// Nested inside the action bar (not boxed) with a top border as separator,
// matching the design's placement.

function renderHistory(
  history: ThesisResponse[],
  feedbackHistory: string[][],
): HTMLElement | null {
  if (history.length === 0) return null;
  const wrap = el("div", undefined, "space-y-2 mt-3");
  history.forEach((prev, i) => {
    const item = el("div", undefined, "text-xs px-3 py-2.5 bg-base-300/30 rounded-field");
    item.append(el("p", `Version ${i + 1}`, "font-medium mb-0.5"));
    item.append(
      el(
        "p",
        `Score: ${prev.opportunity_score}/5 · Recommendation: ${prev.recommendation}`,
        "text-base-content/60",
      ),
    );
    // Pair from the end so a longer feedback_history can't shift annotations.
    const j = i + (feedbackHistory.length - history.length);
    const feedback = j >= 0 && j < feedbackHistory.length ? feedbackHistory[j] : [];
    if (feedback && feedback.length > 0) {
      item.append(
        el("p", `Refined with: ${feedback.join(", ")}`, "mt-0.5 text-[10px] text-base-content/60"),
      );
    }
    wrap.append(item);
  });
  return collapsible(`Previous versions (${history.length})`, wrap, false, false);
}

// --- Hallucination analysis (only when invalid tools were found) ---

function renderHallucination(raw: JobResponse["hallucination"]): HTMLElement | null {
  const h = raw as HallucinationAnalysis | null | undefined;
  if (!h || !h.invalid_tools || h.invalid_tools.length === 0) return null;
  const section = el(
    "section",
    undefined,
    "bg-error/10 border border-error/30 rounded-box px-6 py-4",
  );
  section.append(el("p", "Hallucinations Detected", "text-sm font-semibold text-error mb-2"));
  const body = el("div", undefined, "space-y-2 text-xs");
  if (h.summary) body.append(el("pre", h.summary, "whitespace-pre-wrap"));
  body.append(el("p", `Invalid tools (do not exist): ${h.invalid_tools.join(", ")}`));
  section.append(collapsible("Tool Call Analysis", body, true));
  return section;
}

// --- Execution trace ---

function renderExecutionTrace(log: unknown[]): HTMLElement | null {
  if (log.length === 0) return null;
  const body = el("div", undefined, "space-y-2");
  body.append(
    el("p", "Tools that actually executed during refinement:", "text-xs text-base-content/60"),
  );
  const list = el("ol", undefined, "space-y-1.5");
  log.forEach((raw, idx) => {
    const event = raw as ExecutionEvent;
    const item = el("li", undefined, "text-xs font-mono text-base-content/60");
    item.append(
      el("span", `${idx + 1}. `, "text-primary"),
      document.createTextNode(`${event.tool_name ?? "unknown"} — ${event.status ?? "unknown"}`),
    );
    list.append(item);
    if (event.refinement_number) {
      list.append(
        el("li", `Refinement #${event.refinement_number}`, "text-[10px] font-mono text-base-content/60"),
      );
    }
    if (event.reason) {
      list.append(el("li", `Reason: ${event.reason}`, "text-[10px] font-mono text-base-content/60"));
    }
    for (const change of event.changes ?? []) {
      list.append(el("li", change, "text-[10px] font-mono text-base-content/60"));
    }
  });
  body.append(list);
  return collapsible("Execution Trace", body);
}

// --- Resume picker (controller filters the list; view builds the widget) ---

export function renderResumePicker(
  jobs: ThesisSummaryResponse[],
  onResume: ResumeHandler,
): HTMLElement {
  const details = el("details", undefined, "group");

  const summary = el(
    "summary",
    undefined,
    "flex items-center gap-1.5 text-xs text-base-content/60 hover:text-base-content transition-colors cursor-pointer list-none [&::-webkit-details-marker]:hidden",
  );
  // Static, hardcoded chevron markup (not user/LLM data) - safe as innerHTML.
  const chevron = el("span", undefined, "transition-transform group-open:rotate-90");
  chevron.innerHTML =
    '<svg class="w-3 h-3" viewBox="0 0 12 12" fill="none" aria-hidden="true">' +
    '<path d="M4.5 2.5L8 6l-3.5 3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" /></svg>';
  summary.append(
    chevron,
    document.createTextNode(`Resume a previous refinement (${jobs.length} available)`),
  );
  details.append(summary);

  const row = el("div", undefined, "mt-3 flex gap-3 items-center");

  const select = el(
    "select",
    undefined,
    "select select-sm bg-base-200 border-base-300 flex-1 text-xs",
  );
  for (const j of jobs) {
    const created = j.created_at ? fmtDate(j.created_at) : "";
    const option = el(
      "option",
      `${j.query} - round ${j.refinement_count}/3 - ${created}`,
    );
    option.value = j.job_id;
    select.append(option);
  }

  const button = el("button", "Resume", "btn btn-sm bg-base-300 hover:bg-base-300/70 border-none text-base-content");
  const resume = () => {
    const jobId = select.value;
    if (!jobId) return;
    button.disabled = true;
    // The view owns its button state; the controller owns network + URL.
    void onResume(jobId).finally(() => {
      button.disabled = false;
    });
  };
  button.addEventListener("click", resume);
  // No <form> wraps this picker, so Enter needs an explicit handler here -
  // native form-submit-on-Enter doesn't apply to a bare <select>.
  select.addEventListener("keydown", (e) => {
    if (e.key === "Enter") resume();
  });

  row.append(select, button);
  details.append(row);
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

  const timestamp = job.created_at ? ` · ${fmtDate(job.created_at)}` : "";
  container.append(
    el(
      "p",
      `Thesis generated · job_id: ${job.job_id}${timestamp}`,
      "text-xs text-base-content/60 font-mono",
    ),
  );

  const card = el("div", undefined, "bg-base-200 border border-base-300 rounded-box px-6 py-5 space-y-4");
  container.append(card);

  const thesis = job.thesis;
  if (!thesis) {
    card.append(el("p", "No thesis was returned."));
    return;
  }

  card.append(renderMetricsStrip(thesis));

  const sources = renderSources(job.sources);
  if (sources) card.append(sources);

  if (thesis.raw_output) {
    const raw = el(
      "p",
      thesis.raw_output,
      "text-sm text-base-content/60 leading-relaxed whitespace-pre-wrap",
    );
    card.append(collapsible("Raw Summary", raw, true));
  }

  if (thesis.key_themes.length === 0) {
    card.append(
      el("p", "Could not parse structured output. See raw output above.", "text-sm text-base-content/60"),
    );
    return;
  }

  card.append(renderThesisDetails(thesis));

  const related = renderRelated(job.related_theses);
  if (related) card.append(related);

  // Approval first (matches app.py). When approved, no refinement controls.
  if (job.approved_at) {
    container.append(
      el(
        "p",
        "This thesis has been approved. No further refinements needed.",
        "flex items-center gap-1.5 text-xs text-primary font-semibold",
      ),
    );
  } else {
    container.append(renderActionBar(job, feedbackOptions, onRefine, onApprove));
  }

  const hallucination = renderHallucination(job.hallucination);
  if (hallucination) container.append(hallucination);

  const trace = renderExecutionTrace(job.execution_log);
  if (trace) container.append(trace);
}