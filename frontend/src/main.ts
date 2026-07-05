// Frontend entry point.


import {
  ApiError,
  createRefinement,
  createThesis,
  getFeedbackOptions,
} from "./api";
import type { components } from "./types.gen";

type JobResponse = components["schemas"]["JobResponse"];
type ThesisResponse = components["schemas"]["ThesisResponse"];
type SourceResponse = components["schemas"]["SourceResponse"];

// Loosely-typed backend payloads (execution_log is unknown[], hallucination is
// an open dict); narrow them to just the fields we render.
interface ExecutionEvent {
  tool_name?: string;
  status?: string;
  refinement_number?: number;
  reason?: string;
  changes?: string[];
}
interface HallucinationAnalysis {
  invalid_tools?: string[];
  summary?: string;
}

type RefineHandler = (jobId: string, feedback: string[]) => void;

const MAX_REFINEMENTS = 3;

// Cached once; the option list is fixed server-side (see /api/feedback-options).
let feedbackOptions: string[] = [];
async function ensureFeedbackOptions(): Promise<void> {
  if (feedbackOptions.length === 0) {
    feedbackOptions = await getFeedbackOptions();
  }
}

// --- Small DOM helpers ---

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function bulletList(items: string[], empty: string): HTMLElement {
  if (items.length === 0) return el("p", empty);
  const ul = el("ul");
  for (const item of items) ul.append(el("li", item));
  return ul;
}

// --- Source articles (mirrors app.py's date-range label + expander) ---

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** "Jul 03, 2026" from an ISO date/datetime string. */
function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const day = String(d.getDate()).padStart(2, "0");
  return `${MONTHS[d.getMonth()]} ${day}, ${d.getFullYear()}`;
}

/** "Source Articles (Jun 01, 2026 - Jul 03, 2026)", collapsing a single date. */
function sourcesLabel(sources: SourceResponse[]): string {
  const times = sources
    .map((s) => s.published_at)
    .filter((p): p is string => Boolean(p))
    .map((p) => new Date(p).getTime())
    .filter((t) => !Number.isNaN(t));
  if (times.length === 0) return "Source Articles";
  const lo = new Date(Math.min(...times));
  const hi = new Date(Math.max(...times));
  const loStr = fmtDate(lo.toISOString());
  const hiStr = fmtDate(hi.toISOString());
  const span = loStr === hiStr ? loStr : `${loStr} - ${hiStr}`;
  return `Source Articles (${span})`;
}

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

// --- Structured thesis (mirrors display_structured_thesis in app.py) ---

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

// --- Refinement panel (mirrors show_refinement_controls / escalation) ---

function renderRefinementPanel(job: JobResponse, onRefine: RefineHandler): HTMLElement {
  const section = el("section");
  section.className = "refinement";

  if (job.refinement_status === "escalated") {
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
  section.append(el("small", `Refinements left: ${MAX_REFINEMENTS - job.refinement_count}`));
  return section;
}

// --- Version history (mirrors show_history's end-anchored feedback pairing) ---

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

// --- Execution trace (mirrors show_execution_trace, icons dropped) ---

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

// --- Full-job render (thesis + refinement affordances) ---

function renderJob(container: HTMLElement, job: JobResponse, onRefine: RefineHandler): void {
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
  container.append(renderRefinementPanel(job, onRefine));

  const history = renderHistory(job.thesis_history, job.feedback_history);
  if (history) container.append(history);

  const hallucination = renderHallucination(job.hallucination);
  if (hallucination) container.append(hallucination);

  const trace = renderExecutionTrace(job.execution_log);
  if (trace) container.append(trace);
}

// --- Page shell + wiring ---

function main(): void {
  const app = document.querySelector<HTMLElement>("#app");
  if (!app) return;

  app.append(el("h1", "FinThesis: Fintech Market Research Assistant"));

  const input = el("input");
  input.type = "text";
  input.placeholder = "e.g., Future of Digital Lending in Asia";
  input.setAttribute("aria-label", "Market topic or question");

  const button = el("button", "Generate Thesis");
  const status = el("p");
  status.className = "status";
  const results = el("section");

  app.append(input, button, status, results);

  // The current job in view; refinement POSTs against it and re-renders. Also
  // used to restore the panel after a failed refinement.
  let currentJob: JobResponse | null = null;

  function handleRefine(jobId: string, feedback: string[]): void {
    status.textContent = "Refining thesis based on your feedback...";
    void (async () => {
      try {
        const job = await createRefinement(jobId, feedback);
        currentJob = job;
        status.textContent = "";
        renderJob(results, job, handleRefine);
      } catch (err) {
        if (err instanceof ApiError) {
          status.textContent = `Refinement failed: ${err.message}`;
        } else {
          console.error("Refinement request failed", err);
          status.textContent = "An unexpected error occurred during refinement.";
        }
        // Restore an interactive panel (the clicked button was disabled).
        if (currentJob) renderJob(results, currentJob, handleRefine);
      }
    })();
  }

  async function generate(): Promise<void> {
    const query = input.value.trim();
    if (!query) {
      status.textContent = "Please enter a non-empty query.";
      return;
    }
    button.disabled = true;
    results.replaceChildren();
    status.textContent = "Retrieving context and generating thesis...";
    try {
      const job = await createThesis(query);
      currentJob = job;
      // Feedback options power the refinement panel; ensure them before render.
      await ensureFeedbackOptions();
      status.textContent = "";
      try {
        renderJob(results, job, handleRefine);
      } catch (renderErr) {
        console.error("Failed to render thesis", renderErr);
        results.replaceChildren();
        status.textContent = "Received a thesis but could not display it (unexpected shape).";
      }
    } catch (err) {
      if (err instanceof ApiError && err.code === "no_relevant_documents") {
        status.textContent = "No relevant documents found in the corpus for this query.";
      } else if (err instanceof ApiError) {
        status.textContent = `Error: ${err.message}`;
      } else {
        console.error("Thesis request failed", err);
        status.textContent = "An unexpected error occurred. Is the API running?";
      }
    } finally {
      button.disabled = false;
    }
  }

  button.addEventListener("click", () => void generate());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") void generate();
  });
}

// Guard bootstrap so a startup error surfaces in the console instead of a
// silently blank page.
try {
  main();
} catch (err) {
  console.error("Failed to start the FinThesis UI", err);
}