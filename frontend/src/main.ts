// Frontend entry point.

import { ApiError, createThesis } from "./api";
import type { components } from "./types.gen";

type JobResponse = components["schemas"]["JobResponse"];
type ThesisResponse = components["schemas"]["ThesisResponse"];
type SourceResponse = components["schemas"]["SourceResponse"];

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

// --- Result orchestration ---

function renderResult(results: HTMLElement, job: JobResponse): void {
  results.replaceChildren();

  const sources = renderSources(job.sources);
  if (sources) results.append(sources);

  const thesis = job.thesis;
  if (!thesis) {
    results.append(el("p", "No thesis was returned."));
    return;
  }

  if (thesis.raw_output) {
    results.append(el("h3", "Raw Summary"));
    results.append(el("pre", thesis.raw_output));
  }

  if (thesis.key_themes.length > 0) {
    results.append(el("p", "Structured thesis generated successfully"));
    results.append(renderThesis(thesis));
  } else {
    results.append(
      el("p", "Could not parse structured output. See raw output above."),
    );
  }
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
      status.textContent = "";
      // Rendering failures are distinct from network failures: an unexpected
      // response shape shouldn't look like the API being down.
      try {
        renderResult(results, job);
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