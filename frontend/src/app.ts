// Controller: owns app state (current job, feedback options) and orchestrates
// network calls, rendering, and error reporting. The view (render.ts) is pure
// and receives callbacks from here; the controller never builds DOM beyond the
// page shell and delegating to the view.

import {
  ApiError,
  ErrorCode,
  approveThesis,
  createRefinement,
  createThesis,
  getFeedbackOptions,
  getThesis,
  listTheses,
} from "./api";
import { el } from "./dom";
import { renderJob, renderPastTheses, renderResumePicker } from "./render";
import { RefinementStatus } from "./types";
import type { JobResponse, ThesisSummaryResponse } from "./types";

/** Signed-in user info + sign-out handler, passed in by the auth gate (main.ts). */
export interface AuthInfo {
  email?: string | null;
  onSignOut: () => void;
}

export class FinThesisApp {
  private currentJob: JobResponse | null = null;
  private feedbackOptions: string[] = [];

  private readonly pickerContainer: HTMLElement;
  private readonly input: HTMLInputElement;
  private readonly generateButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  private readonly results: HTMLElement;
  private readonly pastTheses: HTMLElement;

  private constructor(root: HTMLElement, auth?: AuthInfo) {
    const header = el(
      "header",
      undefined,
      "border-b border-base-300 bg-base-100/80 backdrop-blur-sm sticky top-0 z-50",
    );
    const headerInner = el(
      "div",
      undefined,
      "max-w-5xl mx-auto px-6 h-14 flex items-center justify-between",
    );

    const brand = el("div", undefined, "flex items-center gap-3");
    const logo = el("div", undefined, "w-7 h-7 rounded bg-primary flex items-center justify-center");
    // Static, hardcoded icon markup (not user/LLM data) - safe as innerHTML.
    logo.innerHTML =
      '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<path d="M2 11L5.5 6.5L8 9L11 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="text-primary-content" /></svg>';
    brand.append(
      logo,
      el("span", "FinThesis", "font-semibold tracking-tight text-sm"),
      el(
        "span",
        "Fintech Market Research",
        "hidden sm:block text-xs text-base-content/60 border-l border-base-300 pl-3",
      ),
    );

    const systemStatus = el(
      "div",
      undefined,
      "flex items-center gap-2 text-xs text-base-content/60 font-mono",
    );
    systemStatus.append(
      el("span", undefined, "w-1.5 h-1.5 rounded-full bg-primary animate-pulse"),
      document.createTextNode("System active"),
    );

    headerInner.append(brand, auth ? this.buildUserMenu(auth) : systemStatus);
    header.append(headerInner);

    const main = el("section", undefined, "max-w-5xl mx-auto px-6 pt-12 pb-8");

    const hero = el("div", undefined, "mb-8");
    hero.append(
      el(
        "p",
        "AI Research Assistant",
        "text-xs font-mono text-primary uppercase tracking-widest mb-2",
      ),
      el(
        "h1",
        "What fintech market do you want to analyze?",
        "text-2xl font-semibold leading-snug",
      ),
      el(
        "p",
        "Enter a topic or question - we'll research recent articles and return a scored investment thesis.",
        "text-sm text-base-content/60 mt-1 leading-relaxed",
      ),
    );
    main.append(hero);

    this.pickerContainer = el("div", undefined, "mt-4");

    this.input = el(
      "input",
      undefined,
      "w-full bg-base-200 border border-base-300 rounded-field px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/60",
    );
    this.input.type = "text";
    this.input.placeholder = "e.g., Future of Digital Lending in Asia";
    this.input.setAttribute("aria-label", "Market topic or question");

    this.generateButton = el(
      "button",
      "Generate Thesis",
      "btn btn-primary disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-primary! disabled:text-primary-content! disabled:border-primary! disabled:opacity-40!",
    );
    this.generateButton.disabled = true; // empty query bar on first load
    this.status = el("p", undefined, "text-xs text-base-content/60 font-mono mt-3");
    this.results = el("section", undefined, "max-w-5xl mx-auto px-6 pb-16 space-y-4");

    const inputRow = el("div", undefined, "flex gap-3");
    inputRow.append(this.input, this.generateButton);
    main.append(inputRow, this.pickerContainer, this.status);

    this.pastTheses = el("section", undefined, "max-w-5xl mx-auto px-6 pb-16 -mt-8");

    root.replaceChildren(header, main, this.results, this.pastTheses);

    this.generateButton.addEventListener("click", () => void this.generate());
    this.input.addEventListener("input", () => this.syncGenerateButton());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") void this.generate();
    });
  }

  // Keeps the button disabled while the query bar is empty.
  private syncGenerateButton(): void {
    this.generateButton.disabled = this.input.value.trim().length === 0;
  }

  // Signed-in header: user email + sign-out, replacing the "System active" chip.
  private buildUserMenu(auth: AuthInfo): HTMLElement {
    const menu = el("div", undefined, "flex items-center gap-3 text-xs");
    if (auth.email) {
      menu.append(el("span", auth.email, "text-base-content/60 font-mono hidden sm:block"));
    }
    const signOut = el("button", "Sign out", "btn btn-ghost btn-xs");
    signOut.addEventListener("click", () => auth.onSignOut());
    menu.append(signOut);
    return menu;
  }

  /** Build the app in the given root element and start it. */
  static mount(selector: string, auth?: AuthInfo): void {
    const root = document.querySelector<HTMLElement>(selector);
    if (!root) return;
    new FinThesisApp(root, auth).init();
  }

  private init(): void {
    // Always offer the resume picker (if any resumable runs exist), and
    // additionally restore a specific run when the URL carries ?job_id.
    void this.showResumePicker();
    void this.showPastTheses();
    const jobId = new URLSearchParams(location.search).get("job_id");
    if (jobId) void this.restore(jobId);
  }

  // --- Helpers ---

  private setStatus(text: string): void {
    this.status.textContent = text;
  }

  /** Map an error to a status message; ApiError carries a server message. */
  private reportError(err: unknown, fallback: string, prefix: string): void {
    if (err instanceof ApiError) {
      this.setStatus(`${prefix}: ${err.message}`);
    } else {
      console.error(fallback, err);
      this.setStatus(fallback);
    }
  }

  // Load the fixed feedback options once. Non-fatal: a failure just leaves the
  // refine panel without options, so it never blocks showing a thesis.
  private async ensureFeedbackOptions(): Promise<void> {
    if (this.feedbackOptions.length > 0) return;
    try {
      this.feedbackOptions = await getFeedbackOptions();
    } catch (err) {
      console.error("Failed to load feedback options", err);
    }
  }

  // Guarded render: an unexpected response shape shows a message instead of
  // throwing (which, inside an async handler, would be an unhandled rejection).
  private render(job: JobResponse): void {
    try {
      renderJob(this.results, job, this.feedbackOptions, this.onRefine, this.onApprove);
    } catch (err) {
      console.error("Failed to render job", err);
      this.results.replaceChildren();
      this.setStatus("Could not display the thesis (unexpected response shape).");
    }
  }

  // --- Actions ---

  private async generate(): Promise<void> {
    const query = this.input.value.trim();
    if (!query) {
      this.setStatus("Please enter a non-empty query.");
      return;
    }
    this.generateButton.disabled = true;
    this.results.replaceChildren();
    this.setStatus("Retrieving context and generating thesis...");
    try {
      const job = await createThesis(query);
      this.currentJob = job;
      // A ?job_id URL makes the run restorable on refresh/new tab. The resume
      // picker is left in place so you can still switch sessions.
      history.replaceState(null, "", `?job_id=${encodeURIComponent(job.job_id)}`);
      await this.ensureFeedbackOptions();
      this.setStatus("");
      this.render(job);
    } catch (err) {
      if (err instanceof ApiError && err.code === ErrorCode.NoRelevantDocuments) {
        this.setStatus(
          "No relevant documents found for this query. Try a broader or different fintech topic.",
        );
      } else {
        this.reportError(err, "An unexpected error occurred. Is the API running?", "Error");
      }
    } finally {
      this.generateButton.disabled = false;
    }
  }

  private onRefine = (jobId: string, feedback: string[]): void => {
    this.setStatus("Refining thesis based on your feedback...");
    void (async () => {
      try {
        const job = await createRefinement(jobId, feedback);
        this.currentJob = job;
        this.setStatus("");
        this.render(job);
      } catch (err) {
        this.reportError(
          err,
          "An unexpected error occurred during refinement.",
          "Refinement failed",
        );
        // Restore the interactive panel (the clicked button was disabled).
        if (this.currentJob) this.render(this.currentJob);
      }
    })();
  };

  private onApprove = (jobId: string): void => {
    this.setStatus("Approving thesis...");
    void (async () => {
      try {
        const job = await approveThesis(jobId);
        this.currentJob = job;
        this.setStatus("");
        this.render(job);
      } catch (err) {
        this.reportError(
          err,
          "An unexpected error occurred during approval.",
          "Approval failed",
        );
        if (this.currentJob) this.render(this.currentJob);
      }
    })();
  };

  // Load a persisted job by id and render it. Returns whether it loaded, so the
  // resume picker can update the URL on success; the ?job_id path ignores it.
  private async restore(jobId: string): Promise<boolean> {
    this.setStatus("Loading saved thesis...");
    try {
      await this.ensureFeedbackOptions();
      const job = await getThesis(jobId);
      this.currentJob = job;
      // Sync the query bar to the loaded run
      this.input.value = job.query;
      this.syncGenerateButton();
      this.setStatus("");
      this.render(job);
      return true;
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        this.setStatus("That saved thesis was not found.");
      } else {
        this.reportError(err, "An unexpected error occurred loading the saved thesis.", "Error");
      }
      return false;
    }
  }

  private onResume = async (jobId: string): Promise<void> => {
    // Picker persists after a resume so you can switch sessions without reload.
    if (await this.restore(jobId)) {
      history.replaceState(null, "", `?job_id=${encodeURIComponent(jobId)}`);
    }
  };

  // Strict app.py parity on WHICH runs are resumable: only those mid-refinement
  // (refinement_status "refining"); never-refined ("N/A") and terminal
  // (escalated / approved) runs are excluded. The list is fetched once, so
  // same-tab state changes aren't reflected until reload.
  private async showResumePicker(): Promise<void> {
    let jobs: ThesisSummaryResponse[];
    try {
      jobs = await listTheses(20, RefinementStatus.Refining);
    } catch (err) {
      console.error("Failed to load resumable sessions", err);
      this.pickerContainer.replaceChildren(
        el("small", "Could not load previous sessions."),
      );
      return;
    }
    if (jobs.length === 0) return;
    this.pickerContainer.replaceChildren(renderResumePicker(jobs, this.onResume));
  }

  // Full research library: every past run regardless of refinement status.
  // Loaded once at startup (see init()).
  private async showPastTheses(): Promise<void> {
    let jobs: ThesisSummaryResponse[];
    try {
      jobs = await listTheses(20);
    } catch (err) {
      console.error("Failed to load past theses", err);
      this.pastTheses.replaceChildren();
      return;
    }
    const list = renderPastTheses(jobs);
    this.pastTheses.replaceChildren(...(list ? [list] : []));
  }
}
