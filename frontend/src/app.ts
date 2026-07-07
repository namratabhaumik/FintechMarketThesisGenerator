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
import { renderJob, renderResumePicker } from "./render";
import { RefinementStatus } from "./types";
import type { JobResponse, ThesisSummaryResponse } from "./types";

export class FinThesisApp {
  private currentJob: JobResponse | null = null;
  private feedbackOptions: string[] = [];

  private readonly pickerContainer: HTMLElement;
  private readonly input: HTMLInputElement;
  private readonly generateButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  private readonly results: HTMLElement;

  private constructor(root: HTMLElement) {
    root.append(el("h1", "FinThesis: Fintech Market Research Assistant"));

    // Resume picker lives above the query input (mirrors app.py's ordering).
    this.pickerContainer = el("div");

    this.input = el("input");
    this.input.type = "text";
    this.input.placeholder = "e.g., Future of Digital Lending in Asia";
    this.input.setAttribute("aria-label", "Market topic or question");

    this.generateButton = el("button", "Generate Thesis");
    this.status = el("p");
    this.status.className = "status";
    this.results = el("section");

    root.append(
      this.pickerContainer,
      this.input,
      this.generateButton,
      this.status,
      this.results,
    );

    this.generateButton.addEventListener("click", () => void this.generate());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") void this.generate();
    });
  }

  /** Build the app in the given root element and start it. */
  static mount(selector: string): void {
    const root = document.querySelector<HTMLElement>(selector);
    if (!root) return;
    new FinThesisApp(root).init();
  }

  private init(): void {
    // Always offer the resume picker (if any resumable runs exist), and
    // additionally restore a specific run when the URL carries ?job_id.
    void this.showResumePicker();
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
}
