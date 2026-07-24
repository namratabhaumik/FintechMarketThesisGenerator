import type { ResumeHandler, StatusMessage, ThesisSummaryResponse } from "../types";
import { ResumePicker } from "./ResumePicker";

// Hero + query input + resume picker + the app-wide status line. Presentational:
// the query string and every action are owned by the App controller.
export function QueryPanel({
  query,
  onQueryChange,
  onGenerate,
  generating,
  resumeJobs,
  resumeError,
  onResume,
  status,
}: {
  query: string;
  onQueryChange: (query: string) => void;
  onGenerate: () => void;
  generating: boolean;
  resumeJobs: ThesisSummaryResponse[];
  resumeError: boolean;
  onResume: ResumeHandler;
  status: StatusMessage;
}) {
  const statusClass =
    status.isError && status.text
      ? "text-xs font-mono mt-3 text-error border border-error/30 bg-error/10 rounded-field px-3 py-2"
      : "text-xs text-base-content/60 font-mono mt-3";

  return (
    <section className="print:hidden max-w-5xl mx-auto px-6 pt-12 pb-8">
      <div className="mb-8">
        <p className="text-xs font-mono text-primary uppercase tracking-widest mb-2">
          AI Research Assistant
        </p>
        <h1 className="text-2xl font-semibold leading-snug">
          What fintech market do you want to analyze?
        </h1>
        <p className="text-sm text-base-content/60 mt-1 leading-relaxed">
          Enter a topic or question - we&apos;ll research recent articles and return a scored
          investment thesis.
        </p>
      </div>

      <div className="flex gap-3">
        <input
          type="text"
          className="w-full bg-base-200 border border-base-300 rounded-field px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/60"
          placeholder="e.g., What's the outlook for cross-border payments infrastructure companies?"
          aria-label="Market topic or question"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onGenerate();
          }}
        />
        <button
          type="button"
          className="btn btn-primary disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-primary! disabled:text-primary-content! disabled:border-primary! disabled:opacity-40!"
          disabled={query.trim().length === 0 || generating}
          onClick={onGenerate}
        >
          Generate Thesis
        </button>
      </div>

      <div className="mt-4">
        {resumeError ? (
          <small className="text-base-content/60">Could not load previous sessions.</small>
        ) : (
          resumeJobs.length > 0 && <ResumePicker jobs={resumeJobs} onResume={onResume} />
        )}
      </div>

      <p className={statusClass}>{status.text}</p>
    </section>
  );
}
