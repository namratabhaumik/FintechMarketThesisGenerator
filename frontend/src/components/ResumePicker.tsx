import { useState } from "react";
import { fmtDate } from "../format";
import type { ResumeHandler, ThesisSummaryResponse } from "../types";

// Collapsible picker of resumable (mid-refinement) runs. The controller decides
// which runs are resumable and passes them in; this widget owns only its
// selection + in-flight button state.
export function ResumePicker({
  jobs,
  onResume,
}: {
  jobs: ThesisSummaryResponse[];
  onResume: ResumeHandler;
}) {
  const [selected, setSelected] = useState(jobs[0]?.job_id ?? "");
  const [busy, setBusy] = useState(false);

  const resume = () => {
    if (!selected) return;
    setBusy(true);
    void onResume(selected).finally(() => setBusy(false));
  };

  return (
    <details className="group">
      <summary className="flex items-center gap-1.5 text-xs text-base-content/60 hover:text-base-content transition-colors cursor-pointer list-none [&::-webkit-details-marker]:hidden">
        <span className="transition-transform group-open:rotate-90">
          <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path
              d="M4.5 2.5L8 6l-3.5 3.5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        {`Resume a previous refinement (${jobs.length} available)`}
      </summary>
      <div className="mt-3 flex gap-3 items-center">
        <select
          className="select select-sm bg-base-200 border-base-300 flex-1 text-xs"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          // No <form> wraps this picker, so Enter needs an explicit handler.
          onKeyDown={(e) => {
            if (e.key === "Enter") resume();
          }}
        >
          {jobs.map((j) => (
            <option key={j.job_id} value={j.job_id}>
              {`${j.query} - round ${j.refinement_count}/3 - ${j.created_at ? fmtDate(j.created_at) : ""}`}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-sm bg-base-300 hover:bg-base-300/70 border-none text-base-content"
          disabled={busy}
          onClick={resume}
        >
          Resume
        </button>
      </div>
    </details>
  );
}
