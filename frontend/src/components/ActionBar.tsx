import { useState } from "react";
import { RefinementStatus } from "../types";
import type { ApproveHandler, JobResponse, RefineHandler } from "../types";
import { VersionHistory } from "./VersionHistory";

// Display cap for the refinement counters. Must match MAX_REFINEMENTS in the
// backend's refinement_graph.py; the server enforces the real limit and signals
// exhaustion via RefinementStatus.Escalated, which the UI honors regardless.
const MAX_REFINEMENTS = 3;

// Action bar: Approve + Refine live together. Once escalated (max refinements
// reached), only Approve remains. A local `submitting` flag guards against
// double-submit until the parent re-renders with the new job.
export function ActionBar({
  job,
  feedbackOptions,
  onRefine,
  onApprove,
}: {
  job: JobResponse;
  feedbackOptions: string[];
  onRefine: RefineHandler;
  onApprove: ApproveHandler;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const escalated = job.refinement_status === RefinementStatus.Escalated;

  const approve = () => {
    setSubmitting(true);
    onApprove(job.job_id);
  };
  const refine = () => {
    if (selected.length === 0) return;
    setSubmitting(true);
    onRefine(job.job_id, selected);
  };
  const toggle = (opt: string, checked: boolean) => {
    setSelected((prev) => (checked ? [...prev, opt] : prev.filter((x) => x !== opt)));
  };

  const approveButton = (
    <button
      type="button"
      className="btn btn-primary btn-sm disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-primary! disabled:text-primary-content! disabled:border-primary! disabled:opacity-40!"
      disabled={submitting}
      onClick={approve}
    >
      Approve
    </button>
  );

  return (
    <section className="print:hidden bg-base-200 border border-base-300 rounded-box px-6 py-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">
            Refine Thesis{" "}
            <span className="font-normal text-xs text-base-content/60">
              {`(refinement ${job.refinement_count}/${MAX_REFINEMENTS})`}
            </span>
          </p>
          <p className="text-xs text-base-content/60 mt-0.5">
            Select reasons to guide the next iteration, or approve to finalize.
          </p>
        </div>
        <span className="text-xs font-mono text-base-content/60">
          {`${MAX_REFINEMENTS - job.refinement_count} left`}
        </span>
      </div>

      {escalated ? (
        <>
          <p className="bg-accent/10 border border-accent/30 rounded-field px-4 py-3 text-xs text-accent">
            {`Max refinements reached (${MAX_REFINEMENTS}/${MAX_REFINEMENTS}). ` +
              "Please refine your original query for a fresh analysis."}
          </p>
          {approveButton}
        </>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {feedbackOptions.length === 0 ? (
              <small className="text-base-content/60">Feedback options unavailable.</small>
            ) : (
              feedbackOptions.map((opt) => (
                <label
                  key={opt}
                  className="flex items-center gap-2.5 px-3 py-2.5 rounded-field border border-base-300 bg-base-300/30 hover:bg-base-300 cursor-pointer text-xs select-none"
                >
                  <input
                    type="checkbox"
                    className="checkbox checkbox-primary checkbox-xs flex-shrink-0"
                    value={opt}
                    checked={selected.includes(opt)}
                    onChange={(e) => toggle(opt, e.target.checked)}
                  />
                  {` ${opt}`}
                </label>
              ))
            )}
          </div>
          <div className="flex gap-3">
            <button
              type="button"
              className="btn btn-outline btn-sm disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-transparent! disabled:text-base-content! disabled:border-base-content! disabled:opacity-40!"
              disabled={selected.length === 0 || submitting}
              onClick={refine}
            >
              Refine Thesis
            </button>
            {approveButton}
          </div>
        </>
      )}

      <VersionHistory history={job.thesis_history} feedbackHistory={job.feedback_history} />
    </section>
  );
}
