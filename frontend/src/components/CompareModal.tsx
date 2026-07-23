import { useEffect, useRef, type ReactNode } from "react";
import { fmtDate } from "../format";
import type { JobResponse } from "../types";
import { BulletList } from "./BulletList";
import { RecommendationBadge } from "./RecommendationBadge";

// Attributes-as-rows / theses-as-columns, scoped to a small selection (the
// caller caps it). The first job is the current thesis (marked as such); the
// rest are the selected past ones. Opens as a native modal <dialog>; onClose
// fires on close (backdrop click, Close button, or Esc) so the parent unmounts.
const ROWS: { label: string; cell: (job: JobResponse) => ReactNode }[] = [
  { label: "Date", cell: (j) => (j.created_at ? fmtDate(j.created_at) : "-") },
  { label: "Score", cell: (j) => (j.thesis ? `${j.thesis.opportunity_score}/5` : "-") },
  {
    label: "Confidence",
    cell: (j) => (j.thesis ? `${Math.round(j.thesis.confidence_level * 100)}%` : "-"),
  },
  {
    label: "Recommendation",
    cell: (j) => (j.thesis ? <RecommendationBadge recommendation={j.thesis.recommendation} /> : "-"),
  },
  {
    label: "Key Themes",
    cell: (j) => (j.thesis ? <BulletList items={j.thesis.key_themes} empty="None" dotClass="bg-primary" /> : "-"),
  },
  {
    label: "Risks",
    cell: (j) => (j.thesis ? <BulletList items={j.thesis.risks} empty="None" dotClass="bg-error" /> : "-"),
  },
  {
    label: "Investment Signals",
    cell: (j) =>
      j.thesis ? <BulletList items={j.thesis.investment_signals} empty="None" dotClass="bg-accent" /> : "-",
  },
];

export function CompareModal({ jobs, onClose }: { jobs: JobResponse[]; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dlg = ref.current;
    if (dlg && !dlg.open) dlg.showModal();
  }, []);

  return (
    <dialog
      ref={ref}
      className="modal"
      // A click landing on the dialog itself (not the modal-box) hit the
      // ::backdrop - daisyUI's documented click-outside-to-close pattern.
      onClick={(e) => {
        if (e.target === ref.current) ref.current?.close();
      }}
      onClose={onClose}
    >
      <div className="modal-box max-w-5xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-sm">Compare Theses</h3>
          <button type="button" className="btn btn-sm btn-ghost" onClick={() => ref.current?.close()}>
            Close
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="table table-sm">
            <thead>
              <tr>
                <th className="w-28" />
                {jobs.map((job, i) => (
                  <th key={job.job_id} className="min-w-[200px] align-top">
                    {i === 0 && (
                      <span className="inline-block mb-1 px-1.5 py-0.5 rounded bg-primary/15 text-primary text-[10px] font-mono uppercase tracking-wider">
                        Current
                      </span>
                    )}
                    <a
                      href={`?job_id=${encodeURIComponent(job.job_id)}`}
                      className="block text-primary hover:text-primary/80 font-medium text-xs"
                    >
                      {job.query}
                    </a>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.label}>
                  <td className="text-xs font-mono text-base-content/60 align-top whitespace-nowrap">
                    {row.label}
                  </td>
                  {jobs.map((job) => (
                    <td key={job.job_id} className="align-top text-xs">
                      {row.cell(job)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </dialog>
  );
}
