import { fmtDate } from "../format";
import type { DeleteHandler, ThesisSummaryResponse } from "../types";
import { Collapsible } from "./Collapsible";
import { RecommendationBadge } from "./RecommendationBadge";

function metaLine(j: ThesisSummaryResponse, isAdmin?: boolean): string {
  const date = j.created_at ? fmtDate(j.created_at) : "";
  const parts = j.opportunity_score != null ? [`score ${j.opportunity_score}/5`, date] : [date];
  let meta = parts.filter(Boolean).join(" · ");
  if (j.approved_at) meta += " · approved";
  else if (j.refinement_status && j.refinement_status !== "N/A") meta += ` · ${j.refinement_status}`;
  // In the admin (all-users) view, label whose thesis each row is. Only the
  // short prefix of the owner UUID is available.
  if (isAdmin && j.user_id) meta += ` · owner ${j.user_id.slice(0, 8)}`;
  return meta;
}

// Browsable research library. Used both for the caller's own past theses and,
// with isAdmin + onDelete, for the admin cross-user management list. Without
// pagination an empty list collapses to nothing; with it the shell stays
// mounted (a page can be empty while other pages have content).
interface PastThesesListProps {
  jobs: ThesisSummaryResponse[];
  onPrevPage?: () => void;
  onNextPage?: () => void;
  canPrevPage?: boolean;
  canNextPage?: boolean;
  isAdmin?: boolean;
  onDelete?: DeleteHandler;
  title?: string;
}

export function PastThesesList({
  jobs,
  onPrevPage,
  onNextPage,
  canPrevPage,
  canNextPage,
  isAdmin,
  onDelete,
  title = "Past theses",
}: PastThesesListProps) {
  const paginated = Boolean(onPrevPage || onNextPage);
  if (jobs.length === 0 && !paginated) return null;

  return (
    <Collapsible summary={`${title} (${jobs.length})`}>
      <div className="space-y-2">
        {jobs.length === 0 && (
          <p className="text-xs text-base-content/60">No other past theses on this page.</p>
        )}

        {jobs.map((j) => (
          <div
            key={j.job_id}
            className="flex items-center justify-between py-2.5 px-3 rounded-field bg-base-300/50 hover:bg-base-300"
          >
            <div className="flex flex-col gap-0.5 min-w-0">
              <a
                href={`?job_id=${encodeURIComponent(j.job_id)}`}
                className="text-xs text-primary hover:text-primary/80 font-medium truncate"
              >
                {j.query}
              </a>
              <span className="text-[10px] text-base-content/60 font-mono">
                {metaLine(j, isAdmin)}
              </span>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0 ml-4">
              {j.recommendation && <RecommendationBadge recommendation={j.recommendation} />}
              {isAdmin && onDelete && (
                <button
                  type="button"
                  className="btn btn-ghost btn-xs text-error/70 hover:text-error"
                  aria-label={`Delete thesis: ${j.query}`}
                  title="Delete (admin)"
                  onClick={() => {
                    if (window.confirm(`Delete this thesis permanently?\n\n"${j.query}"`)) {
                      onDelete(j.job_id);
                    }
                  }}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                    <path d="M10 11v6" />
                    <path d="M14 11v6" />
                    <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        ))}

        {paginated && (
          <div className="mt-3 flex gap-2">
            {onPrevPage && (
              <button
                type="button"
                className="btn btn-xs btn-outline"
                disabled={!canPrevPage}
                onClick={onPrevPage}
              >
                ← Previous
              </button>
            )}
            {onNextPage && (
              <button
                type="button"
                className="btn btn-xs btn-outline"
                disabled={!canNextPage}
                onClick={onNextPage}
              >
                Next →
              </button>
            )}
          </div>
        )}
      </div>
    </Collapsible>
  );
}
