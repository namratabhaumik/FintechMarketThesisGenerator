import { fmtDate } from "../format";
import type { ThesisResponse } from "../types";
import { RecommendationBadge } from "./RecommendationBadge";

// Score / confidence / recommendation strip. Rendered first (right after the
// job-id line), matching the design's placement ahead of Sources/Raw Summary.
export function MetricsStrip({ thesis }: { thesis: ThesisResponse }) {
  const pct = Math.round(thesis.confidence_level * 100);
  return (
    <div className="border-b border-base-300 pb-4 flex flex-wrap items-center justify-between gap-6">
      <div className="min-w-0">
        <p className="text-xs text-base-content/60 font-mono uppercase tracking-wider mb-1">
          Investment Score
        </p>
        <div className="flex items-end gap-2">
          <span className="text-3xl font-bold font-mono leading-none">{thesis.opportunity_score}</span>
          <span className="text-sm text-base-content/60 mb-0.5 font-mono">/5</span>
        </div>
      </div>

      <div className="w-48 min-w-0">
        <p className="text-xs text-base-content/60 font-mono uppercase tracking-wider mb-1">
          Confidence
        </p>
        <div className="flex items-end gap-2 mb-1.5">
          <span className="text-3xl font-bold font-mono leading-none">{pct}</span>
          <span className="text-sm text-base-content/60 mb-0.5 font-mono">%</span>
        </div>
        <div
          className="h-1.5 w-full bg-base-300 rounded-full overflow-hidden"
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemax={100}
        >
          <div className="h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
        </div>
        {thesis.confidence_as_of && (
          <p className="text-[10px] text-base-content/60 font-mono mt-1">
            {`trends as of ${fmtDate(thesis.confidence_as_of)}`}
          </p>
        )}
      </div>

      <div className="flex flex-col items-start gap-2">
        <p className="text-xs text-base-content/60 font-mono uppercase tracking-wider">
          Recommendation
        </p>
        <RecommendationBadge recommendation={thesis.recommendation} />
      </div>
    </div>
  );
}
