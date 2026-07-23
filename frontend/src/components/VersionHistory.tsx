import type { ThesisResponse } from "../types";
import { Collapsible } from "./Collapsible";

// Version history (end-anchored feedback pairing). Nested inside the action bar
// (not boxed) with a top border as separator, matching the design's placement.
export function VersionHistory({
  history,
  feedbackHistory,
}: {
  history: ThesisResponse[];
  feedbackHistory: string[][];
}) {
  if (history.length === 0) return null;
  return (
    <Collapsible summary={`Previous versions (${history.length})`} boxed={false}>
      <div className="space-y-2 mt-3">
        {history.map((prev, i) => {
          // Pair from the end so a longer feedback_history can't shift annotations.
          const j = i + (feedbackHistory.length - history.length);
          const feedback = j >= 0 && j < feedbackHistory.length ? feedbackHistory[j] : [];
          return (
            <div key={i} className="text-xs px-3 py-2.5 bg-base-300/30 rounded-field">
              <p className="font-medium mb-0.5">{`Version ${i + 1}`}</p>
              <p className="text-base-content/60">
                {`Score: ${prev.opportunity_score}/5 · Recommendation: ${prev.recommendation}`}
              </p>
              {feedback && feedback.length > 0 && (
                <p className="mt-0.5 text-[10px] text-base-content/60">
                  {`Refined with: ${feedback.join(", ")}`}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </Collapsible>
  );
}
