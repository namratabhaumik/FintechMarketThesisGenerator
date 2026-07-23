import { useState } from "react";
import { fmtDate } from "../format";
import type { CompareHandler, RelatedThesisResponse } from "../types";
import { Collapsible } from "./Collapsible";
import { RecommendationBadge } from "./RecommendationBadge";

// Compare renders as a table (attributes as rows), which stops being
// skimmable past a handful of columns. The modal always includes the current
// thesis as one column, so at most 2 past theses can be selected (2 + current
// = 3 columns total).
const MAX_COMPARE_PAST = 2;

function metaLine(r: RelatedThesisResponse): string {
  const date = r.created_at ? fmtDate(r.created_at) : "";
  let meta = [`score ${r.score}/5`, date].filter(Boolean).join(" · ");
  if (r.approved) meta += " · approved";
  return meta;
}

// Related past theses (episodic recall). Each row is selectable; picking one or
// two enables a side-by-side compare against the current thesis.
export function RelatedTheses({
  related,
  onCompare,
}: {
  related: RelatedThesisResponse[];
  onCompare: CompareHandler;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  if (related.length === 0) return null;

  const atMax = selected.length >= MAX_COMPARE_PAST;
  const toggle = (id: string, checked: boolean) => {
    setSelected((prev) => (checked ? [...prev, id] : prev.filter((x) => x !== id)));
  };

  return (
    <Collapsible summary={`Related past theses (${related.length})`}>
      <div className="space-y-2">
        {related.map((r) => {
          const checked = selected.includes(r.job_id);
          return (
            <div
              key={r.job_id}
              className="flex items-center justify-between py-2.5 px-3 rounded-field bg-base-300/50 hover:bg-base-300"
            >
              <input
                type="checkbox"
                className="checkbox checkbox-primary checkbox-xs flex-shrink-0 mr-3"
                aria-label={`Select "${r.query}" to compare`}
                checked={checked}
                disabled={!checked && atMax}
                onChange={(e) => toggle(r.job_id, e.target.checked)}
              />
              <div className="flex flex-col gap-0.5 min-w-0">
                <a
                  href={`?job_id=${encodeURIComponent(r.job_id)}`}
                  className="text-xs text-primary hover:text-primary/80 font-medium truncate"
                >
                  {r.query}
                </a>
                <span className="text-[10px] text-base-content/60 font-mono">{metaLine(r)}</span>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0 ml-4">
                <span className="text-[10px] text-base-content/60 font-mono">
                  {`${Math.round(r.similarity * 100)}% match`}
                </span>
                <RecommendationBadge recommendation={r.recommendation} />
              </div>
            </div>
          );
        })}
        <button
          type="button"
          className="btn btn-outline btn-xs mt-1"
          disabled={selected.length < 1}
          onClick={() => selected.length >= 1 && onCompare(selected)}
        >
          {selected.length > 0
            ? `Compare with current (${selected.length + 1})`
            : "Compare with current"}
        </button>
      </div>
    </Collapsible>
  );
}
