import type { JobResponse } from "../types";
import { MetricsStrip } from "./MetricsStrip";

// Split text into sentences WITHOUT regex lookbehind - Safari < 16.4 lacks it
// and would throw at parse time, so lookbehind is off-limits (browser-agnostic).
// Matches runs of non-terminal chars followed by any trailing .!? punctuation.
function sentences(text: string): string[] {
  const out: string[] = [];
  const re = /[^.!?]+[.!?]*/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const s = m[0].trim();
    if (s) out.push(s);
  }
  return out;
}

// One side of the side-by-side diff. A sentence present in the other version is
// muted (unchanged); one that isn't is highlighted - struck red for removed,
// tinted emerald for added.
function DiffColumn({
  title,
  sentences: sents,
  otherSet,
  mode,
}: {
  title: string;
  sentences: string[];
  otherSet: Set<string>;
  mode: "removed" | "added";
}) {
  return (
    <div className="border border-base-300 rounded-box overflow-hidden">
      <div className="px-4 py-2 border-b border-base-300 bg-base-200/50">
        <p className="text-xs font-mono text-base-content/60 uppercase tracking-wider">{title}</p>
      </div>
      <div className="p-3 space-y-2">
        {sents.length === 0 ? (
          <p className="text-xs text-base-content/40">(empty)</p>
        ) : (
          sents.map((s, i) => {
            const changed = !otherSet.has(s);
            const base = "text-sm leading-relaxed rounded-field px-3 py-2 border block";
            const cls = !changed
              ? `${base} border-transparent text-base-content/70`
              : mode === "removed"
                ? `${base} border-error/30 bg-error/10 text-error/80`
                : `${base} border-primary/30 bg-primary/10 text-base-content`;
            return (
              <span key={i} className={cls}>
                <span className={changed && mode === "removed" ? "line-through" : ""}>{s}</span>
              </span>
            );
          })
        )}
      </div>
    </div>
  );
}

// Diffs the current thesis against the most recent previous version, using the
// stored thesis_history / feedback_history (no backend). Read-only and
// browser-agnostic. A "revert to previous" action is intentionally omitted: it
// is a write and needs a new backend endpoint + version semantics.
export function DiffView({ job }: { job: JobResponse }) {
  const current = job.thesis;
  const history = job.thesis_history;

  if (!current || history.length === 0) {
    return (
      <p className="text-sm text-base-content/60">
        No previous version to compare yet. Refine this thesis to see what changed between versions.
      </p>
    );
  }

  const prev = history[history.length - 1];
  const prevVersion = history.length;
  const currVersion = job.refinement_count + 1;
  const lastFeedback = job.feedback_history[job.feedback_history.length - 1] ?? [];

  const prevSents = sentences(prev.raw_output ?? "");
  const currSents = sentences(current.raw_output ?? "");
  const prevSet = new Set(prevSents);
  const currSet = new Set(currSents);

  return (
    <div className="space-y-6">
      <MetricsStrip thesis={current} />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="border border-error/30 bg-error/10 text-error/80 rounded px-2 py-1">
            v{prevVersion} — {prev.opportunity_score}/5
          </span>
          <span className="text-base-content/40" aria-hidden>
            →
          </span>
          <span className="border border-primary/30 bg-primary/10 text-primary rounded px-2 py-1">
            v{currVersion} — {current.opportunity_score}/5
          </span>
        </div>
        {lastFeedback.length > 0 && (
          <p className="text-xs font-mono text-base-content/60">
            Refined with: &quot;{lastFeedback.join(", ")}&quot;
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DiffColumn
          title={`Previous (v${prevVersion})`}
          sentences={prevSents}
          otherSet={currSet}
          mode="removed"
        />
        <DiffColumn
          title={`Current (v${currVersion})`}
          sentences={currSents}
          otherSet={prevSet}
          mode="added"
        />
      </div>
    </div>
  );
}
