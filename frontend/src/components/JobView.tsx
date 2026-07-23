import { fmtDate, refusalSummaryMessage } from "../format";
import type {
  ApproveHandler,
  CompareHandler,
  JobResponse,
  RefineHandler,
  ThesisResponse,
} from "../types";
import { ActionBar } from "./ActionBar";
import { Collapsible } from "./Collapsible";
import { ExecutionTrace } from "./ExecutionTrace";
import { ExportBar } from "./ExportBar";
import { Hallucination } from "./Hallucination";
import { MetricsStrip } from "./MetricsStrip";
import { RelatedTheses } from "./RelatedTheses";
import { SourcesList } from "./SourcesList";
import { ThesisDetails } from "./ThesisDetails";

// Raw model summary, with a local-summarizer warning and a refusal message
// substituted for the body when the summary was refused.
function RawSummary({ thesis }: { thesis: ThesisResponse }) {
  return (
    <Collapsible summary="Raw Summary" defaultOpen>
      <div>
        {thesis.summary_source === "local" && (
          <p className="text-xs text-accent border border-accent/30 bg-accent/10 rounded-field px-3 py-2 mb-3">
            Generated without an LLM (local extractive summarizer) - narrative quality may be reduced.
          </p>
        )}
        {thesis.summary_status === "refused" ? (
          <p className="text-sm text-base-content/60 leading-relaxed">
            {refusalSummaryMessage(thesis)}
          </p>
        ) : (
          <p className="text-sm text-base-content/60 leading-relaxed whitespace-pre-wrap">
            {thesis.raw_output}
          </p>
        )}
      </div>
    </Collapsible>
  );
}

// Full job composition: id line + export bar, the thesis card (metrics,
// sources, raw summary, details, related), then the action bar / approval
// notice and the diagnostic sections. Mirrors renderJob's early-return
// structure: a missing thesis or unparsed structured output stops before the
// trailing sections.
export function JobView({
  job,
  feedbackOptions,
  onRefine,
  onApprove,
  onCompare,
}: {
  job: JobResponse;
  feedbackOptions: string[];
  onRefine: RefineHandler;
  onApprove: ApproveHandler;
  onCompare: CompareHandler;
}) {
  const thesis = job.thesis;
  const parsed = thesis != null && thesis.key_themes.length > 0;

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-base-content/60 font-mono">
          {`Thesis generated · job_id: ${job.job_id}${
            job.created_at ? ` · ${fmtDate(job.created_at)}` : ""
          }`}
        </p>
        <ExportBar job={job} />
      </div>

      <div className="bg-base-200 border border-base-300 rounded-box px-6 py-5 space-y-4">
        {!thesis ? (
          <p>No thesis was returned.</p>
        ) : (
          <>
            <MetricsStrip thesis={thesis} />
            <SourcesList sources={job.sources} />
            {thesis.raw_output && <RawSummary thesis={thesis} />}
            {thesis.key_themes.length === 0 ? (
              <p className="text-sm text-base-content/60">
                Could not parse structured output. See raw output above.
              </p>
            ) : (
              <>
                <ThesisDetails thesis={thesis} />
                <RelatedTheses related={job.related_theses} onCompare={onCompare} />
              </>
            )}
          </>
        )}
      </div>

      {parsed &&
        (job.approved_at ? (
          <p className="flex items-center gap-1.5 text-xs text-primary font-semibold">
            This thesis has been approved. No further refinements needed.
          </p>
        ) : (
          <ActionBar
            job={job}
            feedbackOptions={feedbackOptions}
            onRefine={onRefine}
            onApprove={onApprove}
          />
        ))}

      {parsed && <Hallucination raw={job.hallucination} />}
      {parsed && <ExecutionTrace log={job.execution_log} />}
    </>
  );
}
