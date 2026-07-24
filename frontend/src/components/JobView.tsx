import * as Tabs from "@radix-ui/react-tabs";
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
import { DiffView } from "./DiffView";
import { ExecutionTrace } from "./ExecutionTrace";
import { ExportBar } from "./ExportBar";
import { Hallucination } from "./Hallucination";
import { MetricsStrip } from "./MetricsStrip";
import { RelatedTheses } from "./RelatedTheses";
import { SourcesList } from "./SourcesList";
import { ThesisDetails } from "./ThesisDetails";

// daisyUI-styled Radix tab trigger. Radix owns keyboard/focus/a11y (arrow-key
// nav, roving tabindex) across browsers; the classes keep the daisyUI look.
const tabTriggerClass =
  "px-3 py-1.5 text-xs rounded-field text-base-content/60 cursor-pointer transition-colors data-[state=active]:bg-base-300 data-[state=active]:text-base-content";

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
  const version = job.refinement_count + 1;

  return (
    <Tabs.Root defaultValue="document">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="text-xs font-mono text-primary border border-primary/30 bg-primary/10 rounded px-1.5 py-0.5">
            v{version}
          </span>
          <h2 className="text-lg font-semibold mt-1.5 break-words">{job.query}</h2>
          <p className="text-[10px] text-base-content/50 font-mono mt-0.5 break-all">
            job_id: {job.job_id}
            {job.created_at ? ` · ${fmtDate(job.created_at)}` : ""}
          </p>
        </div>
        <Tabs.List
          aria-label="Thesis view"
          className="flex gap-1 bg-base-200 border border-base-300 rounded-field p-1 flex-shrink-0"
        >
          <Tabs.Trigger value="document" className={tabTriggerClass}>
            Document
          </Tabs.Trigger>
          <Tabs.Trigger value="diff" className={tabTriggerClass}>
            Diff
          </Tabs.Trigger>
        </Tabs.List>
      </div>

      <Tabs.Content value="document" className="mt-4 space-y-4 focus:outline-none">
        <div className="flex justify-end">
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
      </Tabs.Content>

      <Tabs.Content value="diff" className="mt-4 focus:outline-none">
        <DiffView job={job} />
      </Tabs.Content>
    </Tabs.Root>
  );
}
