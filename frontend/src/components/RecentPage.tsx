import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import {
  ApiError,
  ErrorCode,
  approveThesis,
  createRefinement,
  createThesis,
  getFeedbackOptions,
  getThesis,
  listTheses,
} from "../api";
import { applyHighlights, clearHighlights } from "../anchoring";
import { isNoOpRound } from "../format";
import { RefinementStatus } from "../types";
import type {
  AuthInfo,
  ExecutionEvent,
  JobResponse,
  StatusMessage,
  ThesisSummaryResponse,
} from "../types";
import { PAGE_SIZE } from "../usePaginatedTheses";
import { useAnnotations } from "../useAnnotations";
import { AnnotationsPanel } from "./AnnotationsPanel";
import { CompareModal } from "./CompareModal";
import { SelectionPopover, useSelectionDraft } from "./SelectionPopover";
import { ErrorBoundary } from "./ErrorBoundary";
import { JobView } from "./JobView";
import { QueryPanel } from "./QueryPanel";

// Backend error codes the UI explains in its own words rather than surfacing
// the raw message. Codes absent here fall through to the generic error path,
// so adding one is an entry here, not a new branch. Values must match the
// codes emitted by routes.py (see ErrorCode in api.ts).
const GENERATE_ERROR_MESSAGES: Record<string, string> = {
  [ErrorCode.NoRelevantDocuments]:
    "No relevant documents found for this query. Try a broader or different fintech topic.",
  [ErrorCode.InsufficientEvidence]:
    "Not enough tagged evidence to build a complete thesis for this query.",
};

// The working page: generate a thesis, then refine/approve/compare it. The
// thesis on screen is addressed by the route (/thesis/:jobId), so the URL is
// the single source of truth - reloading, sharing or hitting back all resolve
// through the same restore path.
export function RecentPage({ auth }: { auth: AuthInfo }) {
  const { jobId } = useParams<{ jobId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [currentJob, setCurrentJob] = useState<JobResponse | null>(null);
  const [feedbackOptions, setFeedbackOptions] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusMessage>({ text: "", isError: false });
  const [resumeJobs, setResumeJobs] = useState<ThesisSummaryResponse[]>([]);
  const [resumeError, setResumeError] = useState(false);
  const [compareJobs, setCompareJobs] = useState<JobResponse[] | null>(null);
  const [generating, setGenerating] = useState(false);
  // Bumped to force-remount JobView (resetting the action bar's in-flight
  // state) after a failed refine/approve, redisplaying the unchanged job.
  const [nonce, setNonce] = useState(0);

  const feedbackRef = useRef<string[]>([]);
  // Ids this page already holds in state, so navigating to a thesis we just
  // created or refined doesn't refetch it.
  const loadedRef = useRef<string | null>(null);

  // --- Annotations ---
  // Anchored to the Raw Summary prose and pinned to the version on screen, so
  // refining shows a clean slate rather than notes pointing at rewritten text.
  const summaryRef = useRef<HTMLParagraphElement | null>(null);
  // Radix unmounts the inactive tab, so switching to Diff and back gives a
  // brand-new paragraph node. A plain ref assignment is invisible to effects,
  // which would leave the remounted summary unpainted until something else
  // changed. This callback ref makes the (re)mount itself a dependency.
  const [summaryMounts, setSummaryMounts] = useState(0);
  const attachSummary = useCallback((el: HTMLParagraphElement | null) => {
    summaryRef.current = el;
    if (el) setSummaryMounts((n) => n + 1);
  }, []);
  const [panelOpen, setPanelOpen] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const version = (currentJob?.refinement_count ?? 0) + 1;
  const annotations = useAnnotations(currentJob?.job_id, version);
  // Enabled whenever a thesis is on screen, NOT only when the panel is open:
  // gating it on the panel made annotating undiscoverable, since nothing hints
  // that you must open the panel before selecting text.
  const { draft, lockDraft, clearDraft } = useSelectionDraft(
    summaryRef,
    Boolean(currentJob),
  );

  // --- Helpers ---

  const setStatusText = (text: string, isError = false) => setStatus({ text, isError });

  // Every action funnels failures through here. A dead session is reported the
  // same way everywhere (it can expire mid-refine as easily as mid-generate),
  // and it must not be prefixed with the action name - "Refinement failed" is
  // the wrong thing to tell someone who just needs to sign in again.
  const reportError = (err: unknown, fallback: string, prefix: string) => {
    if (err instanceof ApiError && err.code === ErrorCode.SessionExpired) {
      setStatusText("Your session has expired. Please sign out and sign in again.", true);
    } else if (err instanceof ApiError) {
      setStatusText(`${prefix}: ${err.message}`, true);
    } else {
      console.error(fallback, err);
      setStatusText(fallback, true);
    }
  };

  // Load the fixed feedback options once. Non-fatal: a failure just leaves the
  // refine panel without options, so it never blocks showing a thesis.
  const ensureFeedbackOptions = async () => {
    if (feedbackRef.current.length > 0) return;
    try {
      const opts = await getFeedbackOptions();
      feedbackRef.current = opts;
      setFeedbackOptions(opts);
    } catch (err) {
      console.error("Failed to load feedback options", err);
    }
  };

  const redisplayCurrent = () => setNonce((n) => n + 1);

  // Adopt a job returned by an action, marking it loaded so the route effect
  // doesn't turn around and refetch it.
  const adopt = (job: JobResponse) => {
    loadedRef.current = job.job_id;
    setCurrentJob(job);
  };

  // --- Actions ---

  const generate = async () => {
    const q = query.trim();
    if (!q) {
      setStatusText("Please enter a non-empty query.");
      return;
    }
    setGenerating(true);
    setCurrentJob(null);
    setStatusText("Retrieving context and generating thesis...");
    try {
      const job = await createThesis(q);
      adopt(job);
      setStatusText("");
      await ensureFeedbackOptions();
      // The route carries the run, so a refresh or shared link restores it.
      void navigate(`/thesis/${encodeURIComponent(job.job_id)}`);
    } catch (err) {
      const explained = err instanceof ApiError ? GENERATE_ERROR_MESSAGES[err.code] : undefined;
      if (explained) {
        setStatusText(explained);
      } else {
        reportError(err, "An unexpected error occurred. Is the API running?", "Error");
      }
    } finally {
      setGenerating(false);
    }
  };

  const onRefine = (id: string, feedback: string[]) => {
    setStatusText("Refining thesis based on your feedback...");
    void (async () => {
      try {
        const job = await createRefinement(id, feedback);
        adopt(job);
        // An executed round that changed nothing gets said outright; a silent
        // re-render of an identical thesis reads as "nothing happened".
        const events = job.execution_log as ExecutionEvent[];
        const last = events.length > 0 ? events[events.length - 1] : undefined;
        setStatusText(
          isNoOpRound(last)
            ? "This round made no changes - the thesis reflects the selected feedback. " +
                "Try different feedback, or approve if it looks right."
            : "",
        );
      } catch (err) {
        reportError(err, "An unexpected error occurred during refinement.", "Refinement failed");
        redisplayCurrent();
      }
    })();
  };

  const onApprove = (id: string) => {
    setStatusText("Approving thesis...");
    void (async () => {
      try {
        const job = await approveThesis(id);
        adopt(job);
        setStatusText("");
      } catch (err) {
        reportError(err, "An unexpected error occurred during approval.", "Approval failed");
        redisplayCurrent();
      }
    })();
  };

  // Open the compare modal with the current thesis as the first column plus the
  // selected past ones (already capped in the view).
  //
  // allSettled, not all: one unloadable thesis should cost its own column, not
  // the whole comparison. But a partial load is reported rather than silently
  // rendering fewer columns than were asked for.
  const onCompare = (jobIds: string[]) => {
    if (!currentJob) return;
    setStatusText("Loading theses to compare...");
    void (async () => {
      const results = await Promise.allSettled(jobIds.map((id) => getThesis(id)));
      for (const r of results) {
        if (r.status === "rejected") console.error("Failed to load a thesis to compare", r.reason);
      }
      const selected = results
        .filter((r): r is PromiseFulfilledResult<JobResponse> => r.status === "fulfilled")
        .map((r) => r.value);
      if (selected.length < 1) {
        setStatusText("Could not load the selected theses to compare.", true);
        return;
      }
      const failed = results.length - selected.length;
      setStatusText(
        failed > 0
          ? `Comparing without ${failed} thesis that could not be loaded.`
          : "",
        failed > 0,
      );
      setCompareJobs([currentJob, ...selected]);
    })();
  };

  // Resuming is just navigation now: the route effect below does the loading.
  const onResume = async (id: string): Promise<void> => {
    await navigate(`/thesis/${encodeURIComponent(id)}`);
  };

  // Legacy ?job_id= links (shared before routing existed) redirect onto the
  // route so there is only ever one URL shape in play.
  useEffect(() => {
    const legacy = searchParams.get("job_id");
    if (!jobId && legacy) {
      void navigate(`/thesis/${encodeURIComponent(legacy)}`, { replace: true });
    }
  }, [jobId, searchParams, navigate]);

  // The route owns which thesis is shown: load whenever it names one we don't
  // already hold. Clearing on a bare "/" is what makes the logo/Recent link
  // return to an empty workspace.
  useEffect(() => {
    if (!jobId) {
      if (!searchParams.get("job_id")) {
        loadedRef.current = null;
        setCurrentJob(null);
        setQuery("");
      }
      return;
    }
    if (loadedRef.current === jobId) return;

    let cancelled = false;
    setStatusText("Loading saved thesis...");
    void (async () => {
      try {
        await ensureFeedbackOptions();
        const job = await getThesis(jobId);
        if (cancelled) return;
        loadedRef.current = jobId;
        setCurrentJob(job);
        setQuery(job.query);
        setStatusText("");
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setStatusText("That saved thesis was not found.", true);
        } else {
          reportError(err, "An unexpected error occurred loading the saved thesis.", "Error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // ensureFeedbackOptions/reportError are stable closures over setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  // Repaint on any change to the annotations, the active thread, or the
  // document itself. clearHighlights runs on teardown so React never unmounts a
  // paragraph whose text nodes we have split.
  const repaint = useCallback(() => {
    const root = summaryRef.current;
    if (!root) return;
    applyHighlights(
      root,
      annotations.threads
        .filter((t) => t.root.start_offset != null && t.root.end_offset != null)
        .map((t) => ({
          id: t.root.id,
          start: t.root.start_offset ?? 0,
          end: t.root.end_offset ?? 0,
          active: t.root.id === activeId,
        })),
      (id) => {
        setActiveId(id);
        setPanelOpen(true);
      },
    );
  }, [annotations.threads, activeId]);

  useEffect(() => {
    repaint();
    const root = summaryRef.current;
    return () => {
      if (root) clearHighlights(root);
    };
  }, [repaint, currentJob?.job_id, nonce, summaryMounts]);

  const saveDraft = (body: string) => {
    if (!draft) return;
    void (async () => {
      const ok = await annotations.add({
        section: "raw_summary",
        start: draft.start,
        end: draft.end,
        quote: draft.quote,
        body,
      });
      if (ok) {
        clearDraft();
        window.getSelection()?.removeAllRanges();
        // Reveal where the note went, so the first one is not saved into an
        // invisible panel.
        setPanelOpen(true);
      }
    })();
  };

  // Resumable runs for the picker (only mid-refinement runs qualify).
  useEffect(() => {
    listTheses(PAGE_SIZE, 0, RefinementStatus.Refining)
      .then(setResumeJobs)
      .catch((err) => {
        console.error("Failed to load resumable sessions", err);
        setResumeError(true);
      });
  }, []);

  return (
    <>
      <QueryPanel
        query={query}
        onQueryChange={setQuery}
        onGenerate={() => void generate()}
        generating={generating}
        resumeJobs={resumeJobs}
        resumeError={resumeError}
        onResume={onResume}
        status={status}
      />

      {/* Document + annotations. The document column is flex-1 min-w-0, so it
          reclaims the full width whenever the panel is closed. */}
      <div className="flex gap-0">
        <section className="flex-1 min-w-0 px-6 md:px-8 pb-16 space-y-4">
          {currentJob && (
            <>
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => setPanelOpen((open) => !open)}
                  aria-pressed={panelOpen}
                  className="print:hidden flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-field border border-base-300 text-base-content/60 hover:text-base-content hover:border-base-content/30 transition-colors"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Notes
                  {annotations.threads.filter((t) => t.root.resolution == null).length > 0 && (
                    <span className="text-primary font-mono">
                      {annotations.threads.filter((t) => t.root.resolution == null).length}
                    </span>
                  )}
                </button>
              </div>

              <ErrorBoundary
                key={`${currentJob.job_id}:${currentJob.refinement_count}:${nonce}`}
                fallback={
                  <p className="text-sm text-base-content/60">
                    Could not display the thesis (unexpected response shape).
                  </p>
                }
              >
                <JobView
                  job={currentJob}
                  feedbackOptions={feedbackOptions}
                  onRefine={onRefine}
                  onApprove={onApprove}
                  onCompare={onCompare}
                  summaryRef={attachSummary}
                />
              </ErrorBoundary>
            </>
          )}
        </section>

        {/* From lg up the panel is a column beside the document; below that it
            would leave the document unreadably narrow, so it becomes a sheet. */}
        {currentJob && panelOpen && (
          <aside className="print:hidden hidden lg:block w-80 flex-shrink-0 border-l border-base-300 sticky top-14 h-[calc(100vh-3.5rem)]">
            <AnnotationsPanel
              threads={annotations.threads}
              loading={annotations.loading}
              error={annotations.error}
              userId={auth.userId}
              activeId={activeId}
              onSelect={setActiveId}
              onReply={(parentId, body) => void annotations.reply(parentId, body)}
              onEdit={(id, body) => void annotations.edit(id, body)}
              onDelete={(id) => void annotations.remove(id)}
              onResolve={(id, resolution) => void annotations.resolve(id, resolution)}
              onClose={() => setPanelOpen(false)}
            />
          </aside>
        )}
      </div>

      {/* Mobile: a bottom sheet rather than a side column. */}
      {currentJob && panelOpen && (
        <div className="lg:hidden fixed inset-0 z-[70] print:hidden">
          <button
            type="button"
            aria-label="Close annotations"
            className="absolute inset-0 w-full bg-black/50"
            onClick={() => setPanelOpen(false)}
          />
          <div className="absolute left-0 right-0 bottom-0 h-[70vh] bg-base-100 border-t border-base-300 rounded-t-box overflow-hidden">
            <AnnotationsPanel
              threads={annotations.threads}
              loading={annotations.loading}
              error={annotations.error}
              userId={auth.userId}
              activeId={activeId}
              onSelect={setActiveId}
              onReply={(parentId, body) => void annotations.reply(parentId, body)}
              onEdit={(id, body) => void annotations.edit(id, body)}
              onDelete={(id) => void annotations.remove(id)}
              onResolve={(id, resolution) => void annotations.resolve(id, resolution)}
              onClose={() => setPanelOpen(false)}
            />
          </div>
        </div>
      )}

      {draft && (
        <SelectionPopover
          draft={draft}
          onCompose={lockDraft}
          onSubmit={saveDraft}
          onDismiss={clearDraft}
        />
      )}

      {compareJobs && <CompareModal jobs={compareJobs} onClose={() => setCompareJobs(null)} />}
    </>
  );
}
