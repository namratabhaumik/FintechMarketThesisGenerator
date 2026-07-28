import { useEffect, useRef, useState } from "react";
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
import { isNoOpRound } from "../format";
import { RefinementStatus } from "../types";
import type {
  ExecutionEvent,
  JobResponse,
  StatusMessage,
  ThesisSummaryResponse,
} from "../types";
import { PAGE_SIZE } from "../usePaginatedTheses";
import { CompareModal } from "./CompareModal";
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
export function RecentPage() {
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

      <section className="px-6 md:px-8 pb-16 space-y-4">
        {currentJob && (
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
            />
          </ErrorBoundary>
        )}
      </section>

      {compareJobs && <CompareModal jobs={compareJobs} onClose={() => setCompareJobs(null)} />}
    </>
  );
}
