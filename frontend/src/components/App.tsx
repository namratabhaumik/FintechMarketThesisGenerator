import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  ErrorCode,
  approveThesis,
  createRefinement,
  createThesis,
  deleteThesis,
  getFeedbackOptions,
  getThesis,
  listTheses,
} from "../api";
import { isNoOpRound } from "../format";
import { RefinementStatus } from "../types";
import type {
  AuthInfo,
  ExecutionEvent,
  JobResponse,
  StatusMessage,
  ThesisSummaryResponse,
} from "../types";
import { PAGE_SIZE, usePaginatedTheses } from "../usePaginatedTheses";
import { AppHeader } from "./AppHeader";
import { CompareModal } from "./CompareModal";
import { ErrorBoundary } from "./ErrorBoundary";
import { JobView } from "./JobView";
import { PastThesesList } from "./PastThesesList";
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

// Controller: owns app state (current job, feedback options, status) and
// orchestrates network calls. The section components are pure and receive
// callbacks from here.
export function App({ auth }: { auth: AuthInfo }) {
  const [currentJob, setCurrentJob] = useState<JobResponse | null>(null);
  const [feedbackOptions, setFeedbackOptions] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusMessage>({ text: "", isError: false });
  const [resumeJobs, setResumeJobs] = useState<ThesisSummaryResponse[]>([]);
  const [resumeError, setResumeError] = useState(false);
  const [compareJobs, setCompareJobs] = useState<JobResponse[] | null>(null);
  const [generating, setGenerating] = useState(false);
  // The caller's own past runs and the admin-only cross-user list. Same
  // pagination machinery, so both come from the same hook.
  const past = usePaginatedTheses({ errorLabel: "Failed to load past theses" });
  const all = usePaginatedTheses({
    allUsers: true,
    enabled: Boolean(auth.isAdmin),
    errorLabel: "Failed to load all theses (admin)",
  });
  // Bumped to force-remount JobView (resetting the action bar's in-flight state)
  // after a failed refine/approve, mirroring app.ts re-rendering the current job.
  const [nonce, setNonce] = useState(0);

  const feedbackRef = useRef<string[]>([]);

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
      setCurrentJob(job);
      // A ?job_id URL makes the run restorable on refresh/new tab.
      history.replaceState(null, "", `?job_id=${encodeURIComponent(job.job_id)}`);
      await ensureFeedbackOptions();
      setStatusText("");
      // Past Theses excludes the now-current job, so a thesis we just switched
      // away from surfaces and this fresh one stays out.
      void past.load();
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

  const onRefine = (jobId: string, feedback: string[]) => {
    setStatusText("Refining thesis based on your feedback...");
    void (async () => {
      try {
        const job = await createRefinement(jobId, feedback);
        setCurrentJob(job);
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

  const onApprove = (jobId: string) => {
    setStatusText("Approving thesis...");
    void (async () => {
      try {
        const job = await approveThesis(jobId);
        setCurrentJob(job);
        setStatusText("");
      } catch (err) {
        reportError(err, "An unexpected error occurred during approval.", "Approval failed");
        redisplayCurrent();
      }
    })();
  };

  // Open the compare modal with the current thesis as the first column plus the
  // selected past ones (already capped in the view).
  const onCompare = (jobIds: string[]) => {
    if (!currentJob) return;
    setStatusText("Loading theses to compare...");
    void (async () => {
      const results = await Promise.allSettled(jobIds.map((id) => getThesis(id)));
      const selected = results
        .filter((r): r is PromiseFulfilledResult<JobResponse> => r.status === "fulfilled")
        .map((r) => r.value);
      setStatusText("");
      if (selected.length < 1) {
        setStatusText("Could not load the selected theses to compare.");
        return;
      }
      setCompareJobs([currentJob, ...selected]);
    })();
  };

  // Load a persisted job by id and render it. Returns whether it loaded, so the
  // resume picker can update the URL on success; the ?job_id path ignores it.
  const restore = async (jobId: string): Promise<boolean> => {
    setStatusText("Loading saved thesis...");
    try {
      await ensureFeedbackOptions();
      const job = await getThesis(jobId);
      setCurrentJob(job);
      setQuery(job.query);
      setStatusText("");
      void past.load();
      return true;
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setStatusText("That saved thesis was not found.");
      } else {
        reportError(err, "An unexpected error occurred loading the saved thesis.", "Error");
      }
      return false;
    }
  };

  const onResume = async (jobId: string): Promise<void> => {
    // Picker persists after a resume so you can switch sessions without reload.
    if (await restore(jobId)) {
      history.replaceState(null, "", `?job_id=${encodeURIComponent(jobId)}`);
    }
  };

  const onDeleteThesis = (jobId: string) => {
    void (async () => {
      try {
        await deleteThesis(jobId);
      } catch (err) {
        reportError(err, "Could not delete thesis.", "Delete failed");
        return;
      }
      // Deletion can affect either list, so refresh both.
      void all.load();
      void past.load();
    })();
  };

  // Boot: offer the resume picker (only mid-refinement runs are resumable), load
  // the past-theses library (+ admin list), and restore a specific run when the
  // URL carries ?job_id.
  useEffect(() => {
    listTheses(PAGE_SIZE, 0, RefinementStatus.Refining)
      .then(setResumeJobs)
      .catch((err) => {
        console.error("Failed to load resumable sessions", err);
        setResumeError(true);
      });
    void past.load();
    void all.load();
    const jobId = new URLSearchParams(location.search).get("job_id");
    if (jobId) void restore(jobId);
    // Mount-only boot sequence; handlers close over stable setters. Adding the
    // hook/restore deps would re-run this on every render and refetch in a loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <AppHeader auth={auth} />

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

      <section className="print:hidden px-6 md:px-8 pb-16 -mt-8">
        <PastThesesList
          jobs={past.page.filter((j) => j.job_id !== currentJob?.job_id)}
          onPrevPage={() => past.goToPage(-1)}
          onNextPage={() => past.goToPage(1)}
          canPrevPage={past.offset > 0}
          canNextPage={past.hasMore}
        />
      </section>

      {auth.isAdmin && (
        <section className="print:hidden px-6 md:px-8 pb-16 -mt-8">
          <PastThesesList
            jobs={all.page.filter((j) => j.job_id !== currentJob?.job_id)}
            onPrevPage={() => all.goToPage(-1)}
            onNextPage={() => all.goToPage(1)}
            canPrevPage={all.offset > 0}
            canNextPage={all.hasMore}
            isAdmin
            onDelete={onDeleteThesis}
            title="Admin - all users' theses"
          />
        </section>
      )}

      {compareJobs && <CompareModal jobs={compareJobs} onClose={() => setCompareJobs(null)} />}
    </>
  );
}
