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
import type { ExecutionEvent, JobResponse, ThesisSummaryResponse } from "../types";
import { CompareModal } from "./CompareModal";
import { ErrorBoundary } from "./ErrorBoundary";
import { JobView } from "./JobView";
import { PastThesesList } from "./PastThesesList";
import { ResumePicker } from "./ResumePicker";

const PAGE_SIZE = 10;

/** Signed-in user info + sign-out handler, passed in by the auth gate. */
export interface AuthInfo {
  email?: string | null;
  isAdmin?: boolean;
  onSignOut: () => void;
}

// Clicking the logo/name reloads to a clean root.
function goHome() {
  window.location.href = window.location.pathname;
}

// Controller: owns app state (current job, feedback options, status) and
// orchestrates network calls. The section components are pure and receive
// callbacks from here.
export function App({ auth }: { auth: AuthInfo }) {
  const [currentJob, setCurrentJob] = useState<JobResponse | null>(null);
  const [feedbackOptions, setFeedbackOptions] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<{ text: string; isError: boolean }>({
    text: "",
    isError: false,
  });
  const [resumeJobs, setResumeJobs] = useState<ThesisSummaryResponse[]>([]);
  const [resumeError, setResumeError] = useState(false);
  const [compareJobs, setCompareJobs] = useState<JobResponse[] | null>(null);
  const [generating, setGenerating] = useState(false);
  // The caller's own past runs (page + whether a next page exists), and the
  // admin-only cross-user list. Offsets drive server-side pagination.
  const [pastPage, setPastPage] = useState<ThesisSummaryResponse[]>([]);
  const [pastHasMore, setPastHasMore] = useState(false);
  const [pastOffset, setPastOffset] = useState(0);
  const [allPage, setAllPage] = useState<ThesisSummaryResponse[]>([]);
  const [allHasMore, setAllHasMore] = useState(false);
  const [allOffset, setAllOffset] = useState(0);
  // Bumped to force-remount JobView (resetting the action bar's in-flight state)
  // after a failed refine/approve, mirroring app.ts re-rendering the current job.
  const [nonce, setNonce] = useState(0);

  const feedbackRef = useRef<string[]>([]);

  // --- Helpers ---

  const setStatusText = (text: string, isError = false) => setStatus({ text, isError });

  const reportError = (err: unknown, fallback: string, prefix: string) => {
    if (err instanceof ApiError) {
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
      void showPastTheses();
    } catch (err) {
      if (err instanceof ApiError && err.code === ErrorCode.NoRelevantDocuments) {
        setStatusText(
          "No relevant documents found for this query. Try a broader or different fintech topic.",
        );
      } else if (err instanceof ApiError && err.code === ErrorCode.InsufficientEvidence) {
        setStatusText("Not enough tagged evidence to build a complete thesis for this query.");
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
      const past = results
        .filter((r): r is PromiseFulfilledResult<JobResponse> => r.status === "fulfilled")
        .map((r) => r.value);
      setStatusText("");
      if (past.length < 1) {
        setStatusText("Could not load the selected theses to compare.");
        return;
      }
      setCompareJobs([currentJob, ...past]);
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
      void showPastTheses();
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

  // The caller's OWN past runs the user can switch to. The current thesis (shown
  // in the results panel) is filtered out at render time, so switching away from
  // a thesis surfaces it here and the one you're viewing never self-lists.
  const showPastTheses = async (offset = pastOffset) => {
    try {
      // Over-fetch by one to detect a next page without a total count.
      const fetched = await listTheses(PAGE_SIZE + 1, offset);
      setPastHasMore(fetched.length > PAGE_SIZE);
      setPastPage(fetched.slice(0, PAGE_SIZE));
    } catch (err) {
      console.error("Failed to load past theses", err);
      setPastPage([]);
    }
  };

  const pagePast = (direction: number) => {
    const next = Math.max(0, pastOffset + direction * PAGE_SIZE);
    setPastOffset(next);
    void showPastTheses(next);
  };

  // Admin-only cross-user management list: every user's theses with owner labels
  // and a delete control (the backend 403s this for non-admins).
  const showAllTheses = async (offset = allOffset) => {
    if (!auth.isAdmin) return;
    try {
      const fetched = await listTheses(PAGE_SIZE + 1, offset, undefined, true);
      setAllHasMore(fetched.length > PAGE_SIZE);
      setAllPage(fetched.slice(0, PAGE_SIZE));
    } catch (err) {
      console.error("Failed to load all theses (admin)", err);
      setAllPage([]);
    }
  };

  const pageAll = (direction: number) => {
    const next = Math.max(0, allOffset + direction * PAGE_SIZE);
    setAllOffset(next);
    void showAllTheses(next);
  };

  const onDeleteThesis = async (jobId: string): Promise<void> => {
    try {
      await deleteThesis(jobId);
    } catch (err) {
      reportError(err, "Could not delete thesis.", "Delete failed");
      return;
    }
    // Deletion can affect either list, so refresh both.
    void showAllTheses();
    void showPastTheses();
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
    void showPastTheses();
    if (auth.isAdmin) void showAllTheses();
    const jobId = new URLSearchParams(location.search).get("job_id");
    if (jobId) void restore(jobId);
    // Mount-only boot sequence; handlers close over stable setters.
  }, []);

  const statusClass =
    status.isError && status.text
      ? "text-xs font-mono mt-3 text-error border border-error/30 bg-error/10 rounded-field px-3 py-2"
      : "text-xs text-base-content/60 font-mono mt-3";

  return (
    <>
      <header className="print:hidden border-b border-base-300 bg-base-100/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer"
            role="button"
            tabIndex={0}
            aria-label="FinThesis home - reload the app"
            onClick={goHome}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                goHome();
              }
            }}
          >
            <div className="w-7 h-7 rounded bg-primary flex items-center justify-center">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <path
                  d="M2 11L5.5 6.5L8 9L11 4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-primary-content"
                />
              </svg>
            </div>
            <span className="font-semibold tracking-tight text-sm">FinThesis</span>
            <span className="hidden sm:block text-xs text-base-content/60 border-l border-base-300 pl-3">
              Fintech Market Research
            </span>
          </div>

          <div className="flex items-center gap-2">
            <a
              href="https://finthesis-docs.onrender.com"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost btn-xs font-mono"
            >
              Docs
            </a>
            <div className="flex items-center gap-3 text-xs">
              {auth.email && (
                <span className="text-base-content/60 font-mono hidden sm:block">{auth.email}</span>
              )}
              <button type="button" className="btn btn-ghost btn-xs" onClick={() => auth.onSignOut()}>
                Sign out
              </button>
            </div>
          </div>
        </div>
      </header>

      <section className="print:hidden max-w-5xl mx-auto px-6 pt-12 pb-8">
        <div className="mb-8">
          <p className="text-xs font-mono text-primary uppercase tracking-widest mb-2">
            AI Research Assistant
          </p>
          <h1 className="text-2xl font-semibold leading-snug">
            What fintech market do you want to analyze?
          </h1>
          <p className="text-sm text-base-content/60 mt-1 leading-relaxed">
            Enter a topic or question - we'll research recent articles and return a scored
            investment thesis.
          </p>
        </div>

        <div className="flex gap-3">
          <input
            type="text"
            className="w-full bg-base-200 border border-base-300 rounded-field px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/60"
            placeholder="e.g., What's the outlook for cross-border payments infrastructure companies?"
            aria-label="Market topic or question"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void generate();
            }}
          />
          <button
            type="button"
            className="btn btn-primary disabled:pointer-events-auto disabled:cursor-not-allowed disabled:bg-primary! disabled:text-primary-content! disabled:border-primary! disabled:opacity-40!"
            disabled={query.trim().length === 0 || generating}
            onClick={() => void generate()}
          >
            Generate Thesis
          </button>
        </div>

        <div className="mt-4">
          {resumeError ? (
            <small className="text-base-content/60">Could not load previous sessions.</small>
          ) : (
            resumeJobs.length > 0 && <ResumePicker jobs={resumeJobs} onResume={onResume} />
          )}
        </div>

        <p className={statusClass}>{status.text}</p>
      </section>

      <section className="max-w-5xl mx-auto px-6 pb-16 space-y-4">
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

      <section className="print:hidden max-w-5xl mx-auto px-6 pb-16 -mt-8">
        <PastThesesList
          jobs={pastPage.filter((j) => j.job_id !== currentJob?.job_id)}
          onPrevPage={() => pagePast(-1)}
          onNextPage={() => pagePast(1)}
          canPrevPage={pastOffset > 0}
          canNextPage={pastHasMore}
        />
      </section>

      {auth.isAdmin && (
        <section className="print:hidden max-w-5xl mx-auto px-6 pb-16 -mt-8">
          <PastThesesList
            jobs={allPage.filter((j) => j.job_id !== currentJob?.job_id)}
            onPrevPage={() => pageAll(-1)}
            onNextPage={() => pageAll(1)}
            canPrevPage={allOffset > 0}
            canNextPage={allHasMore}
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
