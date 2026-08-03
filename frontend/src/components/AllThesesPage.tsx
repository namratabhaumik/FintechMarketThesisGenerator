import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { ApiError, ErrorCode, deleteThesis, listTheses } from "../api";
import { fmtDate } from "../format";
import type { AuthInfo, ThesisSummaryResponse } from "../types";
import { RecommendationBadge } from "./RecommendationBadge";

// The library is loaded once and sorted/filtered/paged in the browser, because
// /theses offers no total count or sort parameter.
//
// The endpoint caps limit at 100, so a bigger library is walked in pages rather
// than truncated - a short page means the end. MAX_PAGES bounds the walk so a
// runaway can't spin forever; hitting it is the signal to move this table's
// sorting/filtering/counting server-side.
const REQUEST_LIMIT = 100;
const MAX_PAGES = 10;
const ROWS_PER_PAGE = 10;

type Scope = "all" | "mine" | "admin";
type SortKey = "recent" | "score" | "title";

// A row plus whether it belongs to the caller (the admin list excludes their
// own rows, so ownership is known from which request returned it).
interface Row {
  job: ThesisSummaryResponse;
  isOwn: boolean;
}

const SORTS: { key: SortKey; label: string }[] = [
  { key: "recent", label: "Recently created" },
  { key: "score", label: "Highest score" },
  { key: "title", label: "Title (A-Z)" },
];

function chipClass(active: boolean): string {
  return `px-3 py-1 rounded-full text-xs border transition-colors cursor-pointer ${
    active
      ? "bg-primary/15 text-primary border-primary/40"
      : "border-base-300 text-base-content/60 hover:text-base-content hover:border-base-content/30"
  }`;
}

export function AllThesesPage({ auth }: { auth: AuthInfo }) {
  const isAdmin = Boolean(auth.isAdmin);

  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [error, setError] = useState("");

  const [scope, setScope] = useState<Scope>("all");
  const [sort, setSort] = useState<SortKey>("recent");
  const [page, setPage] = useState(0);

  // Walk pages until a short one comes back (or the bound is hit), so the count
  // and filters cover the whole library instead of just its first page.
  const fetchAll = async (allUsers: boolean): Promise<ThesisSummaryResponse[]> => {
    const out: ThesisSummaryResponse[] = [];
    for (let i = 0; i < MAX_PAGES; i++) {
      const batch = await listTheses(REQUEST_LIMIT, i * REQUEST_LIMIT, undefined, allUsers);
      out.push(...batch);
      if (batch.length < REQUEST_LIMIT) break;
    }
    return out;
  };

  const load = async () => {
    setLoading(true);
    try {
      const own = await fetchAll(false);
      const others = isAdmin ? await fetchAll(true) : [];
      setRows([
        ...own.map((job) => ({ job, isOwn: true })),
        ...others.map((job) => ({ job, isOwn: false })),
      ]);
      setLoadFailed(false);
    } catch (err) {
      console.error("Failed to load theses", err);
      // Distinguished from an empty library: an error must not read as "you
      // have no theses".
      setLoadFailed(true);
      setRows([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    // Mount-only; isAdmin is fixed for a session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Filtering and sorting run over the whole library, so paging walks the real
  // result set rather than re-ordering one page at a time.
  const filtered = useMemo(() => {
    let out = rows;
    if (scope === "mine") out = out.filter((r) => r.isOwn);
    else if (scope === "admin") out = out.filter((r) => !r.isOwn);

    const sorted = [...out];
    if (sort === "score") {
      sorted.sort((a, b) => (b.job.opportunity_score ?? -1) - (a.job.opportunity_score ?? -1));
    } else if (sort === "title") {
      sorted.sort((a, b) => a.job.query.localeCompare(b.job.query));
    } else {
      sorted.sort((a, b) => (b.job.created_at ?? "").localeCompare(a.job.created_at ?? ""));
    }
    return sorted;
  }, [rows, scope, sort]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / ROWS_PER_PAGE));
  // Clamp rather than store a page that filtering may have removed.
  const current = Math.min(page, pageCount - 1);
  const start = current * ROWS_PER_PAGE;
  const visible = filtered.slice(start, start + ROWS_PER_PAGE);

  const setFilter = <T,>(setter: (v: T) => void) => (value: T) => {
    setter(value);
    setPage(0);
  };

  const onDelete = (job: ThesisSummaryResponse) => {
    if (!window.confirm(`Delete this thesis permanently?\n\n"${job.query}"`)) return;
    void (async () => {
      try {
        await deleteThesis(job.job_id);
      } catch (err) {
        if (err instanceof ApiError && err.code === ErrorCode.SessionExpired) {
          setError("Your session has expired. Please sign out and sign in again.");
        } else if (err instanceof ApiError) {
          setError(`Delete failed: ${err.message}`);
        } else {
          console.error("Could not delete thesis.", err);
          setError("Could not delete thesis.");
        }
        return;
      }
      setError("");
      await load();
    })();
  };

  return (
    <section className="px-6 md:px-8 pt-8 pb-16">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">All Theses</h1>
        <p className="text-sm text-base-content/60 mt-0.5">
          {loading ? "Loading..." : `${filtered.length} ${filtered.length === 1 ? "thesis" : "theses"}`}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2 mb-4">
        <select
          value={sort}
          onChange={(e) => setFilter(setSort)(e.target.value as SortKey)}
          aria-label="Sort theses"
          className="select select-sm bg-base-200 border-base-300 text-xs"
        >
          {SORTS.map((s) => (
            <option key={s.key} value={s.key}>
              Sort: {s.label}
            </option>
          ))}
        </select>

        {isAdmin && (
          <>
            <button type="button" className={chipClass(scope === "all")} onClick={() => setFilter(setScope)("all")}>
              All
            </button>
            <button type="button" className={chipClass(scope === "mine")} onClick={() => setFilter(setScope)("mine")}>
              My Theses
            </button>
            <button type="button" className={chipClass(scope === "admin")} onClick={() => setFilter(setScope)("admin")}>
              Admin
            </button>
          </>
        )}
      </div>

      {error && (
        <p className="mb-3 text-xs font-mono text-error border border-error/30 bg-error/10 rounded-field px-3 py-2">
          {error}
        </p>
      )}

      {loadFailed ? (
        <p className="text-sm text-error border border-error/30 bg-error/10 rounded-field px-4 py-3">
          Could not load theses. Please reload the page.
        </p>
      ) : (
        <>
          {/* Wide table scrolls inside its own container so the page never
              scrolls sideways on a phone. */}
          <div className="overflow-x-auto border border-base-300 rounded-box">
            <table className="table table-sm">
              <thead>
                <tr className="text-xs">
                  <th>Title</th>
                  <th className="w-20">Score</th>
                  <th className="w-32">Recommendation</th>
                  <th className="w-28">Created</th>
                  {isAdmin && <th className="w-12" />}
                </tr>
              </thead>
              <tbody>
                {visible.length === 0 && (
                  <tr>
                    <td colSpan={isAdmin ? 5 : 4} className="text-center text-xs text-base-content/50 py-8">
                      {loading ? "Loading theses..." : "No theses match these filters."}
                    </td>
                  </tr>
                )}
                {visible.map(({ job, isOwn }) => (
                  <tr key={job.job_id} className="hover:bg-base-300/40">
                    <td>
                      <Link
                        to={`/thesis/${encodeURIComponent(job.job_id)}`}
                        className="text-primary hover:text-primary/80 text-sm"
                      >
                        {job.query}
                      </Link>
                      {!isOwn && job.user_id && (
                        <span className="block text-[10px] text-base-content/50 font-mono mt-0.5">
                          owner {job.user_id.slice(0, 8)}
                        </span>
                      )}
                    </td>
                    <td className="font-mono text-sm">
                      {job.opportunity_score != null ? `${job.opportunity_score}/5` : "-"}
                    </td>
                    <td>
                      {job.recommendation ? (
                        <RecommendationBadge recommendation={job.recommendation} />
                      ) : (
                        <span className="text-base-content/40 text-xs">-</span>
                      )}
                    </td>
                    <td className="text-xs text-base-content/60 font-mono whitespace-nowrap">
                      {job.created_at ? fmtDate(job.created_at) : "-"}
                    </td>
                    {isAdmin && (
                      <td>
                        {!isOwn && (
                          <button
                            type="button"
                            className="btn btn-ghost btn-xs text-error/70 hover:text-error"
                            aria-label={`Delete thesis: ${job.query}`}
                            title="Delete (admin)"
                            onClick={() => onDelete(job)}
                          >
                            <svg
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              aria-hidden="true"
                            >
                              <polyline points="3 6 5 6 21 6" />
                              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                              <path d="M10 11v6" />
                              <path d="M14 11v6" />
                              <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
                            </svg>
                          </button>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filtered.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 mt-4">
              <p className="text-xs text-base-content/60 font-mono">
                {start + 1}-{Math.min(start + ROWS_PER_PAGE, filtered.length)} of {filtered.length}
              </p>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  className="btn btn-xs btn-outline"
                  disabled={current === 0}
                  onClick={() => setPage(current - 1)}
                >
                  Prev
                </button>
                {Array.from({ length: pageCount }, (_, i) => (
                  <button
                    key={i}
                    type="button"
                    aria-current={i === current ? "page" : undefined}
                    className={`btn btn-xs ${i === current ? "btn-primary" : "btn-ghost"}`}
                    onClick={() => setPage(i)}
                  >
                    {i + 1}
                  </button>
                ))}
                <button
                  type="button"
                  className="btn btn-xs btn-outline"
                  disabled={current >= pageCount - 1}
                  onClick={() => setPage(current + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}

    </section>
  );
}
