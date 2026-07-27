import { Command, defaultFilter } from "cmdk";
import { useEffect, useMemo, useState } from "react";
import { listTheses } from "../api";
import type { ThesisSummaryResponse } from "../types";

// Results shown for a typed query. Ranking is best-match (no backend search endpoint).
const MAX_RESULTS = 10;

// How many theses to load for client-side search.
const FETCH_SIZE = 50;

// Dot color by recommendation, matching the metrics/badge palette.
function dotColor(rec?: string | null): string {
  if (rec === "Pursue") return "bg-primary";
  if (rec === "Skip") return "bg-error";
  return "bg-accent";
}

// Text cmdk's scorer matches against title + id.
const searchableText = (job: ThesisSummaryResponse) => `${job.query} ${job.job_id}`;

function PaletteItem({
  job,
  onSelect,
}: {
  job: ThesisSummaryResponse;
  onSelect: (jobId: string) => void;
}) {
  return (
    <Command.Item
      value={searchableText(job)}
      onSelect={() => onSelect(job.job_id)}
      className="flex items-center gap-2.5 px-3 py-2 rounded-field text-sm cursor-pointer data-[selected=true]:bg-base-300"
    >
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotColor(job.recommendation)}`} />
      <span className="truncate">{job.query}</span>
    </Command.Item>
  );
}

// Thesis search overlay (cmdk over Radix Dialog: focus-trap, esc, backdrop and
// a11y for free, cross-browser). Opened by ⌘K/Ctrl-K or by clicking the header
// search field. Pure search (nothing listed until something typed).
//
// `scope` decides whose theses are searchable: "mine" (the caller's own - the
// default, used by the global palette) or "all" (own + every other user's, used
// on the All Theses page). "all" only requests the cross-user list for admins;
// the endpoint 403s that for everyone else.
export function CommandPalette({
  open,
  onOpenChange,
  isAdmin,
  scope = "mine",
  onOpenThesis,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  isAdmin: boolean;
  scope?: "mine" | "all";
  onOpenThesis: (jobId: string) => void;
}) {
  const includeOthers = scope === "all" && isAdmin;
  const [mine, setMine] = useState<ThesisSummaryResponse[]>([]);
  const [admin, setAdmin] = useState<ThesisSummaryResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const [search, setSearch] = useState("");

  // (Re)load whenever the palette opens, so newly created theses are searchable.
  // Nothing renders until the user types, so this is invisible.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setLoadFailed(false);
    const requests: Promise<ThesisSummaryResponse[]>[] = [listTheses(FETCH_SIZE, 0)];
    if (includeOthers) requests.push(listTheses(FETCH_SIZE, 0, undefined, true));
    Promise.all(requests)
      .then(([own, all]) => {
        if (cancelled) return;
        setMine(own);
        setAdmin(all ?? []);
      })
      .catch((err) => {
        console.error("Failed to load theses for search", err);
        // Surfaced in the list: with nothing loaded.
        if (!cancelled) setLoadFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, includeOthers]);

  // Clear the query on close, so it never reopens mid-search.
  useEffect(() => {
    if (!open) setSearch("");
  }, [open]);

  const query = search.trim();

  // Rank BY MATCH QUALITY across both groups, then take the global top N -
  // capping per group (or by recency) would drop better matches. Uses cmdk's
  // own scorer, so cmdk's filtering is turned off to avoid ranking twice.
  const { topMine, topAdmin } = useMemo(() => {
    if (!query) return { topMine: [], topAdmin: [] };
    const scored = [
      ...mine.map((job) => ({ job, isAdminRow: false })),
      ...admin.map((job) => ({ job, isAdminRow: true })),
    ]
      .map((entry) => ({ ...entry, score: defaultFilter(searchableText(entry.job), query) }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_RESULTS);
    return {
      topMine: scored.filter((e) => !e.isAdminRow).map((e) => e.job),
      topAdmin: scored.filter((e) => e.isAdminRow).map((e) => e.job),
    };
  }, [query, mine, admin]);

  const select = (jobId: string) => {
    onOpenChange(false);
    onOpenThesis(jobId);
  };

  // A failed load takes precedence over "no matches": with nothing to search,
  // an empty result would misreport the failure as a successful empty search.
  const noMatches =
    !loadFailed && query.length > 0 && topMine.length === 0 && topAdmin.length === 0;

  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Search theses"
      shouldFilter={false}
      overlayClassName="fixed inset-0 bg-black/50 z-[100]"
      contentClassName="fixed left-1/2 top-[15%] -translate-x-1/2 w-[calc(100%-2rem)] max-w-xl z-[100] bg-base-200 border border-base-300 rounded-box shadow-xl overflow-hidden"
    >
      <Command.Input
        value={search}
        onValueChange={setSearch}
        placeholder="Search theses..."
        className="w-full bg-transparent border-b border-base-300 px-4 py-3 text-sm focus:outline-none placeholder:text-base-content/40"
      />
      <Command.List className="max-h-80 overflow-y-auto p-2 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-base-content/50">
        {loadFailed && (
          <p className="px-3 py-6 text-center text-xs text-error">
            Could not load theses to search. Close and try again.
          </p>
        )}
        {!loadFailed && !query && (
          <p className="px-3 py-6 text-center text-xs text-base-content/50">
            {loading ? "Loading theses…" : "Type to search your theses."}
          </p>
        )}
        {noMatches && (
          <p className="px-3 py-6 text-center text-xs text-base-content/50">No theses found.</p>
        )}

        {topMine.length > 0 && (
          <Command.Group heading="My Theses">
            {topMine.map((job) => (
              <PaletteItem key={job.job_id} job={job} onSelect={select} />
            ))}
          </Command.Group>
        )}

        {topAdmin.length > 0 && (
          <Command.Group heading="Admin">
            {topAdmin.map((job) => (
              <PaletteItem key={`admin-${job.job_id}`} job={job} onSelect={select} />
            ))}
          </Command.Group>
        )}
      </Command.List>

      <div className="flex items-center gap-4 border-t border-base-300 px-4 py-2 text-[10px] font-mono text-base-content/50">
        <span>↑↓ navigate</span>
        <span>↵ open</span>
        <span>esc close</span>
      </div>
    </Command.Dialog>
  );
}