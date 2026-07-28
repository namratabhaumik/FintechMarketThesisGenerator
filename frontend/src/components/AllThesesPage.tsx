import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { ApiError, ErrorCode, deleteThesis } from "../api";
import type { AuthInfo } from "../types";
import { usePaginatedTheses } from "../usePaginatedTheses";
import { CommandPalette } from "./CommandPalette";
import { PastThesesList } from "./PastThesesList";

// The full library, as its own page. Search here covers every thesis the caller
// may see (own +, for admins, everyone else's), unlike the global ⌘K which is
// always own-only - so this page owns a second, all-scope search overlay.
export function AllThesesPage({ auth }: { auth: AuthInfo }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const past = usePaginatedTheses({ errorLabel: "Failed to load past theses" });
  const all = usePaginatedTheses({
    allUsers: true,
    enabled: Boolean(auth.isAdmin),
    errorLabel: "Failed to load all theses (admin)",
  });

  useEffect(() => {
    void past.load();
    void all.load();
    // Mount-only load; the hooks close over stable setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onDeleteThesis = (jobId: string) => {
    void (async () => {
      try {
        await deleteThesis(jobId);
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
      // Deletion can affect either list, so refresh both.
      void all.load();
      void past.load();
    })();
  };

  return (
    <section className="px-6 md:px-8 pt-8 pb-16 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold">All Theses</h1>
        <button
          type="button"
          onClick={() => setSearchOpen(true)}
          className="flex items-center gap-2 h-8 px-2.5 rounded-field border border-base-300 bg-base-200 text-base-content/50 hover:border-base-content/30 hover:text-base-content/70 transition-colors cursor-pointer sm:w-56"
        >
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
            className="flex-shrink-0"
          >
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
            <path d="M21 21l-4.3-4.3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <span className="text-xs">Search all theses...</span>
        </button>
      </div>

      {error && (
        <p className="text-xs font-mono text-error border border-error/30 bg-error/10 rounded-field px-3 py-2">
          {error}
        </p>
      )}

      <PastThesesList
        jobs={past.page}
        onPrevPage={() => past.goToPage(-1)}
        onNextPage={() => past.goToPage(1)}
        canPrevPage={past.offset > 0}
        canNextPage={past.hasMore}
      />

      {auth.isAdmin && (
        <PastThesesList
          jobs={all.page}
          onPrevPage={() => all.goToPage(-1)}
          onNextPage={() => all.goToPage(1)}
          canPrevPage={all.offset > 0}
          canNextPage={all.hasMore}
          isAdmin
          onDelete={onDeleteThesis}
          title="Admin - all users' theses"
        />
      )}

      <CommandPalette
        open={searchOpen}
        onOpenChange={setSearchOpen}
        isAdmin={Boolean(auth.isAdmin)}
        scope="all"
        onOpenThesis={(jobId) => void navigate(`/thesis/${encodeURIComponent(jobId)}`)}
      />
    </section>
  );
}