import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router";
import { listTheses } from "../api";
import type { ThesisSummaryResponse } from "../types";

// How many theses each sidebar group shows.
const SIDEBAR_COUNT = 5;

function dotColor(rec?: string | null): string {
  if (rec === "Pursue") return "bg-primary";
  if (rec === "Skip") return "bg-error";
  return "bg-accent";
}

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2.5 px-3 py-2 rounded-field text-sm transition-colors ${
    isActive ? "bg-base-300 text-base-content" : "text-base-content/70 hover:bg-base-300/50"
  }`;

function ThesisLinks({ jobs }: { jobs: ThesisSummaryResponse[] }) {
  return (
    <>
      {jobs.map((j) => (
        <NavLink
          key={j.job_id}
          to={`/thesis/${encodeURIComponent(j.job_id)}`}
          className={({ isActive }) =>
            `flex items-start gap-2.5 px-3 py-1.5 rounded-field text-xs transition-colors ${
              isActive ? "bg-base-300 text-base-content" : "text-base-content/70 hover:bg-base-300/50"
            }`
          }
          title={j.query}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full flex-shrink-0 mt-1.5 ${dotColor(j.recommendation)}`}
          />
          <span className="line-clamp-2 leading-snug">{j.query}</span>
        </NavLink>
      ))}
    </>
  );
}

function GroupHeading({ children }: { children: React.ReactNode }) {
  return (
    <p className="px-3 pt-5 pb-1.5 text-[10px] font-mono uppercase tracking-wider text-base-content/50">
      {children}
    </p>
  );
}

// Sidebar contents: navigation plus the most recent theses
function SidebarNav({ isAdmin, onNavigate }: { isAdmin: boolean; onNavigate?: () => void }) {
  const [mine, setMine] = useState<ThesisSummaryResponse[]>([]);
  const [admin, setAdmin] = useState<ThesisSummaryResponse[]>([]);
  const [loadFailed, setLoadFailed] = useState(false);
  const location = useLocation();

  // Refetch on navigation: generating, approving or opening a thesis all change
  // the route, so this keeps the list current without a shared store.
  useEffect(() => {
    let cancelled = false;
    const requests: Promise<ThesisSummaryResponse[]>[] = [listTheses(SIDEBAR_COUNT, 0)];
    if (isAdmin) requests.push(listTheses(SIDEBAR_COUNT, 0, undefined, true));
    Promise.all(requests)
      .then(([own, all]) => {
        if (cancelled) return;
        setMine(own);
        setAdmin(all ?? []);
        setLoadFailed(false);
      })
      .catch((err) => {
        console.error("Failed to load sidebar theses", err);
        // Distinguished from an empty library below: reporting "No theses yet"
        // to someone who has theses is worse than saying nothing loaded.
        if (!cancelled) setLoadFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [isAdmin, location.pathname]);

  return (
    <nav className="p-3" onClick={onNavigate}>
      <NavLink to="/" end className={navLinkClass}>
        Recent
      </NavLink>
      <NavLink to="/theses" className={navLinkClass}>
        All Theses
      </NavLink>

      <GroupHeading>My Theses</GroupHeading>
      {loadFailed ? (
        <p className="px-3 py-1.5 text-xs text-error">Could not load theses.</p>
      ) : mine.length === 0 ? (
        <p className="px-3 py-1.5 text-xs text-base-content/40">No theses yet.</p>
      ) : (
        <ThesisLinks jobs={mine} />
      )}

      {isAdmin && admin.length > 0 && (
        <>
          <GroupHeading>Admin</GroupHeading>
          <ThesisLinks jobs={admin} />
        </>
      )}
    </nav>
  );
}

// Responsive shell for the nav: a collapsible rail from md up, and a slide-in
// drawer below that (no hover dependency - the trigger is a tap target in the
// header). `railOpen` only governs the desktop rail; the drawer has its own
// open state so collapsing on desktop never traps the mobile nav shut.
export function Sidebar({
  isAdmin,
  railOpen,
  mobileOpen,
  onCloseMobile,
}: {
  isAdmin: boolean;
  railOpen: boolean;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}) {
  return (
    <>
      <aside
        className={`print:hidden hidden flex-shrink-0 border-r border-base-300 overflow-y-auto ${
          railOpen ? "md:block w-64" : "w-0"
        }`}
      >
        <SidebarNav isAdmin={isAdmin} />
      </aside>

      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40 print:hidden">
          <button
            type="button"
            aria-label="Close navigation"
            className="absolute inset-0 bg-black/50 w-full"
            onClick={onCloseMobile}
          />
          <aside className="absolute left-0 top-0 bottom-0 w-64 bg-base-100 border-r border-base-300 overflow-y-auto">
            <SidebarNav isAdmin={isAdmin} onNavigate={onCloseMobile} />
          </aside>
        </div>
      )}
    </>
  );
}