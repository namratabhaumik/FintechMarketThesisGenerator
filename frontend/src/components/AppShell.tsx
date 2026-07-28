import { useEffect, useState } from "react";
import { Outlet, useNavigate } from "react-router";
import type { AuthInfo } from "../types";
import { AppHeader } from "./AppHeader";
import { CommandPalette } from "./CommandPalette";
import { Sidebar } from "./Sidebar";

// Persistent chrome around every page: header, sidebar and the global search
// overlay, with the routed page rendered into the middle column.
//
// The middle column is `flex-1 min-w-0` so it fills whatever the side panels
// leave - that is what lets the document pane run full width today and reflow
// unchanged when the annotations panel is added beside it.
// Remembers whether the desktop rail is collapsed, so it survives reloads.
const RAIL_KEY = "finthesis.sidebar.open";

export function AppShell({ auth }: { auth: AuthInfo }) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [railOpen, setRailOpen] = useState(() => {
    try {
      return localStorage.getItem(RAIL_KEY) !== "false";
    } catch {
      // Storage can throw (private mode, blocked cookies); default to open.
      return true;
    }
  });
  const navigate = useNavigate();

  // One trigger for both layouts: below md it opens the drawer, from md up it
  // collapses/expands the rail.
  const toggleNav = () => {
    if (window.matchMedia("(min-width: 768px)").matches) {
      setRailOpen((open) => {
        try {
          localStorage.setItem(RAIL_KEY, String(!open));
        } catch {
          // Persistence is a convenience; ignore storage failures.
        }
        return !open;
      });
    } else {
      setNavOpen(true);
    }
  };

  // Global ⌘K / Ctrl-K toggles search. preventDefault stops Chrome's Ctrl-K
  // omnibox-search default on Windows/Linux.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <AppHeader
        auth={auth}
        onOpenSearch={() => setPaletteOpen(true)}
        onToggleNav={toggleNav}
      />

      <div className="flex flex-1 min-h-0">
        <Sidebar
          isAdmin={Boolean(auth.isAdmin)}
          railOpen={railOpen}
          mobileOpen={navOpen}
          onCloseMobile={() => setNavOpen(false)}
        />
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>

      {/* Global search stays scoped to the caller's own theses; the All Theses
          page opens its own all-scope search. */}
      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        isAdmin={Boolean(auth.isAdmin)}
        onOpenThesis={(jobId) => void navigate(`/thesis/${encodeURIComponent(jobId)}`)}
      />
    </div>
  );
}
