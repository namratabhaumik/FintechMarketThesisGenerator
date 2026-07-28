import { Link } from "react-router";
import type { AuthInfo } from "../types";

// Platform-correct shortcut label (⌘ on Apple, Ctrl elsewhere), computed once.
const shortcutHint =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent)
    ? "⌘K"
    : "Ctrl K";

// Sticky page chrome: nav trigger, logo/home link, search, docs, signed-in
// identity and sign-out. Presentational - search and nav are delegated up.
export function AppHeader({
  auth,
  onOpenSearch,
  onToggleNav,
}: {
  auth: AuthInfo;
  onOpenSearch: () => void;
  onToggleNav: () => void;
}) {
  return (
    <header className="print:hidden border-b border-base-300 bg-base-100/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="px-6 md:px-8 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* One control for both layouts: opens the drawer below md, collapses
              or expands the rail from md up. */}
          <button
            type="button"
            onClick={onToggleNav}
            aria-label="Toggle navigation"
            className="btn btn-ghost btn-xs px-1"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M4 6h16M4 12h16M4 18h16"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>

          <Link to="/" className="flex items-center gap-3" aria-label="FinThesis home">
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
          </Link>
        </div>

        <div className="flex items-center gap-2">
          {/* Looks like a search field but is a button: clicking (or ⌘K)
              expands the real search overlay, so there's no second input to
              keep in sync. Icon-only on the smallest screens. */}
          <button
            type="button"
            onClick={onOpenSearch}
            aria-label="Search theses"
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
            <span className="hidden sm:inline text-xs">Search theses...</span>
            <kbd className="hidden sm:inline ml-auto text-[10px] font-mono border border-base-300 rounded px-1 py-0.5">
              {shortcutHint}
            </kbd>
          </button>
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
  );
}
