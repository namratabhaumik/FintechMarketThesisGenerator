import type { AuthInfo } from "../types";

// Clicking the logo/name reloads to a clean root.
function goHome() {
  window.location.href = window.location.pathname;
}

// Sticky page chrome: logo/home control, docs link, signed-in identity and
// sign-out. Presentational - the only behaviour it owns is goHome.
export function AppHeader({ auth }: { auth: AuthInfo }) {
  return (
    <header className="print:hidden border-b border-base-300 bg-base-100/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="px-6 md:px-8 h-14 flex items-center justify-between">
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
  );
}
