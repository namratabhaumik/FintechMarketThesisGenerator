// React entry point. Step 1 of the React migration: a minimal shell that
// verifies the Vite + Tailwind/daisyUI + env-injection pipeline end to end.
// The real auth gate and app (ported from app.ts / render.ts) land in later
// steps, replacing <Shell/>.

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

function Shell() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="w-full max-w-sm bg-base-200 border border-base-300 rounded-box p-8 text-center space-y-3">
        <h1 className="text-xl font-semibold tracking-tight">FinThesis</h1>
        <p className="text-sm text-base-content/60">
          React + Vite shell - migration in progress.
        </p>
        <p className="text-xs font-mono text-primary">
          API: {import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}
        </p>
      </div>
    </div>
  );
}

const root = document.querySelector<HTMLElement>("#app");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <Shell />
    </StrictMode>,
  );
}
