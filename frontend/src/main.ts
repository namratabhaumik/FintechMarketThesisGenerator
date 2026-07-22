// Entry point + auth gate. Shows the Google login until there's a session, then
// mounts the app. A single onAuthChange subscription drives both the initial
// render and later sign-in/sign-out (including the OAuth redirect back).

import { FinThesisApp } from "./app";
import type { Session } from "@supabase/supabase-js";
import { getSession, onAuthChange, signInWithGoogle, signOut } from "./auth";
import { el } from "./dom";

const root = document.querySelector<HTMLElement>("#app");

function renderLogin(container: HTMLElement): void {
  const wrap = el(
    "div",
    undefined,
    "min-h-screen flex items-center justify-center px-6",
  );
  const card = el(
    "div",
    undefined,
    "w-full max-w-sm bg-base-200 border border-base-300 rounded-box p-8 text-center space-y-5",
  );
  card.append(
    el("h1", "FinThesis", "text-xl font-semibold tracking-tight"),
    el(
      "p",
      "Sign in to generate and track your investment theses.",
      "text-sm text-base-content/60",
    ),
  );
  const button = el("button", "Continue with Google", "btn btn-primary w-full");
  button.addEventListener("click", () => void signInWithGoogle());
  card.append(button);

  const docsLink = el("a", "Read the docs", "link link-hover text-xs text-base-content/60");
  docsLink.href = "https://finthesis-docs.onrender.com";
  docsLink.target = "_blank";
  docsLink.rel = "noopener noreferrer";
  card.append(docsLink);

  wrap.append(card);
  container.replaceChildren(wrap);
}

// Re-render only on actual signed-in <-> signed-out transitions, so token
// refreshes (which also fire onAuthChange) don't remount the app.
let view: "app" | "login" | null = null;

function render(session: Session | null): void {
  if (!root) return;
  const next = session ? "app" : "login";
  if (next === view) return;
  view = next;
  try {
    if (session) {
      FinThesisApp.mount("#app", {
        email: session.user.email,
        isAdmin: session.user.app_metadata?.role === "admin",
        onSignOut: () => void signOut(),
      });
    } else {
      renderLogin(root);
    }
  } catch (err) {
    console.error("Failed to render the FinThesis UI", err);
  }
}

onAuthChange((session) => render(session));

// Bfcache restores the page (and its old DOM) without re-running onAuthChange,
// so back/forward navigation can show a stale signed-in/out view. Re-check the
// real session whenever the page is restored from bfcache.
window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    void getSession().then((session) => render(session));
  }
});
