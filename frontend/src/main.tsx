// Entry point + auth gate. Shows the Google login until there's a session, then
// mounts the app. A single onAuthChange subscription drives both the initial
// render and later sign-in/sign-out (including the OAuth redirect back).

import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { Session } from "@supabase/supabase-js";
import { getSession, onAuthChange, signInWithGoogle, signOut } from "./auth";
import { App } from "./components/App";
import "./styles.css";

function Login() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="w-full max-w-sm bg-base-200 border border-base-300 rounded-box p-8 text-center space-y-5">
        <h1 className="text-xl font-semibold tracking-tight">FinThesis</h1>
        <p className="text-sm text-base-content/60">
          Sign in to generate and track your investment theses.
        </p>
        <button className="btn btn-primary w-full" onClick={() => void signInWithGoogle()}>
          Continue with Google
        </button>
        <a
          href="https://finthesis-docs.onrender.com"
          target="_blank"
          rel="noopener noreferrer"
          className="link link-hover text-xs text-base-content/60"
        >
          Read the docs
        </a>
      </div>
    </div>
  );
}

function AuthGate() {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const subscribed = useRef(false);

  useEffect(() => {
    // StrictMode double-invokes effects in dev; onAuthChange can't unsubscribe,
    // so guard against a second subscription.
    if (subscribed.current) return;
    subscribed.current = true;

    onAuthChange((s) => {
      setSession(s);
      setReady(true);
    });
    // Resolve the initial session explicitly in case the subscription's first
    // fire is delayed.
    void getSession().then((s) => {
      setSession(s);
      setReady(true);
    });

    // Bfcache restores the page (and its old DOM) without re-running
    // onAuthChange, so back/forward can show a stale view. Re-check on restore.
    const onPageShow = (e: PageTransitionEvent) => {
      if (e.persisted) void getSession().then(setSession);
    };
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  }, []);

  if (!ready) return null;
  if (!session) return <Login />;
  return (
    <App
      auth={{
        email: session.user.email,
        isAdmin: session.user.app_metadata?.role === "admin",
        onSignOut: () => {
          // Drop the ?job_id deep-link so sign-out lands on a clean URL (a
          // fresh visit with a shared ?job_id still keeps it through login).
          history.replaceState(null, "", window.location.pathname);
          void signOut();
        },
      }}
    />
  );
}

const root = document.querySelector<HTMLElement>("#app");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <AuthGate />
    </StrictMode>,
  );
}
