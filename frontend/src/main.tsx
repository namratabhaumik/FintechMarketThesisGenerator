// Entry point + auth gate. Shows the Google login until there's a session, then
// mounts the app. A single onAuthChange subscription drives both the initial
// render and later sign-in/sign-out (including the OAuth redirect back).

import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import type { Session } from "@supabase/supabase-js";
import { getSession, onAuthChange, signInWithGoogle, signOut } from "./auth";
import { App } from "./components/App";
import "./styles.css";

function Login() {
  const [error, setError] = useState("");

  // Without a catch a failed OAuth start is a silent no-op, so the button looks
  // dead and the user just clicks it again.
  const signIn = () => {
    setError("");
    void signInWithGoogle().catch((err) => {
      console.error("Could not start Google sign-in", err);
      setError("Could not start sign-in. Please try again.");
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="w-full max-w-sm bg-base-200 border border-base-300 rounded-box p-8 text-center space-y-5">
        <h1 className="text-xl font-semibold tracking-tight">FinThesis</h1>
        <p className="text-sm text-base-content/60">
          Sign in to generate and track your investment theses.
        </p>
        <button className="btn btn-primary w-full" onClick={signIn}>
          Continue with Google
        </button>
        {error && (
          <p className="text-xs font-mono text-error border border-error/30 bg-error/10 rounded-field px-3 py-2">
            {error}
          </p>
        )}
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
    // fire is delayed. On failure still mark ready: `ready` gates all rendering,
    // so leaving it false would strand the user on a blank page with no error.
    // Falling through with a null session shows the login screen instead.
    void getSession()
      .then((s) => {
        setSession(s);
        setReady(true);
      })
      .catch((err) => {
        console.error("Could not resolve the initial session", err);
        setReady(true);
      });

    // Bfcache restores the page (and its old DOM) without re-running
    // onAuthChange, so back/forward can show a stale view. Re-check on restore.
    const onPageShow = (e: PageTransitionEvent) => {
      if (e.persisted) {
        void getSession()
          .then(setSession)
          .catch((err) => console.error("Could not re-check the session on restore", err));
      }
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
        userId: session.user.id,
        isAdmin: session.user.app_metadata?.role === "admin",
        onSignOut: () => {
          // Drop any thesis deep-link so sign-out lands on a clean URL (a fresh
          // visit with a shared link still keeps it through login). Only on
          // success: clearing it after a failed sign-out would strip the deep
          // link while leaving the user signed in.
          void signOut()
            .then(() => history.replaceState(null, "", "/"))
            .catch((err) => console.error("Sign-out failed", err));
        },
      }}
    />
  );
}

const root = document.querySelector<HTMLElement>("#app");
if (root) {
  createRoot(root).render(
    <StrictMode>
      {/* Real paths (not hash routing), so the deployed static site needs a
          rewrite of /* -> /index.html or a hard load of /theses 404s. */}
      <BrowserRouter>
        <AuthGate />
      </BrowserRouter>
    </StrictMode>,
  );
}
