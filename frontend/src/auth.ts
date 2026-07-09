// Supabase auth: Google OAuth login + session/JWT access. The client persists
// the session in localStorage and, on the OAuth redirect back, reads the tokens
// from the URL and fires onAuthStateChange (both default-on).

import { createClient, type Session } from "@supabase/supabase-js";

import { SUPABASE_ANON_KEY, SUPABASE_URL } from "./config";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/** The current access token (a Supabase-signed JWT), or null if signed out.
 * Read from local storage; refreshed transparently by the client when stale. */
export async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

/** The current session, or null. Used at boot to decide login vs app. */
export async function getSession(): Promise<Session | null> {
  const { data } = await supabase.auth.getSession();
  return data.session;
}

/** Start the Google OAuth flow. Redirects away, then back to this origin. */
export async function signInWithGoogle(): Promise<void> {
  await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin },
  });
}

export async function signOut(): Promise<void> {
  await supabase.auth.signOut();
}

/** Subscribe to auth changes (initial session, sign-in, sign-out, refresh). */
export function onAuthChange(cb: (session: Session | null) => void): void {
  supabase.auth.onAuthStateChange((_event, session) => cb(session));
}
