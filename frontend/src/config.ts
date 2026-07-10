// Base URL of the FastAPI backend): prod CI sets the API_BASE env var to the
// deployed API origin; local builds fall back to localhost:8000 here.
declare const __API_BASE__: string | undefined;

export const API_BASE: string =
  typeof __API_BASE__ !== "undefined" ? __API_BASE__ : "http://localhost:8000";

// Supabase project, for Google-OAuth login. Both values are PUBLIC by design:
// the anon key is meant to ship in the browser bundle (Row Level Security, not
// key secrecy, protects the data). Prod CI injects them via build.mjs; the
// local-dev fallbacks below are what the dev server actually uses.
declare const __SUPABASE_URL__: string | undefined;
declare const __SUPABASE_ANON_KEY__: string | undefined;

export const SUPABASE_URL: string =
  typeof __SUPABASE_URL__ !== "undefined"
    ? __SUPABASE_URL__
    : "https://tabjqryubadxilgkdmvz.supabase.co";

export const SUPABASE_ANON_KEY: string =
  typeof __SUPABASE_ANON_KEY__ !== "undefined"
    ? __SUPABASE_ANON_KEY__
    : "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRhYmpxcnl1YmFkeGlsZ2tkbXZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU0MTczNDEsImV4cCI6MjA5MDk5MzM0MX0.G_aPZx2sYSF11-gd9Dv-oPkUxuzSk-TdYq5_xenfe2Q"; // <-- dev fallback: paste the anon/public key
