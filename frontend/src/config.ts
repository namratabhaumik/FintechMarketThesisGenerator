// Base URL of the FastAPI backend. Prod CI injects VITE_API_BASE at build time
// (see .github/workflows/ci.yml); local dev falls back to localhost:8000 here.
export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// Supabase project, for Google-OAuth login. Both values are PUBLIC by design:
// the anon key is meant to ship in the browser bundle (Row Level Security, not
// key secrecy, protects the data). Prod CI injects VITE_SUPABASE_* at build
// time; the local-dev fallbacks below are what the dev server actually uses.
export const SUPABASE_URL: string =
  import.meta.env.VITE_SUPABASE_URL ?? "https://tabjqryubadxilgkdmvz.supabase.co";

export const SUPABASE_ANON_KEY: string =
  import.meta.env.VITE_SUPABASE_ANON_KEY ??
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRhYmpxcnl1YmFkeGlsZ2tkbXZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU0MTczNDEsImV4cCI6MjA5MDk5MzM0MX0.G_aPZx2sYSF11-gd9Dv-oPkUxuzSk-TdYq5_xenfe2Q"; // <-- dev fallback: paste the anon/public key
