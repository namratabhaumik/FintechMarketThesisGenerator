/// <reference types="vite/client" />

// Build-time env, injected by Vite (prod: CI sets these; dev: config.ts falls
// back). All optional so a bare local build still typechecks.
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_SUPABASE_URL?: string;
  readonly VITE_SUPABASE_ANON_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
