// Prod bundle build. esbuild strips types and bundles src/main.ts into
// dist/main.js, injecting the API base URL at build time.
//
// API_BASE comes from the env: the deploy CI sets it to the deployed API origin; 
// a local `npm run build` falls back to localhost.
// JSON.stringify produces a JS string literal for the define.

import esbuild from "esbuild";
import { copyFileSync } from "node:fs";
import { execSync } from "node:child_process";

const apiBase = process.env.API_BASE ?? "http://localhost:8000";
// Public Supabase values (anon key is safe to embed). Prod sets these in the
// deploy env; a local `npm run build` leaves the anon key empty and relies on
// the config.ts dev fallback instead.
const supabaseUrl = process.env.SUPABASE_URL ?? "https://tabjqryubadxilgkdmvz.supabase.co";
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY ?? "";

execSync(
  "npx @tailwindcss/cli -i src/styles.css -o dist/styles.css --minify",
  { stdio: "inherit" },
);

await esbuild.build({
  entryPoints: ["src/main.ts"],
  bundle: true,
  minify: true,
  format: "esm",
  outfile: "dist/main.js",
  define: {
    __API_BASE__: JSON.stringify(apiBase),
    __SUPABASE_URL__: JSON.stringify(supabaseUrl),
    __SUPABASE_ANON_KEY__: JSON.stringify(supabaseAnonKey),
  },
});

copyFileSync("index.html", "dist/index.html");
console.log(`Built dist/ (API_BASE=${apiBase})`);