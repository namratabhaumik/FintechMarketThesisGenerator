// Prod bundle build. esbuild strips types and bundles src/main.ts into
// dist/main.js, injecting the API base URL at build time.
//
// API_BASE comes from the env: the deploy CI sets it to the deployed API origin; 
// a local `npm run build` falls back to localhost.
// JSON.stringify produces a JS string literal for the define.

import esbuild from "esbuild";
import { copyFileSync } from "node:fs";

const apiBase = process.env.API_BASE ?? "http://localhost:8000";

await esbuild.build({
  entryPoints: ["src/main.ts"],
  bundle: true,
  minify: true,
  format: "esm",
  outfile: "dist/main.js",
  define: { __API_BASE__: JSON.stringify(apiBase) },
});

copyFileSync("index.html", "dist/index.html");
console.log(`Built dist/ (API_BASE=${apiBase})`);