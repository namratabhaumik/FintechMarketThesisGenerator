import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build outputs to dist/ (index.html + hashed assets/), which the deploy CI
// publishes wholesale to the deploy-dev/deploy-prod branches. Env values
// (VITE_API_BASE / VITE_SUPABASE_*) are read from the environment at build
// time; the CI sets them per environment, local dev falls back in config.ts.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: "127.0.0.1", port: 3000 },
  build: { outDir: "dist", emptyOutDir: true },
});
