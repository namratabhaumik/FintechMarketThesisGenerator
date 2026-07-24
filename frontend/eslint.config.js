import js from "@eslint/js";
import tseslint from "typescript-eslint";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";

// Flat config. The focus is security + React correctness. 
// typescript-eslint's type-checked rules run against tsconfig.json.
export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "src/types.gen.ts"] },
  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
      globals: { ...globals.browser },
    },
    plugins: { react, "react-hooks": reactHooks },
    settings: { react: { version: "detect" } },
    rules: {
      ...react.configs.flat.recommended.rules,
      ...react.configs.flat["jsx-runtime"].rules,
      ...reactHooks.configs.recommended.rules,

      // Security: the invariants we were holding by hand (no raw HTML sinks,
      // safe external links, no javascript: URLs). Enforced at edit time now.
      "react/no-danger": "error",
      "react/jsx-no-target-blank": ["error", { enforceDynamicLinks: "always" }],
      "react/jsx-no-script-url": "error",
      "no-eval": "error",
      "no-implied-eval": "error",
    },
  },
  // Config file itself isn't part of the app's TS project.
  {
    files: ["eslint.config.js", "vite.config.ts"],
    ...tseslint.configs.disableTypeChecked,
    languageOptions: { globals: { ...globals.node } },
  },
);