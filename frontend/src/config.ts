// Base URL of the FastAPI backend): prod CI sets the API_BASE env var to the
// deployed API origin; local builds fall back to localhost:8000 here.
declare const __API_BASE__: string | undefined;

export const API_BASE: string =
  typeof __API_BASE__ !== "undefined" ? __API_BASE__ : "http://localhost:8000";