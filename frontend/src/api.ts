// Typed client for the FinThesis API. Request/response shapes come from
// types.gen.ts (generated from the FastAPI OpenAPI schema); this stays in
// lockstep with the backend. Errors carry the backend's {code, message}.

import { getAccessToken } from "./auth";
import { API_BASE } from "./config";
import type {
  JobResponse,
  RefinementRequest,
  ThesisRequest,
  ThesisSummaryResponse,
} from "./types";

/** fetch() with the Supabase access token attached as a Bearer header when the
 * user is signed in. Required: the backend verifies it and scopes every jobs
 * query to the caller via RLS; without it, job endpoints return 401. */
async function authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const token = await getAccessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(url, { ...init, headers });
}

/**
 * Machine-readable error codes the UI special-cases. Values must match the
 * codes emitted by the backend's `_error()` calls in routes.py. Codes not
 * listed here still surface via ApiError.message.
 */
export const ErrorCode = {
  NoRelevantDocuments: "no_relevant_documents",
  InsufficientEvidence: "insufficient_evidence",
} as const;

/** An API error carrying the backend's machine-readable code (see routes.py). */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

interface CodeMessage {
  code: string;
  message: string;
}

function isCodeMessage(detail: unknown): detail is CodeMessage {
  return (
    typeof detail === "object" &&
    detail !== null &&
    "code" in detail &&
    "message" in detail
  );
}

/** Turn a non-2xx Response into an ApiError, parsing both error shapes. */
async function toApiError(res: Response): Promise<ApiError> {
  let code = "error";
  let message = res.statusText || `Request failed (${res.status})`;
  try {
    const body: unknown = await res.json();
    const detail = (body as { detail?: unknown } | null)?.detail;
    if (isCodeMessage(detail)) {
      // Our custom errors: {detail: {code, message}}
      code = detail.code;
      message = detail.message;
    } else if (Array.isArray(detail) && detail.length > 0) {
      // FastAPI/pydantic 422: {detail: [{msg, ...}]}
      code = "validation_error";
      message = detail
        .map((d) => (d as { msg?: string }).msg)
        .filter(Boolean)
        .join("; ");
    }
  } catch {
    // Non-JSON error body; keep the status-based defaults.
  }
  return new ApiError(res.status, code, message);
}

/** Generate a thesis synchronously. Resolves once the job is persisted. */
export async function createThesis(query: string): Promise<JobResponse> {
  const payload: ThesisRequest = { query };
  const res = await authedFetch(`${API_BASE}/api/theses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as JobResponse;
}

/** Run one refinement round on a job, returning its updated state. */
export async function createRefinement(
  jobId: string,
  feedback: string[],
): Promise<JobResponse> {
  const payload: RefinementRequest = { feedback };
  const res = await authedFetch(
    `${API_BASE}/api/theses/${encodeURIComponent(jobId)}/refinements`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as JobResponse;
}

/** Approve a thesis (terminal, idempotent). Returns its updated state. */
export async function approveThesis(jobId: string): Promise<JobResponse> {
  const res = await authedFetch(
    `${API_BASE}/api/theses/${encodeURIComponent(jobId)}/approval`,
    { method: "PUT" },
  );
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as JobResponse;
}

/** Full state of one thesis job (for ?job_id restore and resume). */
export async function getThesis(jobId: string): Promise<JobResponse> {
  const res = await authedFetch(`${API_BASE}/api/theses/${encodeURIComponent(jobId)}`);
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as JobResponse;
}

/** Slim list of past jobs, most recent first (optionally filter by
 * refinement_status server-side ). */
export async function listTheses(
  limit = 20,
  status?: string,
): Promise<ThesisSummaryResponse[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  const res = await authedFetch(`${API_BASE}/api/theses?${params.toString()}`);
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as ThesisSummaryResponse[];
}

/** The fixed set of refinement feedback reasons the UI offers. */
export async function getFeedbackOptions(): Promise<string[]> {
  const res = await authedFetch(`${API_BASE}/api/feedback-options`);
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as string[];
}