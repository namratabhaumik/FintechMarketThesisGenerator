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

/**
 * Machine-readable error codes the UI special-cases. The first two must match
 * the codes emitted by the backend's `_error()` calls in routes.py; the rest
 * are raised client-side by this module. Codes not listed here still surface
 * via ApiError.message.
 */
export const ErrorCode = {
  NoRelevantDocuments: "no_relevant_documents",
  InsufficientEvidence: "insufficient_evidence",
  /** The Supabase session could not be read, or the backend rejected the JWT. */
  SessionExpired: "session_expired",
  /** fetch() itself failed: offline, DNS, or a CORS-blocked response. */
  NetworkError: "network_error",
  /** A 2xx response whose body would not parse as JSON. */
  MalformedResponse: "malformed_response",
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
  // A 401 the backend didn't label means the JWT was missing or rejected, which
  // the UI explains as an expired session rather than a generic failure.
  if (res.status === 401 && code === "error") {
    code = ErrorCode.SessionExpired;
  }
  return new ApiError(res.status, code, message);
}

/** fetch() with the Supabase access token attached as a Bearer header when the
 * user is signed in. Required: the backend verifies it and scopes every jobs
 * query to the caller via RLS; without it, job endpoints return 401. */
async function authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  let token: string | null;
  try {
    token = await getAccessToken();
  } catch (err) {
    // Reading the session can fail on an expired refresh token or blocked
    // storage. Without this the raw Supabase error would reach the UI and be
    // reported as an API outage.
    console.error("Could not read the Supabase session", err);
    throw new ApiError(401, ErrorCode.SessionExpired, "Your session could not be verified.");
  }
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  try {
    return await fetch(url, { ...init, headers });
  } catch (err) {
    // fetch() rejects with a TypeError when offline, on DNS failure, or when a
    // CORS-blocked response hides the real status.
    console.error("Request to the API failed", err);
    throw new ApiError(
      0,
      ErrorCode.NetworkError,
      "Could not reach the API. Check your connection and try again.",
    );
  }
}

/**
 * Run an authed request and parse its JSON body. This is the module's error
 * boundary: every failure path - session, network, HTTP status, unparseable
 * body - leaves here as an ApiError, so callers only ever handle one type.
 */
async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await authedFetch(url, init);
  if (!res.ok) {
    throw await toApiError(res);
  }
  try {
    return (await res.json()) as T;
  } catch (err) {
    console.error("Could not parse the API response body", err);
    throw new ApiError(
      res.status,
      ErrorCode.MalformedResponse,
      "The API returned a response that could not be read.",
    );
  }
}

/** As requestJson, for endpoints that return no body. */
async function requestVoid(url: string, init?: RequestInit): Promise<void> {
  const res = await authedFetch(url, init);
  if (!res.ok) {
    throw await toApiError(res);
  }
}

const JSON_HEADERS = { "Content-Type": "application/json" };

/** Generate a thesis synchronously. Resolves once the job is persisted. */
export function createThesis(query: string): Promise<JobResponse> {
  const payload: ThesisRequest = { query };
  return requestJson<JobResponse>(`${API_BASE}/api/theses`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

/** Run one refinement round on a job, returning its updated state. */
export function createRefinement(jobId: string, feedback: string[]): Promise<JobResponse> {
  const payload: RefinementRequest = { feedback };
  return requestJson<JobResponse>(
    `${API_BASE}/api/theses/${encodeURIComponent(jobId)}/refinements`,
    {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    },
  );
}

/** Approve a thesis (terminal, idempotent). Returns its updated state. */
export function approveThesis(jobId: string): Promise<JobResponse> {
  return requestJson<JobResponse>(
    `${API_BASE}/api/theses/${encodeURIComponent(jobId)}/approval`,
    { method: "PUT" },
  );
}

/** Full state of one thesis job (for ?job_id restore and resume). */
export function getThesis(jobId: string): Promise<JobResponse> {
  return requestJson<JobResponse>(`${API_BASE}/api/theses/${encodeURIComponent(jobId)}`);
}

/** Slim list of past jobs, most recent first. Scoped to the caller's own jobs
 * unless allUsers=true (admin only; the backend 403s for non-admins). Optionally
 * filter by refinement_status server-side. */
export function listTheses(
  limit = 20,
  offset = 0,
  status?: string,
  allUsers = false,
): Promise<ThesisSummaryResponse[]> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) params.set("status", status);
  if (allUsers) params.set("all", "true");
  return requestJson<ThesisSummaryResponse[]>(`${API_BASE}/api/theses?${params.toString()}`);
}

/** Delete a thesis job (admin only). */
export function deleteThesis(jobId: string): Promise<void> {
  return requestVoid(`${API_BASE}/api/theses/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
}

/** The fixed set of refinement feedback reasons the UI offers. */
export function getFeedbackOptions(): Promise<string[]> {
  return requestJson<string[]>(`${API_BASE}/api/feedback-options`);
}