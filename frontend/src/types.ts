// Shared types for the frontend.

import type { components } from "./types.gen";

export type JobResponse = components["schemas"]["JobResponse"];
export type ThesisResponse = components["schemas"]["ThesisResponse"];
export type SourceResponse = components["schemas"]["SourceResponse"];
export type RelatedThesisResponse = components["schemas"]["RelatedThesisResponse"];
export type ThesisSummaryResponse = components["schemas"]["ThesisSummaryResponse"];
export type ThesisRequest = components["schemas"]["ThesisRequest"];
export type AnnotationResponse = components["schemas"]["AnnotationResponse"];
export type AnnotationCreateRequest = components["schemas"]["AnnotationCreateRequest"];
export type AnnotationSection = components["schemas"]["AnnotationSection"];
export type AnnotationResolution = components["schemas"]["AnnotationResolution"];
export type AnnotationAuthor = components["schemas"]["AnnotationAuthor"];
export type RefinementRequest = components["schemas"]["RefinementRequest"];

// Generated union of the backend's refinement_status values ("N/A" | ...).
export type RefinementStatus = components["schemas"]["RefinementStatus"];

/**
 * Named refinement-status values the UI branches on.
 */
export const RefinementStatus = {
  Refining: "refining",
  Escalated: "escalated",
} satisfies Record<string, RefinementStatus>;

// execution_log is unknown[] and hallucination is an open dict
export interface ExecutionEvent {
  tool_name?: string;
  status?: string;
  refinement_number?: number;
  reason?: string;
  changes?: string[];
}

export interface HallucinationAnalysis {
  invalid_tools?: string[];
  summary?: string;
}

/** Signed-in user info + sign-out handler, passed in by the auth gate. */
export interface AuthInfo {
  email?: string | null;
  /** The signed-in user's id. Annotations are authored by id, so the UI needs
   * it to tell the caller's own notes (editable) from everyone else's. */
  userId?: string | null;
  isAdmin?: boolean;
  onSignOut: () => void;
}

/** The single app-wide status line; errors get the alert styling. */
export interface StatusMessage {
  text: string;
  isError: boolean;
}

// View -> controller callbacks.
export type RefineHandler = (jobId: string, feedback: string[]) => void;
export type ApproveHandler = (jobId: string) => void;
export type ResumeHandler = (jobId: string) => Promise<void>;
export type CompareHandler = (jobIds: string[]) => void;
export type DeleteHandler = (jobId: string) => void;