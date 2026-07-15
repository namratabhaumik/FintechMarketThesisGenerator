// Shared types for the frontend.

import type { components } from "./types.gen";

export type JobResponse = components["schemas"]["JobResponse"];
export type ThesisResponse = components["schemas"]["ThesisResponse"];
export type SourceResponse = components["schemas"]["SourceResponse"];
export type RelatedThesisResponse = components["schemas"]["RelatedThesisResponse"];
export type ThesisSummaryResponse = components["schemas"]["ThesisSummaryResponse"];
export type ThesisRequest = components["schemas"]["ThesisRequest"];
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

// View -> controller callbacks.
export type RefineHandler = (jobId: string, feedback: string[]) => void;
export type ApproveHandler = (jobId: string) => void;
export type ResumeHandler = (jobId: string) => Promise<void>;
export type CompareHandler = (jobIds: string[]) => void;
export type DeleteHandler = (jobId: string) => void;