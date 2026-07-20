"""Pydantic schemas for the FastAPI layer.

Response design follows REST conventions: one full resource representation
(JobResponse) returned by create/get/refine/approve so clients always see a
consistent shape, and a slim ThesisSummaryResponse for the list endpoint so
collections stay lightweight (no docs, histories, or embeddings).
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Enums ---

class RefinementStatus(str, Enum):
    """Refinement lifecycle of a thesis job. Single source of truth for the
    `refinement_status` values the API emits (and the frontend branches on)."""
    NOT_APPLICABLE = "N/A"  # generated, never refined
    REFINING = "refining"   # refined >=1 round, more possible; the only resumable state
    ESCALATED = "escalated"  # max refinements reached; terminal
    REFINED = "refined"     # refinement finalized (e.g. by approval); terminal


# --- Request schemas ---

class ThesisRequest(BaseModel):
    """Request to generate a new thesis."""
    query: str = Field(..., min_length=1, max_length=500)


class RefinementRequest(BaseModel):
    """Request to refine an existing thesis."""
    feedback: List[str] = Field(..., min_length=1)


# --- Response schemas ---

class SourceResponse(BaseModel):
    """A source article behind the thesis (from retrieved-doc metadata)."""
    title: str = "Untitled"
    url: Optional[str] = None
    published_at: Optional[str] = None
    # Query-to-chunk cosine similarity of the article's best retrieved chunk
    # (0-1), as match_documents defines it.
    similarity: Optional[float] = None


class ThesisResponse(BaseModel):
    """Structured thesis output."""
    key_themes: List[str] = []
    risks: List[str] = []
    investment_signals: List[str] = []
    sources: List[str] = []
    raw_output: Optional[str] = None
    opportunity_score: float = 0.0
    confidence_level: float = 0.0
    confidence_as_of: Optional[str] = None
    recommendation: str = ""
    key_risk_factors: List[str] = []
    # What produced raw_output: "llm" or "local" (extractive fallback).
    summary_source: str = "llm"
    # "refused" when the summarizer found the sources insufficient for the query.
    summary_status: str = "ok"
    # Why summary_status is "refused": "tag_strength_floor" or "llm_judgment".
    # None when summary_status is "ok".
    refusal_reason: Optional[str] = None


class RelatedThesisResponse(BaseModel):
    """A past run surfaced by episodic recall (query-to-query similarity)."""
    job_id: str
    query: str
    created_at: Optional[str] = None
    score: float
    recommendation: str
    approved: bool
    similarity: float


class ThesisSummaryResponse(BaseModel):
    """Slim list-item representation of a job."""
    job_id: str
    query: str
    created_at: Optional[str] = None
    refinement_count: int = 0
    refinement_status: RefinementStatus = RefinementStatus.NOT_APPLICABLE
    approved_at: Optional[str] = None
    opportunity_score: Optional[float] = None
    recommendation: Optional[str] = None
    # Owner of the job. Only meaningfully distinct from the caller for an
    # admin, who sees other users' jobs too (RLS admin policy).
    user_id: Optional[str] = None


class JobResponse(BaseModel):
    """Full job representation: result and refinement state."""
    job_id: str
    query: str
    created_at: Optional[str] = None
    thesis: Optional[ThesisResponse] = None
    thesis_history: List[ThesisResponse] = []
    refinement_count: int = 0
    refinement_status: RefinementStatus = RefinementStatus.NOT_APPLICABLE
    feedback_history: List[List[str]] = []
    execution_log: list = []
    approved_at: Optional[str] = None
    sources: List[SourceResponse] = []
    related_theses: List[RelatedThesisResponse] = []
    # Present only on refinement responses (transient, not stored).
    hallucination: Optional[dict] = None
    # Owner of the job. Only meaningfully distinct from the caller for an
    # admin, who sees other users' jobs too (RLS admin policy).
    user_id: Optional[str] = None
