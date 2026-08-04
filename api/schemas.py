"""Pydantic schemas for the FastAPI layer.

Response design follows REST conventions: one full resource representation
(JobResponse) returned by create/get/refine/approve so clients always see a
consistent shape, and a slim ThesisSummaryResponse for the list endpoint so
collections stay lightweight (no docs, histories, or embeddings).
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


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
    # The fraction confidence_level is, so a client can render "17 of 47 weeks"
    # instead of a bare decimal. Frozen across refinement alongside the ratio.
    covered_weeks: int = 0
    window_weeks: int = 0
    confidence_as_of: Optional[str] = None
    recommendation: str = ""
    key_risk_factors: List[str] = []
    # Citation trail: dimension -> displayed tag -> the source URLs behind it.
    # Theses generated before this existed omit it, hence the empty default.
    tag_sources: Dict[str, Dict[str, List[str]]] = {}
    # "lift" (ranked against corpus base rates) or "count" (raw frequency, the
    # fallback when base rates are unavailable).
    tags_ranked_by: str = "count"
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


# --- Annotations ---

class AnnotationSection(str, Enum):
    """Which block of a thesis an annotation's offsets index into. Mirrors the
    annotations_section_valid constraint in sql/annotations.sql; the section
    keeps a note on Risks from ever resolving against the Raw Summary."""
    RAW_SUMMARY = "raw_summary"
    KEY_THEMES = "key_themes"
    RISKS = "risks"
    INVESTMENT_SIGNALS = "investment_signals"


class AnnotationResolution(str, Enum):
    """Thread outcome. Absent means still open. Single-valued today (the tick);
    kept as an enum rather than a boolean so another outcome is an added member
    rather than a column migration."""
    ACCEPTED = "accepted"


class AnnotationAuthor(BaseModel):
    """Display identity for a comment, resolved from the profiles table. Both
    fields may be null: a profile row can lag a brand-new signup, and a nameless
    author must not break the thread."""
    user_id: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None


class AnnotationCreateRequest(BaseModel):
    """Create a root annotation (anchored) or a reply (parent_id set).

    Anchors are pinned to one version: offsets are only meaningful against that
    version's text, which is frozen once superseded. The server takes `version`
    from the request because the client knows which version it rendered.
    """
    body: str = Field(..., min_length=1)
    # Root only. A reply carries none of these.
    version: Optional[int] = Field(None, gt=0)
    section: Optional[AnnotationSection] = None
    start_offset: Optional[int] = Field(None, ge=0)
    end_offset: Optional[int] = Field(None, gt=0)
    quote: Optional[str] = None
    # Reply only.
    parent_id: Optional[str] = None

    @model_validator(mode="after")
    def check_range(self):
        """An inverted or empty range marks nothing. The DB rejects it too
        (annotations_range_valid), but that arrives as a 500; validating here
        makes it the 400 it actually is."""
        if self.start_offset is not None and self.end_offset is not None:
            if self.end_offset <= self.start_offset:
                raise ValueError("end_offset must be greater than start_offset")
        return self


class AnnotationUpdateRequest(BaseModel):
    """Edit a comment's text. Author-only; resolution is a separate endpoint
    because it is a state change other participants may make."""
    body: str = Field(..., min_length=1)


class AnnotationResolveRequest(BaseModel):
    """Tick or reopen. Null reopens."""
    resolution: Optional[AnnotationResolution] = None


class AnnotationResponse(BaseModel):
    """One annotation - a root with its anchor, or a reply."""
    id: str
    job_id: str
    version: int
    body: str
    section: Optional[AnnotationSection] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    quote: Optional[str] = None
    parent_id: Optional[str] = None
    resolution: Optional[AnnotationResolution] = None
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    author: AnnotationAuthor
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
