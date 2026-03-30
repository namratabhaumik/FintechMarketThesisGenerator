"""Pydantic schemas for the FastAPI layer."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Enums ---

class JobStatus(str, Enum):
    """Job lifecycle states."""
    PENDING = "pending"
    FETCHING_ARTICLES = "fetching_articles"
    BUILDING_VECTORSTORE = "building_vectorstore"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Request schemas ---

class ThesisRequest(BaseModel):
    """Request to generate a new thesis."""
    query: str = Field(..., min_length=1, max_length=500)


class RefinementRequest(BaseModel):
    """Request to refine an existing thesis."""
    feedback: List[str] = Field(..., min_length=1)


# --- Response schemas ---

class ArticleResponse(BaseModel):
    """Article summary for API responses."""
    title: str
    source: str
    url: Optional[str] = None


class ThesisResponse(BaseModel):
    """Structured thesis output."""
    key_themes: List[str] = []
    risks: List[str] = []
    investment_signals: List[str] = []
    sources: List[str] = []
    raw_output: Optional[str] = None
    opportunity_score: float = 0.0
    confidence_level: float = 0.0
    recommendation: str = ""
    key_risk_factors: List[str] = []


class JobResponse(BaseModel):
    """Job status and result."""
    job_id: str
    status: JobStatus
    progress: Optional[str] = None
    query: str
    thesis: Optional[ThesisResponse] = None
    articles: List[ArticleResponse] = []
    error: Optional[str] = None


class RefinementResponse(BaseModel):
    """Response after a refinement step."""
    job_id: str
    refinement_count: int
    status: str
    thesis: ThesisResponse
    execution_log: list = []
