"""Thesis data models."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RankedTag:
    """One Silver tag as ranked for display, with the evidence behind it.

    `lift` is the tag's share of the evidence pool divided by its share of the
    whole corpus: 1.0 --> as common here as everywhere, 3.0 --> three times
    over-represented in this query's evidence. Ranking on lift rather than
    `article_count` is what stops a wide pool (a large slice of the corpus) from
    surfacing the corpus's most common tags on every query. 0.0 means no corpus
    base rate was available, which is the signal to fall back to count ranking
    rather than trust a meaningless ratio.

    `sources` are the URLs of the pool articles carrying this tag - the citation
    trail behind why it surfaced.
    """

    label: str
    article_count: int          # Pool articles carrying it (one vote per article).
    lift: float                 # Pool share / corpus share; 0.0 when unknown.
    sources: List[str] = field(default_factory=list)


@dataclass
class StructuredThesis:
    """Structured thesis output."""
    key_themes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    investment_signals: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    opportunity_score: float = 0.0
    confidence_level: float = 0.0
    # The fraction behind confidence_level, so the UI can show "17 of 47 weeks"
    # next to the percentage instead of a bare decimal. covered_weeks counts the
    # Gold weeks in which the corpus discussed any of the DISPLAYED tags;
    # window_weeks is the retrieval window capped at the span Gold holds.
    covered_weeks: int = 0
    window_weeks: int = 0
    # Latest Gold week the confidence was computed against (data freshness).
    confidence_as_of: Optional[date] = None
    # Citation trail: dimension ("themes"/"risks"/"signals") --> displayed tag
    # --> the source URLs carrying it. Lets the UI answer "why this tag?" without
    # re-deriving it from the stored docs.
    tag_sources: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # How the displayed tags were ranked: "lift" (against corpus base rates),
    # "count" (raw frequency, the fallback), or "mixed" when the dimensions
    # disagree.
    tags_ranked_by: str = "count"
    recommendation: str = ""
    key_risk_factors: List[str] = field(default_factory=list)
    # What produced raw_output: "llm" (Gemini narrative) or "local" (extractive
    # fallback - no LLM). Lets the UI mark degraded summaries.
    summary_source: str = "llm"
    # "refused" when the summarizer (LLM or local) found the sources
    # insufficient/off-topic for the query instead of writing a summary.
    summary_status: str = "ok"
    # Why summary_status is "refused": "tag_strength_floor" (deterministic)
    # or "llm_judgment". None when summary_status is "ok".
    refusal_reason: Optional[str] = None
