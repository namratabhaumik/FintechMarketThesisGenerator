"""Thesis generator service."""

import logging
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from core.interfaces.llm import ILanguageModel
from core.interfaces.trend_repository import ITrendRepository
from core.models.thesis import StructuredThesis
from core.models.trend_metric import TrendMetric
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

logger = logging.getLogger(__name__)

# How many of the most recent Gold weeks count toward the confidence recency
# signal. A theme covered in more of these weeks reads as more current.
RECENCY_WEEKS = 4


def _tag_counters(documents: List[Document]) -> Dict[str, "Counter[str]"]:
    """Count Silver-tag occurrences per dimension across the retrieved chunks.

    Each chunk was tagged deterministically on its raw article text at Silver
    time. Missing metadata (e.g. older chunks embedded before tagging existed) is
    treated as no tags.
    """
    counters: Dict[str, "Counter[str]"] = {
        "themes": Counter(),
        "risks": Counter(),
        "signals": Counter(),
    }
    for doc in documents:
        metadata = doc.metadata or {}
        for key, counter in counters.items():
            # `or []` guards both an absent key and an explicit None.
            counter.update(metadata.get(key) or [])
    return counters


def _ranked_tags_from_documents(
    documents: List[Document],
) -> Tuple[List[str], List[str], List[str]]:
    """Frequency-rank the Silver tags carried in the retrieved chunks' metadata.

    Returns the full ranked candidate list per dimension (themes, risks,
    signals), most common first and UNCAPPED. These are the only tags the thesis
    can ever surface - they come straight from the evidence, never invented.
    """
    counters = _tag_counters(documents)

    def _ranked(counter: "Counter[str]") -> List[str]:
        return [label for label, _ in counter.most_common()]

    return (
        _ranked(counters["themes"]),
        _ranked(counters["risks"]),
        _ranked(counters["signals"]),
    )


def _tag_strengths_from_documents(
    documents: List[Document],
) -> Tuple[int, int, int]:
    """Total Silver-tag occurrences per dimension across the retrieved chunks.

    Returns (signal_strength, theme_strength, risk_strength) - the grounded
    analog of the old keyword-hit counts, now summed from the Silver tags rather
    than scanned from the LLM prose. Computed from the FULL tags (uncapped), so
    the score is independent of how many tags are displayed and is identical for
    the same retrieved set - which is what keeps it stable across refinement.
    """
    counters = _tag_counters(documents)
    return (
        sum(counters["signals"].values()),
        sum(counters["themes"].values()),
        sum(counters["risks"].values()),
    )


def _gold_confidence_inputs(
    documents: List[Document],
    metrics: List[TrendMetric],
    recency_weeks: int,
) -> Tuple[int, int, int, Optional[date]]:
    """Derive Gold-based confidence signals for the thesis's categories.

    Matches the FULL set of (dimension, category) tags the retrieved evidence
    carries - across all three dimensions - against the Gold trend metrics, and
    returns (coverage_count, recent_weeks_covered, recency_window, as_of):
      - coverage_count: total Gold coverage (summed article_count) logged for
        those categories - the depth of corpus support.
      - recent_weeks_covered: how many of the last `recency_weeks` Gold weeks had
        any such coverage - how current that support is.
      - as_of: the latest week present in Gold (data freshness), or None if Gold
        is empty.

    Uses the uncapped evidence tags, so the result is independent of how many
    tags are displayed and is stable across a refinement session.
    """
    counters = _tag_counters(documents)
    evidence = (
        {("theme", c) for c in counters["themes"]}
        | {("risk", c) for c in counters["risks"]}
        | {("signal", c) for c in counters["signals"]}
    )
    if not metrics:
        return 0, 0, recency_weeks, None

    as_of = max(m.week_start for m in metrics)
    matching = [m for m in metrics if (m.dimension, m.category) in evidence]
    coverage_count = sum(m.article_count for m in matching)
    recent_weeks = {as_of - timedelta(weeks=i) for i in range(recency_weeks)}
    recent_weeks_covered = len(
        {m.week_start for m in matching if m.week_start in recent_weeks}
    )
    return coverage_count, recent_weeks_covered, recency_weeks, as_of


def _apply_feedback_caps(
    themes: List[str],
    risks: List[str],
    signals: List[str],
    feedback_items: List[str],
    base_cap: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Apply the QUANTITATIVE effect of feedback to the grounded tags.

    Every feedback item also drives the LLM narrative rewrite (the qualitative
    effect, handled by llm.refine). On top of that, feedback about the SHAPE of
    the structured output adjusts how many grounded tags surface per dimension. 
    Eg: "Sharpen" feedback surfaces MORE concrete grounded tags (+1, capped
    by what the evidence has); "too many" trims to the best-supported (-1).
    Because the score is computed from these tags downstream, it tracks the
    feedback too.
    """
    text = " ".join(feedback_items).lower()
    theme_cap = risk_cap = signal_cap = base_cap

    # "too broad / be more specific" -> sharpen every dimension by surfacing more
    # concrete grounded specifics (bounded by the evidence).
    if "broad" in text or "specific" in text:
        theme_cap = risk_cap = signal_cap = base_cap + 1
    # "too many risks" -> trim risks to the best-supported.
    if "too many" in text and "risk" in text:
        risk_cap = max(1, base_cap - 1)
    # "signals too vague" -> sharpen signals by surfacing more concrete grounded ones.
    if "vague" in text and "signal" in text:
        signal_cap = base_cap + 1
    # "not enough opportunities / score too low" -> surface one more grounded
    # signal (which also lifts the downstream score).
    if "opportunit" in text or "score" in text:
        signal_cap = base_cap + 1
    # "missing recent trends" -> widen theme coverage by one.
    if "recent" in text or "trend" in text:
        theme_cap = base_cap + 1

    return themes[:theme_cap], risks[:risk_cap], signals[:signal_cap]


class ThesisGeneratorService:
    """Service for generating market theses.

    Single Responsibility: Thesis generation orchestration. The LLM writes the
    narrative; the structured themes/risks/signals are derived deterministically
    from the retrieved docs' Silver tags
    """

    def __init__(
        self,
        llm: ILanguageModel,
        scoring_service: OpportunityScoringService,
        trend_repository: ITrendRepository,
        max_tags_per_dimension: int = 3,
    ):
        """Initialize with dependencies.

        Args:
            llm: Injected LLM implementation (for summarization).
            scoring_service: Injected opportunity scoring service.
            trend_repository: Gold-layer trend store, read to ground confidence in
                corpus coverage depth + recency.
            max_tags_per_dimension: How many of the most common themes / risks /
                signals from the retrieved docs to surface per dimension.
        """
        self._llm = llm
        self._scoring_service = scoring_service
        self._trend_repository = trend_repository
        self._max_tags = max_tags_per_dimension

    def generate_thesis(
        self,
        topic: str,
        documents: List[Document]
    ) -> StructuredThesis:
        """Generate structured thesis from documents.

        Args:
            topic: Market topic for analysis.
            documents: Retrieved context documents.

        Returns:
            StructuredThesis object with analysis results.
        """
        logger.info(f"Generating thesis for topic: {topic}")

        # Step 1: Summarize documents. This is the ONLY LLM call - it writes the
        # narrative prose (raw_output); it does not decide the structured tags.
        logger.info("Step 1: Summarizing retrieved documents...")
        summary = self._llm.summarize(documents)

        if not summary:
            logger.error("Empty summary returned by LLM")
            raise RuntimeError("Failed to generate summary")

        # Step 2: Derive the structured tags - still deterministic, no LLM.
        # Previously this keyword-scanned the LLM summary above; now it
        # frequency-ranks the Silver tags already carried in each retrieved
        # chunk's metadata (matched on the raw article text at Silver time). Same
        # determinism, but grounded in what the source articles reported rather
        # than in the LLM's prose.
        logger.info("Step 2: Deriving grounded tags from retrieved documents...")
        ranked_themes, ranked_risks, ranked_signals = _ranked_tags_from_documents(
            documents
        )
        key_themes = ranked_themes[: self._max_tags]
        risks = ranked_risks[: self._max_tags]
        investment_signals = ranked_signals[: self._max_tags]

        # Step 3: Extract sources from document metadata
        logger.info("Step 3: Extracting sources from documents...")
        sources = [doc.metadata["url"] for doc in documents if doc.metadata.get("url")]

        # Step 4: Score from the grounded Silver tag strengths; ground confidence
        # in Gold trend coverage (depth + recency) for the thesis's categories.
        # Both numbers are functions of the retrieved evidence + the Gold snapshot
        # - not the LLM prose - so they are reproducible and stable across
        # refinement. Strengths and the matched categories come from the FULL
        # (uncapped) tags. The as-of date is the latest Gold week (data freshness).
        logger.info("Step 4: Scoring (score<-Silver strengths, confidence<-Gold trends)...")
        signal_strength, theme_strength, risk_strength = _tag_strengths_from_documents(
            documents
        )
        coverage_count, recent_weeks_covered, recency_window, as_of = (
            _gold_confidence_inputs(documents, self._trend_repository.fetch_all(), RECENCY_WEEKS)
        )
        score_result = self._scoring_service.score_opportunity(
            risks=risks,
            signal_strength=signal_strength,
            theme_strength=theme_strength,
            risk_strength=risk_strength,
            coverage_count=coverage_count,
            recent_weeks_covered=recent_weeks_covered,
            recency_window=recency_window,
        )

        logger.info("Successfully generated structured thesis with scoring")
        return StructuredThesis(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_output=summary,
            opportunity_score=score_result["score"],
            confidence_level=score_result["confidence_level"],
            confidence_as_of=as_of,
            recommendation=score_result["recommendation"],
            key_risk_factors=score_result["key_risks"]
        )

    def refine_thesis(
        self,
        topic: str,
        documents: List[Document],
        current_thesis: StructuredThesis,
        feedback_items: List[str],
    ) -> StructuredThesis:
        """Refine an existing thesis based on user feedback.

        Calls the LLM for a feedback-aware rewrite, then runs the same
        structuring as generate_thesis; assemble_node owns the single 
        deterministic scoring step, keeping the scoring service the 
        sole authority for scores.

        Args:
            topic: Original market topic for analysis.
            documents: Retrieved context documents.
            current_thesis: The thesis to refine.
            feedback_items: Predefined feedback strings selected by user.

        Returns:
            New StructuredThesis with refinements applied.
        """
        logger.info(f"Refining thesis for topic: {topic} based on {len(feedback_items)} feedback items")

        # Step 1: Get refined summary from LLM using feedback
        current_thesis_text = current_thesis.raw_output or ""
        logger.info("Step 1: Refining thesis with LLM feedback...")
        refined_summary = self._llm.refine(documents, current_thesis_text, feedback_items)

        if not refined_summary:
            logger.error("Empty refined summary returned by LLM")
            raise RuntimeError("Failed to refine thesis")

        # Step 2: Derive grounded tags from the retrieved docs; apply the
        # qualitative + quantitative effects of the feedback.
        logger.info("Step 2: Deriving feedback-adjusted grounded tags...")
        ranked_themes, ranked_risks, ranked_signals = _ranked_tags_from_documents(
            documents
        )
        key_themes, risks, investment_signals = _apply_feedback_caps(
            ranked_themes, ranked_risks, ranked_signals, feedback_items, self._max_tags
        )

        # Step 3: Extract sources from document metadata (same as generate_thesis)
        logger.info("Step 3: Extracting sources from documents...")
        sources = [doc.metadata["url"] for doc in documents if doc.metadata.get("url")]

        logger.info("Successfully refined thesis content (scoring handled downstream)")
        return StructuredThesis(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_output=refined_summary,
        )
