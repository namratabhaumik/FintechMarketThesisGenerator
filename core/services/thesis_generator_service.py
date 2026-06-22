"""Thesis generator service."""

import logging
from collections import Counter
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from core.interfaces.llm import ILanguageModel
from core.models.thesis import StructuredThesis
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

logger = logging.getLogger(__name__)


def _ranked_tags_from_documents(
    documents: List[Document],
) -> Tuple[List[str], List[str], List[str]]:
    """Frequency-rank the Silver tags carried in the retrieved chunks' metadata.

    Returns the full ranked candidate list per dimension (themes, risks,
    signals), most common first and UNCAPPED. These are the only tags the thesis
    can ever surface - they come straight from the evidence, never invented.
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

    def _ranked(counter: Counter) -> List[str]:
        return [label for label, _ in counter.most_common()]

    return (
        _ranked(counters["themes"]),
        _ranked(counters["risks"]),
        _ranked(counters["signals"]),
    )


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
        max_tags_per_dimension: int = 3,
    ):
        """Initialize with dependencies.

        Args:
            llm: Injected LLM implementation (for summarization).
            scoring_service: Injected opportunity scoring service.
            max_tags_per_dimension: How many of the most common themes / risks /
                signals from the retrieved docs to surface per dimension.
        """
        self._llm = llm
        self._scoring_service = scoring_service
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

        # Step 4: Score opportunity (rule-based, no LLM)
        logger.info("Step 4: Scoring opportunity...")
        score_result = self._scoring_service.score_opportunity(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_text=summary
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
